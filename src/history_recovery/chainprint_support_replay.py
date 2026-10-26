"""Restricted, support-fitted pixel operators; no operation metadata is accepted."""
import numpy as np
import torch
from torch.nn import functional as F


class SupportReplay:
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        self.orbits = {}
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                key = tuple(sorted((abs(dy), abs(dx))))
                if key != (0, 0):
                    self.orbits.setdefault(key, []).append((dy, dx))
        self.orbits = [self.orbits[k] for k in sorted(self.orbits)]
        assert len(self.orbits) == 14
        # Generic quantizer family, including the renderer's integer conversion.
        self.luts = torch.tensor([
            [int(round(v / (255.0 / (levels - 1))) * (255.0 / (levels - 1)))
             for v in range(256)] for levels in range(2, 257)
        ], dtype=torch.float64, device=self.device)

    def image(self, value):
        result = torch.tensor(np.asarray(value), device=self.device, dtype=torch.float64)
        if result.ndim != 3 or result.shape[0] != 3 or not torch.isfinite(result).all():
            raise ValueError('finite CHW RGB image required')
        if result.min() < 0 or result.max() > 1:
            raise ValueError('pixels must be in [0,1]')
        return result

    def basis(self, value):
        h, w = value.shape[-2:]
        padded = F.pad(value[None], (4, 4, 4, 4), mode='replicate')[0]
        return torch.stack([
            sum(padded[:, 4+dy:4+dy+h, 4+dx:4+dx+w] - value for dy, dx in orbit)
            for orbit in self.orbits
        ])

    def spatial_fit(self, before, after, basis, mask):
        mask = mask & (after > 0) & (after < 1)
        n = int(mask.sum())
        if n == 0:
            return torch.zeros(14, dtype=before.dtype, device=self.device)
        x = basis[:, mask]
        y = (after - before)[mask]
        gram = x @ x.T / n
        rhs = x @ y / n
        return torch.linalg.solve(gram + 1e-3 * torch.eye(14, device=self.device, dtype=before.dtype), rhs)

    @staticmethod
    def spatial_predict(before, basis, coefficients):
        result = before + torch.einsum('k,kchw->chw', coefficients, basis)
        return (result.clamp(0, 1) * 255).round() / 255

    def quantizer_fit(self, before, after, mask):
        index = (before[mask] * 255).round().long()
        target = (after[mask] * 255).round()
        # CPU bincount avoids nondeterministic CUDA weighted atomic additions.
        cpu_index = index.cpu().numpy()
        count = torch.tensor(np.bincount(cpu_index, minlength=256), device=self.device, dtype=before.dtype)
        sums = torch.tensor(np.bincount(cpu_index, weights=target.cpu().numpy(), minlength=256),
                            device=self.device, dtype=before.dtype)
        # The omitted target-square term is identical across all hypotheses.
        errors = (self.luts.square() * count - 2 * self.luts * sums).sum(-1)
        return int(errors.argmin())

    def quantizer_predict(self, before, index):
        return self.luts[index][(before * 255).round().long()] / 255

    @torch.no_grad()
    def fit(self, support_before, support_after):
        before = self.image(support_before)
        after = [self.image(a) for a in support_after]
        if len(after) != 4 or any(a.shape != before.shape for a in after):
            raise ValueError('four aligned support outputs required')
        h, w = before.shape[-2:]
        rows = torch.arange(h, device=self.device)[:, None]
        cols = torch.arange(w, device=self.device)[None, :]
        fitmask = ((rows + cols) % 4 != 0).expand_as(before)
        allmask = torch.ones_like(fitmask)
        basis = self.basis(before)
        fitted, diagnostics = [], []
        for target in after:
            coef = self.spatial_fit(before, target, basis, fitmask)
            qi = self.quantizer_fit(before, target, fitmask)
            spatial = self.spatial_predict(before, basis, coef)
            pointwise = self.quantizer_predict(before, qi)
            se = float(((spatial[~fitmask] - target[~fitmask]) * 255).square().mean())
            qe = float(((pointwise[~fitmask] - target[~fitmask]) * 255).square().mean())
            if se <= qe:
                coef = self.spatial_fit(before, target, basis, allmask)
                fitted.append(('spatial', coef))
                params = {'coefficients': coef.cpu().tolist()}
            else:
                qi = self.quantizer_fit(before, target, allmask)
                fitted.append(('pointwise', qi))
                params = {'inferred_levels': qi + 2}
            diagnostics.append({'selected': fitted[-1][0], 'spatial_validation_mse_8bit': se,
                                'pointwise_validation_mse_8bit': qe, **params})
        return fitted, diagnostics

    def apply(self, before, operator):
        kind, value = operator
        if kind == 'pointwise':
            return self.quantizer_predict(before, value)
        return self.spatial_predict(before, self.basis(before), value)

    @torch.no_grad()
    def predict(self, query_before, fitted, candidates):
        before = self.image(query_before)
        if len(candidates) != 4 or any(len(w) != 2 or any(g not in range(4) for g in w) for w in candidates):
            raise ValueError('four anonymous length-two candidates required')
        first = {word[0]: self.apply(before, fitted[word[0]]) for word in candidates}
        return torch.stack([self.apply(first[a], fitted[b]) for a, b in candidates])

    def compare(self, query_after, predictions):
        values = np.asarray(query_after)
        if values.ndim == 3:
            values = values[None]
        after = torch.stack([self.image(v) for v in values])
        # Fixed 8-bit pixel MSE units: numerical tie tolerance has a recorded scale.
        return -((after[:, None] - predictions[None]) * 255).square().mean((-1, -2, -3)).cpu().numpy()

    @torch.no_grad()
    def score(self, *, query_before, query_after, support_before, support_after, candidates):
        fitted, _ = self.fit(support_before, support_after)
        return self.compare(query_after, self.predict(query_before, fitted, candidates))
