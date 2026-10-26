"""Push support-calibrated residual uncertainty through a fitted quantizer.

This is an explicit observation model, separate from r101 pixel MSE.
No query labels, operator identities, learned fusion weights or tuning are used.
"""
import math
import numpy as np
import torch
from history_recovery.chainprint_support_replay import SupportReplay


class QuantizedReplayLikelihood(SupportReplay):
    def interval_log_probability(self, lower, upper, mean, sigma):
        lo, hi = (lower - mean) / sigma, (upper - mean) / sigma
        # Use the smaller CDF tail to avoid subtracting two numbers near one.
        flip = lo + hi > 0
        a = torch.where(flip, -hi, lo)
        b = torch.where(flip, -lo, hi)
        la, lb = torch.special.log_ndtr(a), torch.special.log_ndtr(b)
        difference = (la - lb).clamp(max=-torch.finfo(mean.dtype).eps)
        value = lb + torch.log(-torch.expm1(difference))
        return value.clamp(min=math.log(torch.finfo(mean.dtype).tiny))

    @torch.no_grad()
    def prepare(self, *, query_before, support_before, support_after, candidates):
        fitted, diagnostics = self.fit(support_before, support_after)
        sb, qb = self.image(support_before), self.image(query_before)
        sigmas = []
        for op, after, diag in zip(fitted, support_after, diagnostics):
            residual = (self.apply(sb, op) - self.image(after)) * 255
            sigma = max(float(residual.square().mean().sqrt()), 1 / math.sqrt(12))
            sigmas.append(sigma)
            diag['support_residual_sigma_8bit'] = sigma
        states = []
        for a, b in candidates:
            if {fitted[a][0], fitted[b][0]} != {'spatial', 'pointwise'}:
                raise ValueError('This restricted likelihood needs exactly one spatial and one pointwise step')
            first = self.apply(qb, fitted[a])
            if fitted[b][0] == 'pointwise':
                lut = self.luts[fitted[b][1]].long()
                lower = torch.full((256,), float('nan'), device=self.device, dtype=qb.dtype)
                upper = lower.clone()
                for value in lut.unique():
                    inputs = torch.where(lut == value)[0]
                    left, right = int(inputs[0]), int(inputs[-1])
                    lower[value] = -float('inf') if left == 0 else left - .5
                    upper[value] = float('inf') if right == 255 else right + .5
                states.append((first * 255, sigmas[a], lower, upper))
            else:
                centers = torch.arange(256, device=self.device, dtype=qb.dtype)
                lower, upper = centers - .5, centers + .5
                lower[0], upper[-1] = -float('inf'), float('inf')
                states.append((self.apply(first, fitted[b]) * 255, sigmas[b], lower, upper))
        return states, diagnostics

    @torch.no_grad()
    def compare_states(self, query_after, states):
        values = np.asarray(query_after)
        if values.ndim == 3:
            values = values[None]
        after = torch.stack([self.image(v) for v in values])
        indices = (after * 255).round().long()
        results = []
        for mean, sigma, lower, upper in states:
            lo, hi = lower[indices], upper[indices]
            valid = ~torch.isnan(lo)
            # Structural zero: output absent from fitted quantizer's alphabet.
            lp = self.interval_log_probability(torch.nan_to_num(lo, nan=0, neginf=-float('inf')),
                torch.nan_to_num(hi, nan=0, posinf=float('inf')), mean[None], sigma)
            lp = torch.where(valid, lp, math.log(torch.finfo(mean.dtype).tiny))
            results.append(lp.mean((-1, -2, -3)))
        return torch.stack(results, -1).cpu().numpy()

    @torch.no_grad()
    def score(self, *, query_before, query_after, support_before, support_after, candidates):
        states, _ = self.prepare(query_before=query_before, support_before=support_before,
                                 support_after=support_after, candidates=candidates)
        return self.compare_states(query_after, states)
