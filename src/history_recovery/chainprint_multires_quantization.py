"""A fixed half-resolution chroma hypothesis, inferred from support pixels."""
import torch
from .chainprint_block_quantization import BlockQuantizationLikelihood


class MultiResolutionQuantizationLikelihood(BlockQuantizationLikelihood):
    def __init__(self, device='cuda:0'):
        super().__init__(device)
        self._interpolation = {}

    def interpolation(self, size):
        if size not in self._interpolation:
            position = (torch.arange(2 * size, device=self.device, dtype=torch.float64) + .5) / 2 - .5
            left = position.floor().long()
            alpha = position - left
            rows = torch.arange(2 * size, device=self.device)
            u = torch.zeros((2 * size, size), device=self.device, dtype=torch.float64)
            u[rows, left.clamp(0, size - 1)] = 1 - alpha
            u[rows, (left + 1).clamp(0, size - 1)] += alpha
            inverse = torch.linalg.solve(u.T @ u, u.T)
            self._interpolation[size] = (u, inverse)
        return self._interpolation[size]

    def plane_coefficients(self, plane):
        h, w = plane.shape
        if h % 8 or w % 8:
            raise ValueError('each color plane must be divisible by8')
        patches = (plane - 128).reshape(h // 8, 8, w // 8, 8).permute(0, 2, 1, 3)
        coef = self.dct @ patches @ self.dct.T
        return coef.permute(2, 3, 0, 1).reshape(64, -1)

    def restore_plane(self, coef, h, w):
        patches = coef.reshape(8, 8, h // 8, w // 8).permute(2, 3, 0, 1)
        restored = self.dct.T @ patches @ self.dct
        return restored.permute(0, 2, 1, 3).reshape(h, w) + 128

    def source_coefficients(self, color):
        _, h, w = color.shape
        if h % 16 or w % 16:
            raise ValueError('half-resolution color model requires dimensions divisible by16')
        chroma = color[1:].reshape(2, h // 2, 2, w // 2, 2).mean((2, 4)).round()
        return [self.plane_coefficients(color[0]), *[self.plane_coefficients(p) for p in chroma]]

    def target_coefficients(self, color):
        _, h, w = color.shape
        _, iy = self.interpolation(h // 2)
        _, ix = self.interpolation(w // 2)
        # Infer the latent chroma plane under this hypothesis by ordinary
        # least squares; no true subsampling label or intermediate image.
        latent = iy @ color[1:] @ ix.T
        return [self.plane_coefficients(color[0]), *[self.plane_coefficients(p) for p in latent]]

    def predict_multires(self, source, steps, h, w):
        planes = []
        for i in range(3):
            coef = (source[i] / steps[i, :, None]).round() * steps[i, :, None]
            ph, pw = (h, w) if i == 0 else (h // 2, w // 2)
            planes.append(self.restore_plane(coef, ph, pw).round().clamp(0, 255))
        uy, _ = self.interpolation(h // 2)
        ux, _ = self.interpolation(w // 2)
        color = torch.stack([planes[0], *[(uy @ plane @ ux.T).round() for plane in planes[1:]]])
        return self.to_rgb(color)

    @torch.no_grad()
    def fit(self, support_before, support_after):
        fitted, diagnostics = super().fit(support_before, support_after)
        before = self.image(support_before)
        h, w = before.shape[-2:]
        cut = (3 * (h // 8) // 4) * 8
        if cut % 16:
            raise ValueError('support training crop must contain complete chroma blocks')
        source = self.source_coefficients(self.to_color(before))
        colors = [self.to_color(self.image(v)) for v in support_after]
        targets = [self.target_coefficients(v) for v in colors]
        train_targets = [self.target_coefficients(v[:, :cut]) for v in colors]
        training, full = [], []
        for plane in range(3):
            x = source[plane]
            y = torch.stack([row[plane] for row in targets])
            templates = (x[None] / self.steps[:, None, None]).round() * self.steps[:, None, None]
            train_count = x.shape[-1] * cut // h
            train_y = torch.stack([row[plane] for row in train_targets])
            training.append(self.estimate_steps(templates[:, :, :train_count], train_y))
            full.append(self.estimate_steps(templates, y))
            del templates
        training, full = torch.stack(training, 1), torch.stack(full, 1)
        for i, target_value in enumerate(support_after):
            pred = self.predict_multires(source, training[i], h, w)
            target = self.image(target_value)
            error = float(((pred[:, cut:] - target[:, cut:]) * 255).square().mean())
            current_error = min(diagnostics[i]['block_quantization_held_mse_8bit'],
                                diagnostics[i]['generic_held_mse_8bit'])
            selected = error < current_error
            diagnostics[i].update(half_chroma_selected=selected, half_chroma_held_mse_8bit=error)
            if selected:
                fitted[i] = ('spatial', {'half_chroma_steps': full[i]})
                diagnostics[i].update(selected='spatial', block_quantization_selected=True,
                    response_class='half_resolution_color_block_quantization',
                    half_chroma_steps=full[i].cpu().tolist())
        return fitted, diagnostics

    def apply(self, before, operator):
        kind, value = operator
        if kind == 'spatial' and isinstance(value, dict) and 'half_chroma_steps' in value:
            source = self.source_coefficients(self.to_color(before))
            return self.predict_multires(source, value['half_chroma_steps'], *before.shape[-2:])
        return super().apply(before, operator)
