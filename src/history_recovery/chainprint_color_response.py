"""Generic cross-channel affine spatial response inferred from support pixels."""
import torch
from history_recovery.chainprint_deadzone_response import DeadzoneResponseLikelihood


class ColorResponseLikelihood(DeadzoneResponseLikelihood):
    def color_features(self, before, basis=None):
        basis = self.basis(before) if basis is None else basis
        # 14 spatial orbits x 3 source channels, plus the 3 center values.
        return torch.cat((basis.flatten(0, 1), before), 0)

    def color_fit(self, before, after, features, mask):
        eligible = mask & ((after > 0) & (after < 1)).all(0)
        count = int(eligible.sum())
        dim = features.shape[0]
        if count == 0:
            return {'color_coef': torch.zeros((dim, 3), device=self.device, dtype=before.dtype),
                    'feature_mean': torch.zeros(dim, device=self.device, dtype=before.dtype),
                    'delta_mean': torch.zeros(3, device=self.device, dtype=before.dtype)}
        x = features[:, eligible]
        y = (after - before)[:, eligible]
        xm, ym = x.mean(-1), y.mean(-1)
        xc, yc = x - xm[:, None], y - ym[:, None]
        gram = xc @ xc.T / count
        rhs = xc @ yc.T / count
        coef = torch.linalg.solve(gram + 1e-3 * torch.eye(dim, device=self.device, dtype=before.dtype), rhs)
        return {'color_coef': coef, 'feature_mean': xm, 'delta_mean': ym}

    def color_predict(self, before, features, fitted):
        delta = torch.einsum('kc,khw->chw', fitted['color_coef'],
                            features - fitted['feature_mean'][:, None, None])
        delta += fitted['delta_mean'][:, None, None]
        return ((before + delta).clamp(0, 1) * 255).round() / 255

    @torch.no_grad()
    def fit(self, support_before, support_after):
        fitted, diagnostics = super().fit(support_before, support_after)
        before = self.image(support_before)
        features = self.color_features(before)
        h, w = before.shape[-2:]
        rows = torch.arange(h, device=self.device)[:, None]
        cols = torch.arange(w, device=self.device)[None, :]
        fitmask = (rows + cols) % 4 != 0
        allmask = torch.ones_like(fitmask)
        for i, target_value in enumerate(support_after):
            target = self.image(target_value)
            train_fit = self.color_fit(before, target, features, fitmask)
            predicted = self.color_predict(before, features, train_fit)
            error = float(((predicted[:, ~fitmask] - target[:, ~fitmask]) * 255).square().mean())
            diag = diagnostics[i]
            if fitted[i][0] == 'pointwise':
                baseline_error = diag['pointwise_validation_mse_8bit']
            elif diag['deadzone_selected']:
                baseline_error = diag['deadzone_validation_mse_8bit']
            else:
                baseline_error = diag['linear_validation_mse_8bit']
            selected = error < baseline_error
            diag.update(color_response_selected=selected, color_validation_mse_8bit=error,
                        previous_validation_mse_8bit=baseline_error)
            if selected:
                full = self.color_fit(before, target, features, allmask)
                fitted[i] = ('spatial', full)
                diag.update(selected='spatial', response_class='cross_channel_affine',
                            color_coefficients=full['color_coef'].cpu().tolist(),
                            feature_mean=full['feature_mean'].cpu().tolist(),
                            delta_mean=full['delta_mean'].cpu().tolist())
        return fitted, diagnostics

    def apply(self, before, operator):
        if operator[0] == 'spatial' and isinstance(operator[1], dict) and 'color_coef' in operator[1]:
            return self.color_predict(before, self.color_features(before), operator[1])
        return super().apply(before, operator)
