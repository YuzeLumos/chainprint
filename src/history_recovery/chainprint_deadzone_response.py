"""Support-fitted dead-zone response, selected against r105 on held pixels."""
import numpy as np
import torch
from history_recovery.chainprint_response_calibration import CalibratedResponseLikelihood


class DeadzoneResponseLikelihood(CalibratedResponseLikelihood):
    def fit_deadzone(self, before, after, basis, coefficients, mask):
        eligible = mask & (after > 0) & (after < 1)
        delta = torch.einsum('k,kchw->chw', coefficients, basis)[eligible].cpu().numpy()
        target = (after - before)[eligible].cpu().numpy()
        if not len(delta):
            return 1.0, 0.0
        order = np.argsort(-np.abs(delta), kind='stable')
        x, y = delta[order], target[order]
        magnitude = np.abs(x)
        xy, xx = np.cumsum(x * y), np.cumsum(x * x)
        ends = np.flatnonzero(np.r_[magnitude[:-1] != magnitude[1:], True])
        ends = ends[xx[ends] > 1e-12]
        if not len(ends):
            return 1.0, 0.0
        # Exhaustive sufficient-statistic solution of the two fitted parameters.
        # This uses support observations, not a development hyperparameter grid.
        explained = np.maximum(xy[ends], 0) ** 2 / xx[ends]
        best = ends[np.flatnonzero(explained == explained.max())[-1]]
        gain = max(0.0, float(xy[best] / xx[best]))
        threshold = float((magnitude[best] + magnitude[best + 1]) / 2) if best + 1 < len(x) else 0.0
        return gain, threshold

    def deadzone_predict(self, before, basis, coefficients, gain, threshold):
        delta = torch.einsum('k,kchw->chw', coefficients, basis)
        result = before + gain * delta * (delta.abs() > threshold)
        return (result.clamp(0, 1) * 255).round() / 255

    @torch.no_grad()
    def fit(self, support_before, support_after):
        fitted, diagnostics = super().fit(support_before, support_after)
        before = self.image(support_before)
        basis = self.basis(before)
        h, w = before.shape[-2:]
        rows = torch.arange(h, device=self.device)[:, None]
        cols = torch.arange(w, device=self.device)[None, :]
        fitmask = ((rows + cols) % 4 != 0).expand_as(before)
        allmask = torch.ones_like(fitmask)
        for i, (op, target_value) in enumerate(zip(fitted, support_after)):
            if op[0] != 'spatial':
                continue
            target = self.image(target_value)
            train_coef = self.spatial_fit(before, target, basis, fitmask)
            original_gain = (self.response_gain(before, target, basis, train_coef, fitmask)
                             if diagnostics[i]['response_calibrated'] else 1.0)
            original = self.spatial_predict(before, basis, train_coef * original_gain)
            gain, threshold = self.fit_deadzone(before, target, basis, train_coef, fitmask)
            nonlinear = self.deadzone_predict(before, basis, train_coef, gain, threshold)
            oe = float(((original[~fitmask] - target[~fitmask]) * 255).square().mean())
            ne = float(((nonlinear[~fitmask] - target[~fitmask]) * 255).square().mean())
            selected = ne < oe
            diagnostics[i].update(deadzone_selected=selected, linear_validation_mse_8bit=oe,
                                  deadzone_validation_mse_8bit=ne)
            if selected:
                full_coef = self.spatial_fit(before, target, basis, allmask)
                gain, threshold = self.fit_deadzone(before, target, basis, full_coef, allmask)
                fitted[i] = ('spatial', {'coef': full_coef, 'gain': gain, 'threshold': threshold})
                diagnostics[i].update(deadzone_gain=gain, deadzone_threshold_8bit=threshold * 255,
                                      coefficients=full_coef.cpu().tolist())
        return fitted, diagnostics

    def apply(self, before, operator):
        if operator[0] == 'spatial' and isinstance(operator[1], dict):
            value = operator[1]
            return self.deadzone_predict(before, self.basis(before), value['coef'],
                                         value['gain'], value['threshold'])
        return super().apply(before, operator)
