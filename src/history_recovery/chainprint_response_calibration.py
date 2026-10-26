"""Support-only amplitude calibration of an existing spatial response.

One analytic gain, accepted using held support pixels. Kernel basis, ridge,
quantizer inference, uncertainty model and query scoring remain unchanged.
"""
import torch
from history_recovery.chainprint_quantized_likelihood import QuantizedReplayLikelihood


class CalibratedResponseLikelihood(QuantizedReplayLikelihood):
    def response_gain(self, before, after, basis, coefficients, mask):
        eligible = mask & (after > 0) & (after < 1)
        delta = torch.einsum('k,kchw->chw', coefficients, basis)[eligible]
        target = (after - before)[eligible]
        denominator = delta.square().sum()
        if float(denominator) <= 1e-12:
            return 1.0
        return max(0.0, float((delta * target).sum() / denominator))

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
            gain = self.response_gain(before, target, basis, train_coef, fitmask)
            original = self.spatial_predict(before, basis, train_coef)
            calibrated = self.spatial_predict(before, basis, train_coef * gain)
            original_error = float(((original[~fitmask] - target[~fitmask]) * 255).square().mean())
            calibrated_error = float(((calibrated[~fitmask] - target[~fitmask]) * 255).square().mean())
            selected = calibrated_error < original_error
            final_gain = self.response_gain(before, target, basis, op[1], allmask) if selected else 1.0
            fitted[i] = ('spatial', op[1] * final_gain)
            diagnostics[i].update(response_calibrated=selected, response_gain=final_gain,
                                  uncalibrated_validation_mse_8bit=original_error,
                                  calibrated_validation_mse_8bit=calibrated_error,
                                  coefficients=fitted[i][1].cpu().tolist())
        return fitted, diagnostics
