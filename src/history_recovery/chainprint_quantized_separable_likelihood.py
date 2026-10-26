"""Original-task inference with independently fitted staged integer supports."""
import torch
from .chainprint_resampling_geometry import ResamplingGeometryLikelihood
from .chainprint_quantized_separable import QuantizedSeparableResponse


class QuantizedSeparableLikelihood(ResamplingGeometryLikelihood):
    @torch.no_grad()
    def fit(self, support_before, support_after):
        fitted, diagnostics = super().fit(support_before, support_after)
        before = self.image(support_before); h, w = before.shape[-2:]
        mask = ((torch.arange(h, device=self.device)[:, None] + torch.arange(w, device=self.device)[None, :]) % 4 != 0).expand_as(before)
        for i, diag in enumerate(diagnostics):
            diag['quantized_separable_selected'] = False
            if fitted[i][0] != 'spatial' or diag['block_quantization_selected'] or diag['resampling_selected']:
                continue
            # These parameters belong only to this anonymous support episode.
            # No shared network, operation labels or query pixels enter fitting.
            model, loss = QuantizedSeparableResponse.fit_support(support_before, support_after[i], mask, self.device)
            target = self.image(support_after[i]); prediction = model.predict(before)
            error = float(((prediction[~mask] * 255).round() - (target[~mask] * 255).round()).square().mean())
            previous = (diag['color_validation_mse_8bit'] if diag['color_response_selected'] else
                        diag['deadzone_validation_mse_8bit'] if diag['deadzone_selected'] else diag['linear_validation_mse_8bit'])
            diag.update(quantized_separable_train_loss=loss, quantized_separable_held_mse_8bit=error,
                        quantized_separable_baseline_mse_8bit=previous)
            if error < previous:
                model.requires_grad_(False)
                fitted[i] = ('spatial', {'quantized_separable_model': model})
                diag.update(quantized_separable_selected=True, selected='spatial', response_class='quantized_separable_response',
                    quantized_separable_state={k:v.cpu().tolist() for k,v in model.state_dict().items()})
        return fitted, diagnostics

    def apply(self, before, operator):
        kind, value = operator
        if kind == 'spatial' and isinstance(value, dict) and 'quantized_separable_model' in value:
            return value['quantized_separable_model'].predict(before)
        return super().apply(before, operator)

    @staticmethod
    def route_from_support(diagnostics, candidates):
        active = {g for word in candidates for g in word}
        return any(diagnostics[g]['block_quantization_selected'] or diagnostics[g]['resampling_selected'] or
                   diagnostics[g].get('quantized_separable_selected', False) for g in active)
