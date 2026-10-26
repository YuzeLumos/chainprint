"""Support-selected integer RGB intercepts for block quantization responses."""
import torch
from .chainprint_multires_quantization import MultiResolutionQuantizationLikelihood


class OffsetQuantizationLikelihood(MultiResolutionQuantizationLikelihood):
    @staticmethod
    def offset_prediction(prediction, offset):
        return (prediction * 255 + offset[:, None, None]).round().clamp(0, 255) / 255

    @staticmethod
    def estimate_offset(prediction, target):
        # One empirical RGB intercept, in integer output byte units.
        return ((target - prediction) * 255).mean((1, 2)).round()

    @torch.no_grad()
    def fit(self, support_before, support_after):
        fitted, diagnostics = super().fit(support_before, support_after)
        before = self.image(support_before)
        h, w = before.shape[-2:]
        cut = (3 * (h // 8) // 4) * 8
        color = self.to_color(before)
        targets = [self.image(v) for v in support_after]
        colors = [self.to_color(v) for v in targets]

        source = self.coefficients(color)
        target_coef = torch.stack([self.coefficients(v) for v in colors])
        templates = (source[None] / self.steps[:, None, None]).round() * self.steps[:, None, None]
        count = source.shape[-1] * cut // h
        full_train = self.estimate_steps(templates[:, :, :count], target_coef[:, :, :count])
        full_refit = self.estimate_steps(templates, target_coef)
        del templates

        half_source = self.source_coefficients(color)
        half_targets = [self.target_coefficients(v) for v in colors]
        # Preserve r131's support holdout boundary: only training output rows
        # enter the latent chroma inverse used for training parameter fits.
        half_train_targets = [self.target_coefficients(v[:, :cut]) for v in colors]
        half_train, half_refit = [], []
        for plane in range(3):
            x = half_source[plane]
            templates = (x[None] / self.steps[:, None, None]).round() * self.steps[:, None, None]
            count = x.shape[-1] * cut // h
            half_train.append(self.estimate_steps(templates[:, :, :count],
                torch.stack([row[plane] for row in half_train_targets])))
            half_refit.append(self.estimate_steps(templates,
                torch.stack([row[plane] for row in half_targets])))
            del templates
        half_train, half_refit = torch.stack(half_train, 1), torch.stack(half_refit, 1)

        for i, target in enumerate(targets):
            current_error = min(diagnostics[i][key] for key in
                ('generic_held_mse_8bit', 'block_quantization_held_mse_8bit', 'half_chroma_held_mse_8bit'))
            options = []
            for half in (False, True):
                prediction = (self.predict_multires(half_source, half_train[i], h, w) if half
                    else self.quantized_prediction(source, full_train[i], h, w))
                offset = self.estimate_offset(prediction[:, :cut], target[:, :cut])
                corrected = self.offset_prediction(prediction, offset)
                error = float(((corrected[:, cut:] - target[:, cut:]) * 255).square().mean())
                options.append((error, half, offset))
            error, half, train_offset = min(options, key=lambda row: row[0])
            chosen = error < current_error
            diagnostics[i].update(block_rgb_offset_selected=chosen,
                uncorrected_best_held_mse_8bit=current_error,
                offset_full_held_mse_8bit=options[0][0], offset_half_held_mse_8bit=options[1][0])
            if chosen:
                steps = half_refit[i] if half else full_refit[i]
                prediction = (self.predict_multires(half_source, steps, h, w) if half
                    else self.quantized_prediction(source, steps, h, w))
                offset = self.estimate_offset(prediction, target)
                key = 'half_chroma_steps' if half else 'block_quantization_steps'
                fitted[i] = ('spatial', {key: steps, 'block_rgb_offset_8bit': offset})
                diagnostics[i].update(selected='spatial', block_quantization_selected=True,
                    half_chroma_selected=half, response_class='offset_half_block_quantization' if half else 'offset_full_block_quantization',
                    block_rgb_offset_8bit=offset.cpu().tolist(),
                    block_rgb_offset_train_8bit=train_offset.cpu().tolist(),
                    **{key: steps.cpu().tolist()})
        return fitted, diagnostics

    def apply(self, before, operator):
        prediction = super().apply(before, operator)
        kind, value = operator
        if kind == 'spatial' and isinstance(value, dict) and 'block_rgb_offset_8bit' in value:
            return self.offset_prediction(prediction, value['block_rgb_offset_8bit'])
        return prediction
