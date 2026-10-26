"""Single-pass support fitting for the frozen r135 selection algorithm.

The full/half quantizer searches and support predictions are shared between
uncorrected and offset selection. Their numerical formulas, tie rules and
diagnostics are unchanged. Query scoring remains the r141 prefix API.
"""
import numpy as np
import torch
from .chainprint_color_response import ColorResponseLikelihood
from .chainprint_prefix_cache import PrefixOffsetLikelihood, PrefixOffsetCandidate


class FusedOffsetLikelihood(PrefixOffsetLikelihood):
    @torch.no_grad()
    def fit(self, support_before, support_after):
        if getattr(self, '_within_fit', None) is not None:
            raise RuntimeError('nested or concurrent fit on this instance is unsupported')
        self._within_fit = {'images': {}, 'bases': {}, 'spatial': {}}
        try:
            return self._fit_once(support_before, support_after)
        finally:
            self._within_fit = None

    def _fit_once(self, support_before, support_after):
        before = self.image(support_before)
        h, w = before.shape[-2:]
        cut = (3 * (h // 8) // 4) * 8
        if cut < 16 or cut >= h:
            raise ValueError('Insufficient support rows for block holdout')
        if cut % 16:
            raise ValueError('support training crop must contain complete chroma blocks')
        targets = [self.image(v) for v in support_after]
        color = self.to_color(before)
        colors = [self.to_color(v) for v in targets]
        source = self.coefficients(color)
        target_coef = torch.stack([self.coefficients(v) for v in colors])
        templates = (source[None] / self.steps[:, None, None]).round() * self.steps[:, None, None]
        count = (cut // 8) * (w // 8)
        full_train = self.estimate_steps(templates[:, :, :count], target_coef[:, :, :count])
        full_refit = self.estimate_steps(templates, target_coef)
        del templates

        cropped_before = np.asarray(support_before)[:, :cut, :]
        cropped_after = np.asarray(support_after)[:, :, :cut, :]
        train_base, _ = ColorResponseLikelihood.fit(self, cropped_before, cropped_after)
        full_predictions, full_errors, generic_errors = [], [], []
        for i, target in enumerate(targets):
            qpred = self.quantized_prediction(source, full_train[i], h, w)
            bpred = ColorResponseLikelihood.apply(self, before, train_base[i])
            full_predictions.append(qpred)
            full_errors.append(float(((qpred[:, cut:] - target[:, cut:]) * 255).square().mean()))
            generic_errors.append(float(((bpred[:, cut:] - target[:, cut:]) * 255).square().mean()))
        fitted, diagnostics = ColorResponseLikelihood.fit(self, support_before, support_after)
        for i, diag in enumerate(diagnostics):
            chosen = full_errors[i] < generic_errors[i]
            diag.update(block_quantization_selected=chosen,
                block_quantization_held_mse_8bit=full_errors[i], generic_held_mse_8bit=generic_errors[i])
            if chosen:
                fitted[i] = ('spatial', {'block_quantization_steps': full_refit[i]})
                diag.update(selected='spatial', response_class='color_block_quantization',
                            block_quantization_steps=full_refit[i].cpu().tolist())

        half_source = self.source_coefficients(color)
        half_targets = [self.target_coefficients(v) for v in colors]
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
        half_predictions, half_errors = [], []
        for i, target in enumerate(targets):
            prediction = self.predict_multires(half_source, half_train[i], h, w)
            error = float(((prediction[:, cut:] - target[:, cut:]) * 255).square().mean())
            half_predictions.append(prediction)
            half_errors.append(error)
            selected = error < min(full_errors[i], generic_errors[i])
            diagnostics[i].update(half_chroma_selected=selected, half_chroma_held_mse_8bit=error)
            if selected:
                fitted[i] = ('spatial', {'half_chroma_steps': half_refit[i]})
                diagnostics[i].update(selected='spatial', block_quantization_selected=True,
                    response_class='half_resolution_color_block_quantization',
                    half_chroma_steps=half_refit[i].cpu().tolist())

        for i, target in enumerate(targets):
            current_error = min(generic_errors[i], full_errors[i], half_errors[i])
            options = []
            for half, prediction in [(False, full_predictions[i]), (True, half_predictions[i])]:
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
                    half_chroma_selected=half,
                    response_class='offset_half_block_quantization' if half else 'offset_full_block_quantization',
                    block_rgb_offset_8bit=offset.cpu().tolist(),
                    block_rgb_offset_train_8bit=train_offset.cpu().tolist(),
                    **{key: steps.cpu().tolist()})
        return fitted, diagnostics


class FusedOffsetCandidate(PrefixOffsetCandidate):
    def __init__(self, fallback, *, device='cuda:0'):
        super().__init__(fallback, device=device)
        self.replay = FusedOffsetLikelihood(device)
