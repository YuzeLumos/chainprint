"""Support-fitted quantization grids in a fixed block/color transform.

This adds an explicit structural prior, not a JPEG decoder or a quality table.
Only support pixels identify the 192 quantization steps and model selection.
"""
import math
import numpy as np
from PIL import Image
import torch
from history_recovery.chainprint_general_query_variance import GeneralQueryVarianceLikelihood
from history_recovery.chainprint_color_response import ColorResponseLikelihood


class BlockQuantizationLikelihood(GeneralQueryVarianceLikelihood):
    def __init__(self, device='cuda:0'):
        super().__init__(device)
        k = torch.arange(8, device=self.device, dtype=torch.float64)[:, None]
        n = torch.arange(8, device=self.device, dtype=torch.float64)[None]
        self.dct = .5 * torch.cos(math.pi * (n + .5) * k / 8)
        self.dct[0] *= 2 ** -.5
        self.steps = torch.arange(1, 256, device=self.device, dtype=torch.float64)

    def to_color(self, image):
        rgb = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
        ycc = np.asarray(Image.fromarray(rgb, mode='RGB').convert('YCbCr')).copy()
        return torch.tensor(ycc.transpose(2, 0, 1), device=self.device, dtype=torch.float64)

    def coefficients(self, color):
        c, h, w = color.shape
        if c != 3 or h % 8 or w % 8:
            raise ValueError('Block model needs RGB images divisible by8')
        patches = (color - 128).reshape(3, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)
        value = self.dct @ patches @ self.dct.T
        return value.permute(0, 3, 4, 1, 2).reshape(192, -1)

    def restore_color(self, coef, h, w):
        patches = coef.reshape(3, 8, 8, h // 8, w // 8).permute(0, 3, 4, 1, 2)
        value = self.dct.T @ patches @ self.dct
        return value.permute(0, 1, 3, 2, 4).reshape(3, h, w) + 128

    def to_rgb(self, color):
        array = color.round().clamp(0, 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        rgb = np.asarray(Image.fromarray(array, mode='YCbCr').convert('RGB')).copy()
        return torch.tensor(rgb.transpose(2, 0, 1), device=self.device, dtype=torch.float64) / 255

    def quantized_prediction(self, source, steps, h, w):
        quantized = (source / steps[:, None]).round() * steps[:, None]
        return self.to_rgb(self.restore_color(quantized, h, w))

    def estimate_steps(self, templates, targets):
        # Constants in target-square cancel across spacing hypotheses.
        errors = templates.square().mean(-1)[None] - 2 * torch.einsum('qfn,ofn->oqf', templates, targets) / targets.shape[-1]
        return self.steps[errors.argmin(1)]

    @torch.no_grad()
    def fit(self, support_before, support_after):
        before = self.image(support_before)
        h, w = before.shape[-2:]
        cut = (3 * (h // 8) // 4) * 8
        if cut < 16 or cut >= h:
            raise ValueError('Insufficient support rows for block holdout')
        source = self.coefficients(self.to_color(before))
        targets = torch.stack([self.coefficients(self.to_color(self.image(v))) for v in support_after])
        templates = (source[None] / self.steps[:, None, None]).round() * self.steps[:, None, None]
        train_count = (cut // 8) * (w // 8)
        train_steps = self.estimate_steps(templates[:, :, :train_count], targets[:, :, :train_count])
        full_steps = self.estimate_steps(templates, targets)
        del templates
        # Both models are selected on held bottom rows. The generic baseline
        # is fitted on the top crop, so held target pixels never fit its model.
        cropped_before = np.asarray(support_before)[:, :cut, :]
        cropped_after = np.asarray(support_after)[:, :, :cut, :]
        train_base, _ = ColorResponseLikelihood.fit(self, cropped_before, cropped_after)
        selected, validation = [], []
        for i, target_value in enumerate(support_after):
            target = self.image(target_value)
            qpred = self.quantized_prediction(source, train_steps[i], h, w)
            bpred = ColorResponseLikelihood.apply(self, before, train_base[i])
            qe = float(((qpred[:, cut:] - target[:, cut:]) * 255).square().mean())
            be = float(((bpred[:, cut:] - target[:, cut:]) * 255).square().mean())
            selected.append(qe < be)
            validation.append((qe, be))
        fitted, diagnostics = ColorResponseLikelihood.fit(self, support_before, support_after)
        for i, chosen in enumerate(selected):
            diagnostics[i].update(block_quantization_selected=chosen,
                block_quantization_held_mse_8bit=validation[i][0],
                generic_held_mse_8bit=validation[i][1])
            if chosen:
                fitted[i] = ('spatial', {'block_quantization_steps': full_steps[i]})
                diagnostics[i].update(selected='spatial', response_class='color_block_quantization',
                    block_quantization_steps=full_steps[i].cpu().tolist())
        return fitted, diagnostics

    def apply(self, before, operator):
        kind, value = operator
        if kind == 'spatial' and isinstance(value, dict) and 'block_quantization_steps' in value:
            source = self.coefficients(self.to_color(before))
            return self.quantized_prediction(source, value['block_quantization_steps'], *before.shape[-2:])
        return super().apply(before, operator)

    @staticmethod
    def route_from_support(diagnostics, candidates):
        active = {g for word in candidates for g in word}
        return any(diagnostics[g]['block_quantization_selected'] for g in active)
