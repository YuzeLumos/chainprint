"""Fit content-shared linear residual corrections from anonymous support pixels."""
import numpy as np
import torch
from torch.nn import functional as F


def features(before, prediction):
    """31 channels: intercept, predicted RGB, and 3x3 RGB input-minus-prediction."""
    difference = (before - prediction).to(torch.float64)
    neighborhood = F.unfold(F.pad(difference[None], (1, 1, 1, 1), mode='replicate'), kernel_size=3)[0]
    h, w = before.shape[-2:]
    return torch.cat((torch.ones((1, h*w), device=before.device, dtype=torch.float64),
                      prediction.to(torch.float64).reshape(3, -1), neighborhood), dim=0).T


def fit_coefficients(xs, ys, include):
    # Aggregate sufficient statistics instead of repeatedly copying image patches.
    gram = sum((xs[k].T @ xs[k] for k in include), torch.zeros((31, 31), dtype=torch.float64, device=xs[0].device))
    rhs = sum((xs[k].T @ ys[k] for k in include), torch.zeros((31, 3), dtype=torch.float64, device=xs[0].device))
    # Fixed numerical pseudoinverse tolerance, not a task-score-selected penalty.
    coefficients = torch.linalg.pinv(gram, rtol=1e-10, hermitian=True) @ rhs
    if not torch.isfinite(coefficients).all():
        raise ValueError('Nonfinite residual coefficients')
    return coefficients


def corrected(prediction, design, coefficients):
    h, w = prediction.shape[-2:]
    residual = (design @ coefficients).T.reshape(3, h, w)
    return ((prediction.to(torch.float64) * 255 + residual).clamp(0, 255).round() / 255).to(prediction.dtype)


class ResidualReplay:
    def __init__(self, base):
        self.base, self.device = base, base.device

    def image(self, value):
        return self.base.image(value)

    def apply(self, before, operator):
        if operator[0] != 'shared_residual':
            return self.base.apply(before, operator)
        underlying, coefficients = operator[1]
        prediction = self.base.apply(before, underlying)
        return corrected(prediction, features(before, prediction), coefficients)


@torch.no_grad()
def calibrate_operators(replay, before, after, strict_fits, diagnostics, pooled, selections):
    """Use nested content exclusion for adoption, then refit the accepted correction."""
    count = len(before)
    if count != 8:
        raise ValueError('The fixed development design requires eight supports')
    images = [replay.image(x) for x in before]
    targets = [[replay.image(y) for y in group] for group in after]
    operators, records = list(pooled), []
    for g, selection in enumerate(selections):
        candidates = [group[g] for group in strict_fits] + [
            ('spatial', {'codec_quality': diagnostics[k][g]['codec_support_fit']['quality'],
                         'codec_subsampling': diagnostics[k][g]['codec_support_fit']['subsampling']}) for k in range(count)]
        designs = {}

        def prepare(index):
            if index not in designs:
                predictions = [replay.apply(x, candidates[index]) for x in images]
                xs = [features(x, y) for x, y in zip(images, predictions)]
                ys = [((targets[k][g].to(torch.float64) - predictions[k]) * 255).reshape(3, -1).T for k in range(count)]
                designs[index] = predictions, xs, ys
            return designs[index]

        baseline, residual = [], []
        for held, index in enumerate(selection['leave_one_content_out_selected']):
            if selection['candidates'][index]['origin'] == held:
                raise ValueError('Held content leaked into base-operator fitting')
            predictions, xs, ys = prepare(index)
            coefficients = fit_coefficients(xs, ys, [j for j in range(count) if j != held])
            target = targets[held][g].to(torch.float64)
            calibrated = corrected(predictions[held], xs[held], coefficients)
            baseline.append(float(((predictions[held].to(torch.float64) - target) * 255).square().mean()))
            residual.append(float(((calibrated.to(torch.float64) - target) * 255).square().mean()))
        base, new = np.asarray(baseline), np.asarray(residual)
        if not np.allclose(base, selection['leave_one_content_out_mse_8bit'], rtol=0, atol=1e-10):
            raise ValueError('Content-excluded baseline changed')
        adopt = bool(np.all(new <= base) and new.mean() < base.mean())
        coefficients_record = None
        if adopt:
            _, xs, ys = prepare(selection['selected'])
            coefficients = fit_coefficients(xs, ys, list(range(count)))
            operators[g] = ('shared_residual', (pooled[g], coefficients))
            coefficients_record = coefficients.cpu().tolist()
        records.append({'adopted': adopt, 'held_baseline_mse_8bit': baseline, 'held_corrected_mse_8bit': residual,
                        'coefficients': coefficients_record, 'features': 31,
                        'validation_scope': 'Base candidate and residual regression both exclude held content; these eight folds select adoption and are not an independent test.'})
    return operators, records
