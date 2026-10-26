"""Use unsigned stable sorting for the exact dead-zone sufficient statistics."""
import numpy as np
import torch
from .chainprint_fused_offset import FusedOffsetLikelihood, FusedOffsetCandidate


def descending_absolute_order(delta):
    # For finite native float64, abs() has nonnegative IEEE754 bit patterns.
    # Their uint64 order equals numerical order. Complement reverses it;
    # stable sorting preserves every original tie, including signed zeros.
    if delta.dtype == np.dtype('float64') and delta.dtype.isnative and np.isfinite(delta).all():
        keys = np.bitwise_not(np.abs(delta).view(np.uint64))
        return np.argsort(keys, kind='stable')
    return np.argsort(-np.abs(delta), kind='stable')


class RadixOffsetLikelihood(FusedOffsetLikelihood):
    def fit_deadzone(self, before, after, basis, coefficients, mask):
        eligible = mask & (after > 0) & (after < 1)
        delta = torch.einsum('k,kchw->chw', coefficients, basis)[eligible].cpu().numpy()
        target = (after - before)[eligible].cpu().numpy()
        if not len(delta):
            return 1.0, 0.0
        order = descending_absolute_order(delta)
        x, y = delta[order], target[order]
        magnitude = np.abs(x)
        xy, xx = np.cumsum(x * y), np.cumsum(x * x)
        ends = np.flatnonzero(np.r_[magnitude[:-1] != magnitude[1:], True])
        ends = ends[xx[ends] > 1e-12]
        if not len(ends):
            return 1.0, 0.0
        explained = np.maximum(xy[ends], 0) ** 2 / xx[ends]
        best = ends[np.flatnonzero(explained == explained.max())[-1]]
        gain = max(0.0, float(xy[best] / xx[best]))
        threshold = float((magnitude[best] + magnitude[best + 1]) / 2) if best + 1 < len(x) else 0.0
        return gain, threshold


class RadixOffsetCandidate(FusedOffsetCandidate):
    def __init__(self, fallback, *, device='cuda:0'):
        super().__init__(fallback, device=device)
        self.replay = RadixOffsetLikelihood(device)
