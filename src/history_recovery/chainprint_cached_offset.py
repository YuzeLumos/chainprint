"""Exact within-fit reuse for the frozen support-offset response algorithm."""
import numpy as np
from .chainprint_quantization_offset import OffsetQuantizationLikelihood
from .chainprint_offset_candidate import OffsetQuantizationCandidate


class CachedOffsetQuantizationLikelihood(OffsetQuantizationLikelihood):
    """Caches live only inside one fit; query replay is unchanged.

    Image keys contain complete array bytes, not approximate hashes. Basis and
    spatial-solve entries retain their tensor arguments to prevent identity
    reuse. No tolerance-based equality, persistent support cache, or changed
    numerical solve is introduced.
    """

    def fit(self, support_before, support_after):
        if getattr(self, '_within_fit', None) is not None:
            raise RuntimeError('nested or concurrent fit on this instance is unsupported')
        self._within_fit = {'images': {}, 'bases': {}, 'spatial': {}}
        try:
            return super().fit(support_before, support_after)
        finally:
            self._within_fit = None

    def image(self, value):
        cache = getattr(self, '_within_fit', None)
        if cache is None:
            return super().image(value)
        array = np.asarray(value)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key not in cache['images']:
            cache['images'][key] = super().image(value)
        return cache['images'][key]

    def basis(self, value):
        cache = getattr(self, '_within_fit', None)
        if cache is None:
            return super().basis(value)
        key = id(value)
        if key not in cache['bases']:
            cache['bases'][key] = (value, super().basis(value))
        return cache['bases'][key][1]

    def spatial_fit(self, before, after, basis, mask):
        cache = getattr(self, '_within_fit', None)
        if cache is None:
            return super().spatial_fit(before, after, basis, mask)
        key = (id(before), id(after), id(basis), tuple(mask.shape), mask.cpu().numpy().tobytes())
        if key not in cache['spatial']:
            cache['spatial'][key] = (before, after, basis,
                super().spatial_fit(before, after, basis, mask))
        return cache['spatial'][key][3]


class CachedOffsetQuantizationCandidate(OffsetQuantizationCandidate):
    def __init__(self, fallback, *, device='cuda:0'):
        super().__init__(fallback, device=device)
        self.replay = CachedOffsetQuantizationLikelihood(device)
