"""Exact operation reuse within a single frozen query preparation."""
import numpy as np
import torch
from .chainprint_cached_offset import CachedOffsetQuantizationLikelihood, CachedOffsetQuantizationCandidate
from .chainprint_offset_candidate import PreparedOffsetEvidence
from .chainprint_block_candidate import _CachedFitReplay, _bank, _image_array


class PrefixOffsetLikelihood(CachedOffsetQuantizationLikelihood):
    def prepare(self, **kwargs):
        if getattr(self, '_prepared_operations', None) is not None:
            raise RuntimeError('nested or concurrent preparation on this instance is unsupported')
        self._prepared_operations = {}
        try:
            return super().prepare(**kwargs)
        finally:
            self._prepared_operations = None

    def apply(self, before, operator):
        cache = getattr(self, '_prepared_operations', None)
        if cache is None:
            return super().apply(before, operator)
        key = (id(before), id(operator))
        if key not in cache:
            # Retain both arguments so Python cannot reuse their identities.
            cache[key] = (before, operator, super().apply(before, operator))
        return cache[key][2]


class _CachedPrefixReplay(_CachedFitReplay, PrefixOffsetLikelihood):
    pass


class PreparedPrefixEvidence(PreparedOffsetEvidence):
    @torch.no_grad()
    def score(self, *, query_before, query_after, candidates):
        if query_before is None:
            raise ValueError('query-before required; endpoint-only use is unsupported')
        bank = _bank(candidates)
        qb, qa = _image_array(query_before), _image_array(query_after)
        if qa.ndim == 3:
            qa = qa[None]
        if (qb.shape != self._before.shape or qa.ndim != 4 or len(qa) == 0
                or qa.shape[1:] != qb.shape):
            raise ValueError('query and frozen support shapes disagree')
        if not self.uses_block_response(bank):
            return self._owner.fallback.score(query_before=qb, query_after=qa,
                support_before=self._before, support_after=self._after, candidates=bank)
        replay = _CachedPrefixReplay(self._owner.replay, self._fitted, self._diagnostics)
        states, _ = replay.prepare(query_before=qb, support_before=self._before,
            support_after=self._after, candidates=bank)
        return replay.compare_states(qa, states)


class PrefixOffsetCandidate(CachedOffsetQuantizationCandidate):
    def __init__(self, fallback, *, device='cuda:0'):
        super().__init__(fallback, device=device)
        self.replay = PrefixOffsetLikelihood(device)

    def prepare_support(self, **kwargs):
        evidence = super().prepare_support(**kwargs)
        return PreparedPrefixEvidence(self, evidence._before, evidence._after,
            evidence._fitted, evidence._diagnostics)
