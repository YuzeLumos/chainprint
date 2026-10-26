"""Reusable support evidence for the frozen block-quantization candidate.

The supplied fallback should be the original accelerated r78 candidate. This
adapter is an optional accuracy candidate, not a replacement for that API.
"""
from copy import deepcopy
import numpy as np
import torch
from .chainprint_block_quantization import BlockQuantizationLikelihood
from .chainprint_frozen_candidate import FrozenChainPrintCandidate


def _image_array(value):
    array = np.asarray(value)
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError('finite RGB pixels in [0,1] required')
    return array


def _bank(candidates):
    if (len(candidates) != 4 or any(len(word) != 2 for word in candidates)
            or any(not isinstance(g, (int, np.integer)) or not 0 <= g < 4
                   for word in candidates for g in word)
            or len({tuple(word) for word in candidates}) != 4):
        raise ValueError('four distinct anonymous length-two histories required')
    return [list(word) for word in candidates]


class _CachedFitReplay(BlockQuantizationLikelihood):
    def __init__(self, original, fitted, diagnostics):
        # Read-only bases/LUTs are shared. Every score gets its own diagnostics;
        # GeneralReplay may annotate residual estimates during preparation.
        self.__dict__ = original.__dict__.copy()
        self._fitted = fitted
        self._diagnostics = diagnostics

    def fit(self, support_before, support_after):
        return self._fitted, deepcopy(self._diagnostics)


class PreparedBlockEvidence:
    def __init__(self, owner, before, after, fitted, diagnostics):
        self._owner = owner
        self._before, self._after = before, after
        self._fitted, self._diagnostics = fitted, deepcopy(diagnostics)

    def uses_block_response(self, candidates):
        return self._owner.replay.route_from_support(self._diagnostics, _bank(candidates))

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)

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
        replay = _CachedFitReplay(self._owner.replay, self._fitted, self._diagnostics)
        states, _ = replay.prepare(query_before=qb, support_before=self._before,
                                   support_after=self._after, candidates=bank)
        return replay.compare_states(qa, states)


class BlockQuantizationCandidate:
    """Exact r128 support routing with explicit reusable support preparation.

    `score` accepts the same pixel/anonymous-history interface as r78. Call
    `prepare_support` once when several independent queries share supports.
    No checkpoint, mixture weight, operator table or query labels are fitted.
    """
    def __init__(self, fallback, *, device='cuda:0'):
        if not callable(getattr(fallback, 'score', None)):
            raise TypeError('a frozen fallback candidate with score() is required')
        self.fallback = fallback
        self.replay = BlockQuantizationLikelihood(device)
        self.device = self.replay.device
        if hasattr(fallback, 'device') and torch.device(fallback.device) != self.device:
            raise ValueError('fallback and block response must use the same device')

    @torch.no_grad()
    def prepare_support(self, *, support_before, support_after):
        sb, sa = _image_array(support_before), _image_array(support_after)
        if sb.ndim != 3 or sb.shape[0] != 3 or sa.shape != (4, *sb.shape):
            raise ValueError('one RGB support origin and four aligned outputs required')
        # Own immutable copies so caller edits cannot invalidate the fitted state.
        sb, sa = np.array(sb, copy=True), np.array(sa, copy=True)
        fitted, diagnostics = self.replay.fit(sb, sa)
        sb.setflags(write=False)
        sa.setflags(write=False)
        return PreparedBlockEvidence(self, sb, sa, fitted, diagnostics)

    @torch.no_grad()
    def score(self, *, query_before, query_after, support_before, support_after, candidates):
        if query_before is None:
            raise ValueError('query-before required; endpoint-only use is unsupported')
        bank = _bank(candidates)
        evidence = self.prepare_support(support_before=support_before, support_after=support_after)
        return evidence.score(query_before=query_before, query_after=query_after, candidates=bank)

    complete_set_posterior = staticmethod(FrozenChainPrintCandidate.complete_set_posterior)
