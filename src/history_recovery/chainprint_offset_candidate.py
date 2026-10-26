"""Reusable support API preserving r135's quantizer offset operator dispatch."""
import numpy as np
import torch
from .chainprint_block_candidate import (
    BlockQuantizationCandidate, PreparedBlockEvidence, _CachedFitReplay, _bank, _image_array)
from .chainprint_quantization_offset import OffsetQuantizationLikelihood


class _CachedOffsetReplay(_CachedFitReplay, OffsetQuantizationLikelihood):
    """Reuse fitted parameters while retaining multiresolution apply()."""


class PreparedOffsetEvidence(PreparedBlockEvidence):
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
        replay = _CachedOffsetReplay(self._owner.replay, self._fitted, self._diagnostics)
        states, _ = replay.prepare(query_before=qb, support_before=self._before,
                                   support_after=self._after, candidates=bank)
        return replay.compare_states(qa, states)


class OffsetQuantizationCandidate(BlockQuantizationCandidate):
    """Optional r135 accuracy candidate with an unchanged accelerated fallback."""
    def __init__(self, fallback, *, device='cuda:0'):
        super().__init__(fallback, device=device)
        self.replay = OffsetQuantizationLikelihood(device)

    @torch.no_grad()
    def prepare_support(self, *, support_before, support_after):
        sb, sa = _image_array(support_before), _image_array(support_after)
        if sb.ndim != 3 or sb.shape[0] != 3 or sa.shape != (4, *sb.shape):
            raise ValueError('one RGB support origin and four aligned outputs required')
        sb, sa = np.array(sb, copy=True), np.array(sa, copy=True)
        fitted, diagnostics = self.replay.fit(sb, sa)
        sb.setflags(write=False)
        sa.setflags(write=False)
        return PreparedOffsetEvidence(self, sb, sa, fitted, diagnostics)
