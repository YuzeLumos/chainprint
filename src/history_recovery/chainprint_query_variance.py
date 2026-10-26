"""Query-local nuisance variance after an inferred spatial last step.

Operator fits and quantizer-last likelihood remain frozen r108. Each query
estimates its own spatial-last residual variance; no sibling scores are used.
"""
import math
import numpy as np
import torch
from history_recovery.chainprint_color_response import ColorResponseLikelihood


class QueryVarianceLikelihood(ColorResponseLikelihood):
    @torch.no_grad()
    def prepare(self, *, query_before, support_before, support_after, candidates):
        states, diagnostics = super().prepare(query_before=query_before, support_before=support_before,
                                              support_after=support_after, candidates=candidates)
        return [(*state, diagnostics[word[-1]]['selected'] == 'spatial')
                for state, word in zip(states, candidates)], diagnostics

    @torch.no_grad()
    def compare_states(self, query_after, states):
        values = np.asarray(query_after)
        if values.ndim == 3:
            values = values[None]
        after = torch.stack([self.image(v) for v in values])
        indices = (after * 255).round().long()
        results = []
        for mean, sigma, lower, upper, spatial_last in states:
            if spatial_last:
                # Plug-in Gaussian variance from observed byte centers. This is
                # not an exact maximization of the later interval likelihood.
                variance = (indices.to(mean.dtype) - mean[None]).square().mean((-1, -2, -3))
                sigma = variance.clamp(min=1 / 12).sqrt()[:, None, None, None]
            lo, hi = lower[indices], upper[indices]
            valid = ~torch.isnan(lo)
            lp = self.interval_log_probability(
                torch.nan_to_num(lo, nan=0, neginf=-float('inf')),
                torch.nan_to_num(hi, nan=0, posinf=float('inf')), mean[None], sigma)
            lp = torch.where(valid, lp, math.log(torch.finfo(mean.dtype).tiny))
            results.append(lp.mean((-1, -2, -3)))
        return torch.stack(results, -1).cpu().numpy()
