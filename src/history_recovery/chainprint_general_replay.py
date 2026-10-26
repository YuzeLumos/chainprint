"""General ordered replay with an explicit approximate residual-noise model."""
import math
import torch
from history_recovery.chainprint_color_response import ColorResponseLikelihood


class GeneralReplayLikelihood(ColorResponseLikelihood):
    @torch.no_grad()
    def prepare(self, *, query_before, support_before, support_after, candidates):
        fitted, diagnostics = self.fit(support_before, support_after)
        if (len(candidates) != 4 or any(len(word) not in (2, 3) or
                any(g not in range(4) for g in word) for word in candidates)):
            raise ValueError('Four anonymous length-two or length-three candidates required')
        # Preserve the complete already-validated r108 regime, including noise floors.
        if all(len(w) == 2 and {fitted[g][0] for g in w} == {'spatial', 'pointwise'} for w in candidates):
            return super().prepare(query_before=query_before, support_before=support_before,
                                   support_after=support_after, candidates=candidates)
        before, query = self.image(support_before), self.image(query_before)
        variances = []
        for op, value, diag in zip(fitted, support_after, diagnostics):
            variance = float(((self.apply(before, op) - self.image(value)) * 255).square().mean())
            variances.append(variance)
            diag['support_residual_variance_8bit'] = variance
        states = []
        for word in candidates:
            current = query
            variance = 0.0
            for g in word[:-1]:
                current = self.apply(current, fitted[g])
                variance += variances[g]
            last = word[-1]
            sigma = math.sqrt(max(variance + variances[last], 1 / 12))
            if fitted[last][0] == 'pointwise':
                lut = self.luts[fitted[last][1]].long()
                lower = torch.full((256,), float('nan'), device=self.device, dtype=query.dtype)
                upper = lower.clone()
                for value in lut.unique():
                    indices = torch.where(lut == value)[0]
                    left, right = int(indices[0]), int(indices[-1])
                    lower[value] = -float('inf') if left == 0 else left - .5
                    upper[value] = float('inf') if right == 255 else right + .5
                states.append((current * 255, sigma, lower, upper))
            else:
                centers = torch.arange(256, device=self.device, dtype=query.dtype)
                lower, upper = centers - .5, centers + .5
                lower[0], upper[-1] = -float('inf'), float('inf')
                states.append((self.apply(current, fitted[last]) * 255, sigma, lower, upper))
        return states, diagnostics
