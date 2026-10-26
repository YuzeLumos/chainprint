"""General replay with the frozen r125 query-local spatial variance rule."""
import torch
from history_recovery.chainprint_general_replay import GeneralReplayLikelihood
from history_recovery.chainprint_query_variance import QueryVarianceLikelihood


class GeneralQueryVarianceLikelihood(GeneralReplayLikelihood):
    @torch.no_grad()
    def prepare(self, *, query_before, support_before, support_after, candidates):
        states, diagnostics = super().prepare(query_before=query_before, support_before=support_before,
                                              support_after=support_after, candidates=candidates)
        return [(*state, diagnostics[word[-1]]['selected'] == 'spatial')
                for state, word in zip(states, candidates)], diagnostics

    compare_states = QueryVarianceLikelihood.compare_states
