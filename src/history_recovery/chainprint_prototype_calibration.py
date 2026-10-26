"""Candidate-only score centering without access to sibling query endpoints."""
import torch
from torch.nn import functional as F


def score_components(head, encoded, bank):
    candidates = torch.as_tensor(bank, dtype=torch.long, device=encoded['query_before'].device)
    qb = encoded['query_before']
    origin = head.project(qb)
    before = head.project(encoded['support_before'])
    after = head.project(encoded['support_after'])
    state = origin.expand(len(bank), -1, -1)
    for position in range(candidates.shape[1]):
        refs = candidates[:, position]
        state = head.step(state, before[refs], after[refs])
    prototypes = F.normalize((state-origin).flatten(1), dim=-1, eps=1e-6)
    count = encoded['query_after'].shape[0]
    pair = head.project(torch.cat((qb.expand(count, -1, -1), encoded['query_after']), 0))
    observed = F.normalize((pair[count:]-pair[:count]).flatten(1), dim=-1, eps=1e-6)
    raw = head.logit_scale * (observed @ prototypes.T)
    # The reference direction uses predicted candidates, never other observations.
    bias = head.logit_scale * (prototypes.mean(0) @ prototypes.T)
    return raw, bias


def calibrated_scores(head, encoded, bank, strength):
    raw, bias = score_components(head, encoded, bank)
    return raw - float(strength) * bias[None, :]
