"""Shared support-conditioned token transitions for the balanced B diagnostic.

No class IDs, endpoint-conditioned transition, or candidate-position parameters.
The mean-broadcast control has exactly the same shapes and trainable parameters.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def retain_or_pool(tokens: torch.Tensor, pooled: torch.Tensor, *, spatial: bool) -> torch.Tensor:
    if tokens.ndim != 3 or pooled.shape != (tokens.shape[0], tokens.shape[2]):
        raise ValueError("frontend evidence axes disagree")
    # Carry amplitude statistics to both arms; only centered spatial deviations
    # differ. The mean-broadcast arm does not lose an extra statistics branch.
    if not spatial:
        return pooled[:, None].expand_as(tokens)
    return tokens - tokens.mean(1, keepdim=True) + pooled[:, None]


class SpatialStateTransition(nn.Module):
    def __init__(self, *, token_dim: int, state_dim: int = 48,
                 attention_heads: int = 4, hidden_dim: int = 128,
                 logit_scale: float = 5.0) -> None:
        super().__init__()
        if min(token_dim, state_dim, attention_heads, hidden_dim) < 1 or state_dim % attention_heads:
            raise ValueError("positive dimensions and divisible attention heads required")
        if not 0 < logit_scale <= 100:
            raise ValueError("invalid fixed logit scale")
        self.token_dim = token_dim
        self.state_dim = state_dim
        self.logit_scale = float(logit_scale)
        self.project = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, state_dim), nn.GELU())
        self.attention_norm = nn.LayerNorm(state_dim)
        self.transfer = nn.MultiheadAttention(state_dim, attention_heads, dropout=0., bias=False, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(3 * state_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, state_dim), nn.Sigmoid())

    def step(self, current: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if current.shape != before.shape or before.shape != after.shape or current.ndim != 3:
            raise ValueError("state and support token axes must align")
        difference = after - before
        transferred, _ = self.transfer(self.attention_norm(current), self.attention_norm(before),
                                        difference, need_weights=False)
        summary = difference.mean(1, keepdim=True).expand_as(current)
        gate = self.gate(torch.cat((current, transferred, summary), -1))
        # No normalization/bias is applied after this residual. Identical
        # before/after support must define an exact identity transition.
        return current + gate * transferred

    def compose(self, query_before: torch.Tensor, support_before: torch.Tensor,
                support_after: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (query_before.ndim != 3 or query_before.shape[0] != 1 or
            support_before.shape != support_after.shape or support_before.shape[0] != 4 or
            support_before.shape[1:] != query_before.shape[1:] or query_before.shape[-1] != self.token_dim or
            candidates.ndim != 2 or candidates.shape[1] != 2 or candidates.dtype != torch.long or
            candidates.numel() == 0 or torch.any(candidates < 0) or torch.any(candidates >= 4)):
            raise ValueError("invalid anonymous support/candidate boundary")
        origin = self.project(query_before)
        before, after = self.project(support_before), self.project(support_after)
        state = origin.expand(candidates.shape[0], -1, -1)
        for position in range(2):
            refs = candidates[:, position]
            state = self.step(state, before[refs], after[refs])
        return origin, state

    def forward(self, query_before: torch.Tensor, query_after: torch.Tensor,
                support_before: torch.Tensor, support_after: torch.Tensor,
                candidates: torch.Tensor) -> torch.Tensor:
        if query_after.ndim != 3 or query_after.shape[1:] != query_before.shape[1:]:
            raise ValueError("query endpoints must share the source token layout")
        origin, predicted = self.compose(query_before, support_before, support_after, candidates)
        count = query_after.shape[0]
        # Project identical-sized before/after rows in one call. Separate
        # single-row vs multi-row GEMMs can introduce roundoff that cosine
        # normalization would amplify into a false unchanged-image signal.
        observed_pair = self.project(torch.cat((query_before.expand(count, -1, -1), query_after), 0))
        observed_change = observed_pair[count:] - observed_pair[:count]
        predicted_delta = F.normalize((predicted - origin).flatten(1), dim=-1, eps=1e-6)
        observed_delta = F.normalize(observed_change.flatten(1), dim=-1, eps=1e-6)
        return self.logit_scale * (observed_delta @ predicted_delta.T)


class AffineSpatialStateTransition(SpatialStateTransition):
    """Add support-conditioned multiplicative spatial gain to the residual step."""

    def __init__(self, *, token_dim: int, state_dim: int = 48,
                 attention_heads: int = 4, hidden_dim: int = 128,
                 logit_scale: float = 5.0, maximum_gain: float = 0.5) -> None:
        super().__init__(token_dim=token_dim, state_dim=state_dim,
                         attention_heads=attention_heads, hidden_dim=hidden_dim,
                         logit_scale=logit_scale)
        if not 0 < maximum_gain <= 1:
            raise ValueError("maximum_gain must be in (0, 1]")
        self.maximum_gain = float(maximum_gain)
        self.spatial_gain = nn.Sequential(
            nn.Linear(3 * state_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, state_dim), nn.Tanh())
        nn.init.zeros_(self.spatial_gain[-2].weight)
        nn.init.zeros_(self.spatial_gain[-2].bias)

    def step(self, current: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if current.shape != before.shape or before.shape != after.shape or current.ndim != 3:
            raise ValueError("state and support token axes must align")
        difference = after - before
        transferred, _ = self.transfer(self.attention_norm(current), self.attention_norm(before),
                                        difference, need_weights=False)
        summary = difference.mean(1, keepdim=True).expand_as(current)
        context = torch.cat((current, transferred, summary), -1)
        gate = self.gate(context)
        active = (difference.detach().abs().amax(dim=(1, 2), keepdim=True) > 0).to(current.dtype)
        gain = self.maximum_gain * self.spatial_gain(context) * active
        spatial_residual = current - current.mean(1, keepdim=True)
        return current + gate * transferred + gain * spatial_residual


class LowRankAffineSpatialStateTransition(AffineSpatialStateTransition):
    """Add a support-conditioned low-rank channel operator to the affine step."""

    def __init__(self, *, token_dim: int, state_dim: int = 48,
                 attention_heads: int = 4, hidden_dim: int = 128,
                 logit_scale: float = 5.0, maximum_gain: float = 0.5,
                 operator_rank: int = 8, maximum_operator_gain: float = 0.25) -> None:
        super().__init__(token_dim=token_dim, state_dim=state_dim,
                         attention_heads=attention_heads, hidden_dim=hidden_dim,
                         logit_scale=logit_scale, maximum_gain=maximum_gain)
        if not 0 < operator_rank <= state_dim:
            raise ValueError("operator_rank must be in [1, state_dim]")
        if not 0 < maximum_operator_gain <= 1:
            raise ValueError("maximum_operator_gain must be in (0, 1]")
        self.operator_rank = int(operator_rank)
        self.maximum_operator_gain = float(maximum_operator_gain)
        self.operator_down = nn.Linear(state_dim, operator_rank, bias=False)
        self.operator_coeff = nn.Sequential(
            nn.Linear(3 * state_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, operator_rank), nn.Tanh())
        self.operator_up = nn.Linear(operator_rank, state_dim, bias=False)
        # Preserve the complete affine checkpoint exactly at warm start.
        nn.init.zeros_(self.operator_up.weight)

    def step(self, current: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if current.shape != before.shape or before.shape != after.shape or current.ndim != 3:
            raise ValueError("state and support token axes must align")
        difference = after - before
        transferred, _ = self.transfer(self.attention_norm(current), self.attention_norm(before),
                                        difference, need_weights=False)
        summary = difference.mean(1, keepdim=True).expand_as(current)
        context = torch.cat((current, transferred, summary), -1)
        gate = self.gate(context)
        active = (difference.detach().abs().amax(dim=(1, 2), keepdim=True) > 0).to(current.dtype)
        spatial_residual = current - current.mean(1, keepdim=True)
        gain = self.maximum_gain * self.spatial_gain(context) * active
        coefficients = self.operator_coeff(context)
        mixed = self.operator_up(self.operator_down(spatial_residual) * coefficients)
        return (current + gate * transferred + gain * spatial_residual +
                self.maximum_operator_gain * active * mixed)


class MomentAffineSpatialStateTransition(AffineSpatialStateTransition):
    """Condition an affine state update on non-cancelling support-effect moments."""

    def __init__(self, *, token_dim: int, state_dim: int = 48,
                 attention_heads: int = 4, hidden_dim: int = 128,
                 logit_scale: float = 5.0, maximum_gain: float = 0.5,
                 maximum_moment_gain: float = 0.25) -> None:
        super().__init__(token_dim=token_dim, state_dim=state_dim,
                         attention_heads=attention_heads, hidden_dim=hidden_dim,
                         logit_scale=logit_scale, maximum_gain=maximum_gain)
        if not 0 < maximum_moment_gain <= 1:
            raise ValueError("maximum_moment_gain must be in (0, 1]")
        self.maximum_moment_gain = float(maximum_moment_gain)
        self.moment_affine = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2 * state_dim), nn.Tanh())
        nn.init.zeros_(self.moment_affine[-2].weight)
        nn.init.zeros_(self.moment_affine[-2].bias)

    def step(self, current: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if current.shape != before.shape or before.shape != after.shape or current.ndim != 3:
            raise ValueError("state and support token axes must align")
        difference = after - before
        transferred, _ = self.transfer(self.attention_norm(current), self.attention_norm(before),
                                        difference, need_weights=False)
        summary = difference.mean(1, keepdim=True).expand_as(current)
        context = torch.cat((current, transferred, summary), -1)
        gate = self.gate(context)
        active = (difference.detach().abs().amax(dim=(1, 2), keepdim=True) > 0).to(current.dtype)
        spatial_residual = current - current.mean(1, keepdim=True)
        gain = self.maximum_gain * self.spatial_gain(context) * active
        absolute_mean = difference.abs().mean(1)
        rms = difference.square().mean(1).clamp_min(1e-12).sqrt()
        moment_gain, moment_shift = self.moment_affine(
            torch.cat((absolute_mean, rms), -1)).chunk(2, -1)
        moment_gain = moment_gain[:, None] * active
        moment_shift = moment_shift[:, None] * active
        return (current + gate * transferred + gain * spatial_residual +
                self.maximum_moment_gain *
                (moment_gain * spatial_residual + moment_shift))


class DecontentAffineSpatialStateTransition(AffineSpatialStateTransition):
    """Add a relation-keyed attention path that does not key on support content."""

    def __init__(self, *, token_dim: int, state_dim: int = 48,
                 attention_heads: int = 4, hidden_dim: int = 128,
                 logit_scale: float = 5.0, maximum_gain: float = 0.5,
                 maximum_relation_gain: float = 0.5) -> None:
        super().__init__(token_dim=token_dim, state_dim=state_dim,
                         attention_heads=attention_heads, hidden_dim=hidden_dim,
                         logit_scale=logit_scale, maximum_gain=maximum_gain)
        if not 0 < maximum_relation_gain <= 1:
            raise ValueError("maximum_relation_gain must be in (0, 1]")
        self.maximum_relation_gain = float(maximum_relation_gain)
        self.relation_gate = nn.Sequential(
            nn.Linear(3 * state_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, state_dim), nn.Tanh())
        nn.init.zeros_(self.relation_gate[-2].weight)
        nn.init.zeros_(self.relation_gate[-2].bias)

    def step(self, current: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if current.shape != before.shape or before.shape != after.shape or current.ndim != 3:
            raise ValueError("state and support token axes must align")
        difference = after - before
        transferred, _ = self.transfer(self.attention_norm(current), self.attention_norm(before),
                                        difference, need_weights=False)
        relation_transferred, _ = self.transfer(
            self.attention_norm(current), self.attention_norm(difference),
            difference, need_weights=False)
        summary = difference.mean(1, keepdim=True).expand_as(current)
        context = torch.cat((current, transferred, summary), -1)
        gate = self.gate(context)
        active = (difference.detach().abs().amax(dim=(1, 2), keepdim=True) > 0).to(current.dtype)
        spatial_residual = current - current.mean(1, keepdim=True)
        gain = self.maximum_gain * self.spatial_gain(context) * active
        magnitude = difference.abs().mean(1, keepdim=True).expand_as(current)
        relation_context = torch.cat((current, relation_transferred, magnitude), -1)
        relation_gain = self.maximum_relation_gain * self.relation_gate(relation_context) * active
        return (current + gate * transferred + gain * spatial_residual +
                relation_gain * relation_transferred)
