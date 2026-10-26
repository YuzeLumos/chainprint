"""Profile output gains independently of a quantizer's input partitions.

Only support before/after pixels enter these statistics. The fixed integer step
grid and operation-wide threshold grid are inherited from the threshold model;
the extra nonnegative gain is shared across all blocks/content at a frequency.
"""
from __future__ import annotations

import time

import torch

from joint_block_cache import JointBlockCache
from threshold_block import THRESHOLD_BIASES, ThresholdReplay, _quantize

GAIN_FAMILIES = ("gain_full", "gain_full_offset")


def profile_nonnegative_gain(energy, cross):
    """Return argmin a>=0 and min(a*a*E-2*a*C); target square is omitted.

    ``energy`` and ``cross`` may broadcast. With zero template energy the
    prediction is identically zero: choose the nested value one, not an
    arbitrary zero gain. No epsilon or tunable shrinkage changes the objective.
    """
    if (not torch.isfinite(energy).all() or not torch.isfinite(cross).all()
            or (energy < 0).any()):
        raise ValueError("Invalid gain sufficient statistics")
    positive = energy > 0
    denominator = torch.where(positive, energy, torch.ones_like(energy))
    clipped_cross = cross.clamp_min(0)
    gains = torch.where(positive, clipped_cross / denominator,
                        torch.ones_like(cross))
    objective = torch.where(positive, -gains * clipped_cross,
                            torch.zeros_like(cross))
    if not torch.isfinite(gains).all() or not torch.isfinite(objective).all():
        raise ValueError("Nonfinite profiled gain or objective")
    return gains, objective


class GainReplay(ThresholdReplay):
    def gain_prediction(self, source, steps, bias, gains, h, w):
        bias = float(bias)
        if bias not in THRESHOLD_BIASES:
            raise ValueError("Threshold bias is outside the frozen finite grid")
        if (gains.shape != steps.shape or gains.shape != source.shape[:1]
                or not torch.isfinite(gains).all() or (gains < 0).any()):
            raise ValueError("Expected one finite nonnegative gain per frequency")
        # Exact identity nesting, including the original nearest-even branch
        # and all original restore/color/rounding behavior.
        if torch.equal(gains, torch.ones_like(gains)):
            return self.threshold_prediction(source, steps, bias, h, w)
        quantized = _quantize(source, steps[:, None], bias)
        quantized = gains[:, None] * quantized
        return self.to_rgb(self.restore_color(quantized, h, w))

    def apply(self, before, operator):
        kind, value = operator
        if (kind == "spatial" and isinstance(value, dict)
                and "gain_block_steps" in value):
            source = self.coefficients(self.to_color(before))
            prediction = self.gain_prediction(
                source, value["gain_block_steps"], value["gain_threshold_bias"],
                value["gain_frequency_scale"], *before.shape[-2:])
            if "block_rgb_offset_8bit" in value:
                prediction = self.offset_prediction(
                    prediction, value["block_rgb_offset_8bit"])
            return prediction
        return super().apply(before, operator)


class GainBlockCache:
    _validate_include = staticmethod(JointBlockCache._validate_include)
    _ordered_sum = staticmethod(JointBlockCache._ordered_sum)

    @torch.no_grad()
    def __init__(self, replay, before, after):
        if not isinstance(replay, GainReplay):
            raise TypeError("GainBlockCache requires GainReplay")
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        before_shape, after_shape = tuple(before.shape), tuple(after.shape)
        if len(before_shape) != 4 or before_shape[0] != 8 or before_shape[1] != 3:
            raise ValueError("Expected before[8,3,H,W]")
        if (len(after_shape) != 5 or after_shape[0] != 8
                or after_shape[2:] != before_shape[1:] or after_shape[1] < 1):
            raise ValueError("Expected aligned after[8,G,3,H,W]")
        self.height, self.width = before_shape[-2:]
        if self.height % 16 or self.width % 16:
            raise ValueError("Frozen block families require dimensions divisible by 16")
        self.replay, self.slot_count = replay, after_shape[1]
        self.steps = replay.steps
        expected = torch.arange(1, 256, device=self.steps.device,
                                dtype=self.steps.dtype)
        if self.steps.shape != (255,) or not torch.equal(self.steps, expected):
            raise ValueError("The frozen step grid must be integers 1..255")
        if self.steps.dtype != torch.float64:
            raise ValueError("Frozen block fitting uses float64")
        self.images = [replay.image(before[k]) for k in range(8)]
        self.targets = [[replay.image(after[k][g]) for g in range(self.slot_count)]
                        for k in range(8)]
        self.source_full, self.energy_statistics, self.cross_statistics = [], [], []
        for k, image in enumerate(self.images):
            source = replay.coefficients(replay.to_color(image))
            targets = [replay.coefficients(replay.to_color(target))
                       for target in self.targets[k]]
            if source.shape[0] != 192:
                raise ValueError("Expected 192 full-color block frequencies")
            energy, cross = self._statistics(source, targets)
            self.source_full.append(source)
            self.energy_statistics.append(energy)
            self.cross_statistics.append(cross)
        self._synchronize()
        self.report = dict(
            version="joint-threshold-gain-block-v1",
            support_count=8, operation_count=self.slot_count,
            support_shape=list(before_shape), families=list(GAIN_FAMILIES),
            energy_statistics_shape=[8, *self.energy_statistics[0].shape],
            cross_statistics_shape=[8, *self.cross_statistics[0].shape],
            statistic_dtype=str(self.energy_statistics[0].dtype),
            statistic_device=str(self.energy_statistics[0].device),
            step_grid="Integers 1..255", grid_chunk_size=32,
            threshold_biases=list(THRESHOLD_BIASES),
            threshold_bias_scope="One scalar shared by all 192 frequencies per operation",
            gain_scope="One nonnegative scalar per frequency, shared across contents and positions",
            gain_fit="a=max(C,0)/E if E>0; a=1 if E=0; no regularization or extra hyperparameter",
            objective="min over a>=0 of a*a*E-2*a*C; target-square term omitted",
            aggregation="Ordered sum over include, then divide objective by included block count",
            selection="Profile gain, first minimum q in ascending order per frequency, then first minimum b in frozen order",
            identity_nesting="All-one gain explicitly dispatches to original threshold_prediction",
            offset="Mean RGB residual per included content, equal-content mean, integer round",
            construction_wall_seconds=time.perf_counter() - wall_start,
            construction_cpu_process_seconds=time.process_time() - cpu_start,
            cached_statistic_bytes=sum(value.numel() * value.element_size()
                                       for value in self.energy_statistics + self.cross_statistics),
            candidate_calls=[],
            information_boundary="Supports only; no query, bank, semantic labels, or audit metadata",
        )

    def _synchronize(self):
        if self.steps.device.type == "cuda":
            torch.cuda.synchronize(self.steps.device)

    def _statistics(self, source, targets):
        if source.ndim != 2 or any(target.shape != source.shape for target in targets):
            raise ValueError("Source and target coefficient dimensions differ")
        shape = (len(THRESHOLD_BIASES), len(self.steps), source.shape[0])
        energy = torch.empty(shape, dtype=source.dtype, device=source.device)
        cross = torch.empty((shape[0], self.slot_count, *shape[1:]),
                            dtype=source.dtype, device=source.device)
        for b, bias in enumerate(THRESHOLD_BIASES):
            for start in range(0, len(self.steps), 32):
                stop = min(start + 32, len(self.steps))
                templates = _quantize(source[None], self.steps[start:stop, None, None], bias)
                energy[b, start:stop] = templates.square().sum(-1)
                for g, target in enumerate(targets):
                    cross[b, g, start:stop] = torch.einsum("qfn,fn->qf", templates, target)
                del templates
        if not torch.isfinite(energy).all() or not torch.isfinite(cross).all():
            raise ValueError("Nonfinite per-content gain statistics")
        return energy, cross

    @torch.no_grad()
    def candidates(self, include):
        indices = self._validate_include(include)
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        energy = self._ordered_sum(self.energy_statistics, indices)
        cross = self._ordered_sum(self.cross_statistics, indices)
        gain_grid, errors = profile_nonnegative_gain(energy[:, None], cross)
        block_count = sum(self.source_full[k].shape[-1] for k in indices)
        errors = errors / block_count
        minimum, step_index = errors.min(dim=2)
        by_bias_steps = self.steps[step_index]
        by_bias_gains = gain_grid.gather(2, step_index.unsqueeze(2)).squeeze(2)
        objectives = minimum.sum(dim=2).transpose(0, 1)
        bias_indices = objectives.argmin(dim=1)
        output, selected_biases, gain_summaries = [], [], []
        for g in range(self.slot_count):
            selected = int(bias_indices[g])
            bias = THRESHOLD_BIASES[selected]
            steps, gains = by_bias_steps[selected, g], by_bias_gains[selected, g]
            value = {"gain_block_steps": steps, "gain_threshold_bias": bias,
                     "gain_frequency_scale": gains}
            offsets = []
            for k in indices:
                prediction = self.replay.gain_prediction(
                    self.source_full[k], steps, bias, gains, self.height, self.width)
                offsets.append(((self.targets[k][g] - prediction) * 255).mean((1, 2)))
            offset = torch.stack(offsets).mean(0).round()
            if not torch.isfinite(offset).all():
                raise ValueError("Nonfinite gain RGB offset")
            output.append({
                "gain_full": ("spatial", value),
                "gain_full_offset": ("spatial", {**value, "block_rgb_offset_8bit": offset}),
            })
            selected_biases.append(bias)
            gain_summaries.append(dict(minimum=float(gains.min()), maximum=float(gains.max()),
                                       mean=float(gains.mean()), zero_count=int((gains == 0).sum()),
                                       exactly_one_count=int((gains == 1).sum())))
        self._synchronize()
        self.report["candidate_calls"].append(dict(
            include=indices, fit_count=len(indices),
            wall_seconds=time.perf_counter() - wall_start,
            cpu_process_seconds=time.process_time() - cpu_start,
            selected_threshold_bias_by_slot=selected_biases,
            selected_gain_summary_by_slot=gain_summaries,
            gain_objective_by_slot_and_bias=objectives.cpu().tolist(),
            objective_units="Summed per-frequency profiled coefficient objective; target-square term omitted",
        ))
        return output
