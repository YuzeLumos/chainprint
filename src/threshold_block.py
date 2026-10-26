"""One shared threshold bias for a support-fitted full-color block quantizer.

The nearest-even branch is exactly the existing block implementation. Other
biases shift every frequency's threshold by the SAME operation-level scalar.
"""
from __future__ import annotations

import time

import torch

from history_recovery.chainprint_codec_replay import SupportCodecReplay
from joint_block_cache import JointBlockCache

THRESHOLD_BIASES = (0.5, 0.0, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875)
THRESHOLD_FAMILIES = (
    "block_full", "block_full_offset", "threshold_full", "threshold_full_offset",
)


def _quantize(coefficient, step, bias):
    if bias == 0.5:
        return (coefficient / step).round() * step
    return step * coefficient.sign() * (coefficient.abs() / step + bias).floor()


class ThresholdReplay(SupportCodecReplay):
    def threshold_prediction(self, source, steps, bias, h, w):
        bias = float(bias)
        if bias not in THRESHOLD_BIASES:
            raise ValueError("Threshold bias is outside the frozen finite grid")
        if bias == 0.5:
            return self.quantized_prediction(source, steps, h, w)
        quantized = _quantize(source, steps[:, None], bias)
        return self.to_rgb(self.restore_color(quantized, h, w))

    def apply(self, before, operator):
        kind, value = operator
        if (kind == "spatial" and isinstance(value, dict)
                and "threshold_block_steps" in value):
            source = self.coefficients(self.to_color(before))
            prediction = self.threshold_prediction(
                source, value["threshold_block_steps"], value["threshold_bias"],
                *before.shape[-2:])
            if "block_rgb_offset_8bit" in value:
                prediction = self.offset_prediction(
                    prediction, value["block_rgb_offset_8bit"])
            return prediction
        return super().apply(before, operator)


class ThresholdBlockCache(JointBlockCache):
    @torch.no_grad()
    def __init__(self, replay, before, after):
        if not isinstance(replay, ThresholdReplay):
            raise TypeError("ThresholdBlockCache requires ThresholdReplay")
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        super().__init__(replay, before, after)
        extra_wall, extra_cpu = time.perf_counter(), time.process_time()
        self.threshold_statistics = []
        for k in range(8):
            targets = [replay.coefficients(replay.to_color(target))
                       for target in self.targets[k]]
            # Branch zero aliases the original statistics; no recalculation and
            # no replacement of nearest-even by a half-up approximation.
            by_bias = [self.full_statistics[k]]
            for bias in THRESHOLD_BIASES[1:]:
                by_bias.append(self._threshold_statistics(
                    self.source_full[k], targets, bias))
            self.threshold_statistics.append(by_bias)
        self._synchronize()
        self.report.update(
            base_cache_version=self.report["version"],
            version="joint-threshold-block-v1",
            families=list(THRESHOLD_FAMILIES),
            threshold_biases=list(THRESHOLD_BIASES),
            threshold_bias_scope="One scalar shared by all 192 frequencies per operation",
            threshold_statistics_shape=[
                8, len(THRESHOLD_BIASES), self.slot_count, 255, 192],
            threshold_nearest_even_statistics="Direct alias of original full_statistics",
            threshold_additional_statistic_bytes=sum(
                value.numel() * value.element_size()
                for by_bias in self.threshold_statistics for value in by_bias[1:]),
            threshold_additional_wall_seconds=time.perf_counter() - extra_wall,
            threshold_additional_cpu_process_seconds=time.process_time() - extra_cpu,
            total_construction_wall_seconds=time.perf_counter() - wall_start,
            total_construction_cpu_process_seconds=time.process_time() - cpu_start,
            threshold_selection=(
                "Per bias, minimize each frequency's q objective over 1..255; "
                "sum its 192 minima; choose the first minimum in frozen bias order"),
            objective_omits_target_square=True,
        )

    def _threshold_statistics(self, source, targets, bias):
        statistics = torch.empty(
            (self.slot_count, len(self.steps), source.shape[0]),
            dtype=source.dtype, device=source.device)
        for start in range(0, len(self.steps), 32):
            stop = min(start + 32, len(self.steps))
            grid = self.steps[start:stop, None, None]
            templates = _quantize(source[None], grid, bias)
            energy = templates.square().sum(-1)
            for g, target in enumerate(targets):
                statistics[g, start:stop] = (
                    energy - 2 * torch.einsum("qfn,fn->qf", templates, target))
            del templates, energy
        if not torch.isfinite(statistics).all():
            raise ValueError("Nonfinite threshold statistics")
        return statistics

    @torch.no_grad()
    def candidates(self, include):
        indices = self._validate_include(include)
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        parent_models = super().candidates(indices)
        block_count = sum(self.source_full[k].shape[-1] for k in indices)
        bias_objectives, bias_steps = [], []
        for bias_index in range(len(THRESHOLD_BIASES)):
            per_content = [item[bias_index] for item in self.threshold_statistics]
            errors = self._ordered_sum(per_content, indices) / block_count
            if not torch.isfinite(errors).all():
                raise ValueError("Nonfinite aggregate threshold objective")
            minimum, step_index = errors.min(dim=1)
            bias_objectives.append(minimum.sum(dim=1))
            bias_steps.append(self.steps[step_index])
        objectives = torch.stack(bias_objectives, dim=1)
        selected_indices = objectives.argmin(dim=1)
        output, selected_biases = [], []
        for g in range(self.slot_count):
            selected = int(selected_indices[g])
            bias = THRESHOLD_BIASES[selected]
            steps = bias_steps[selected][g]
            threshold = ("spatial", {
                "threshold_block_steps": steps, "threshold_bias": bias})
            offsets = []
            for k in indices:
                prediction = self.replay.threshold_prediction(
                    self.source_full[k], steps, bias, self.height, self.width)
                offsets.append(((self.targets[k][g] - prediction) * 255).mean((1, 2)))
            offset = torch.stack(offsets).mean(0).round()
            if not torch.isfinite(offset).all():
                raise ValueError("Nonfinite threshold RGB offset")
            output.append({
                "block_full": parent_models[g]["block_full"],
                "block_full_offset": parent_models[g]["block_full_offset"],
                "threshold_full": threshold,
                "threshold_full_offset": ("spatial", {
                    "threshold_block_steps": steps, "threshold_bias": bias,
                    "block_rgb_offset_8bit": offset}),
            })
            selected_biases.append(bias)
        self._synchronize()
        # Keep the parent's include/fold record and add the complete threshold
        # stage, so every returned family has the same recorded training set.
        call = self.report["candidate_calls"][-1]
        call.update(
            base_generation_wall_seconds=call["wall_seconds"],
            base_generation_cpu_process_seconds=call["cpu_process_seconds"],
            wall_seconds=time.perf_counter() - wall_start,
            cpu_process_seconds=time.process_time() - cpu_start,
            selected_threshold_bias_by_slot=selected_biases,
            threshold_objective_by_slot_and_bias=objectives.cpu().tolist(),
            objective_units="Summed per-frequency coefficient objective; target-square term omitted",
        )
        return output

