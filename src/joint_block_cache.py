"""Contentwise sufficient statistics for the frozen joint block families.

Only anonymous operation supports enter this cache. A fold aggregates exactly its
listed contents; precomputing other contents does not make them training inputs.
"""
from __future__ import annotations

import numbers
import time

import torch

BLOCK_FAMILIES = (
    "block_full", "block_half", "block_full_offset", "block_half_offset",
)


class JointBlockCache:
    @torch.no_grad()
    def __init__(self, replay, before, after):
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
        self.replay = replay
        self.slot_count = after_shape[1]
        self.images = [replay.image(before[k]) for k in range(8)]
        self.targets = [[replay.image(after[k][g]) for g in range(self.slot_count)]
                        for k in range(8)]
        self.steps = replay.steps
        expected = torch.arange(1, 256, device=self.steps.device, dtype=self.steps.dtype)
        if self.steps.shape != (255,) or not torch.equal(self.steps, expected):
            raise ValueError("The frozen step grid must be integers 1..255")
        if self.steps.dtype != torch.float64:
            raise ValueError("Frozen block fitting uses float64")
        self.source_full, self.source_half = [], []
        self.full_statistics, self.half_statistics = [], []
        for k, image in enumerate(self.images):
            color = replay.to_color(image)
            source_full = replay.coefficients(color)
            source_half = replay.source_coefficients(color)
            target_colors = [replay.to_color(target) for target in self.targets[k]]
            target_full = [replay.coefficients(value) for value in target_colors]
            target_half = [replay.target_coefficients(value) for value in target_colors]
            self.source_full.append(source_full)
            self.source_half.append(source_half)
            self.full_statistics.append(self._statistics(source_full, target_full))
            half = [self._statistics(source_half[plane],
                                     [target[plane] for target in target_half])
                    for plane in range(3)]
            self.half_statistics.append(torch.stack(half, dim=1))
        self._synchronize()
        self.report = dict(
            version="joint-block-content-statistics-v1",
            support_count=8, operation_count=self.slot_count,
            support_shape=list(before_shape), families=list(BLOCK_FAMILIES),
            full_statistics_shape=[8, *self.full_statistics[0].shape],
            half_statistics_shape=[8, *self.half_statistics[0].shape],
            statistic_dtype=str(self.full_statistics[0].dtype),
            statistic_device=str(self.full_statistics[0].device),
            step_grid="Integers 1..255", grid_chunk_size=32,
            objective="sum(template^2 - 2*template*target), per frequency and content",
            aggregation="Ordered sum over include, then division by included block counts",
            offset="Mean RGB residual per included content, equal-content mean, integer round",
            construction_wall_seconds=time.perf_counter() - wall_start,
            construction_cpu_process_seconds=time.process_time() - cpu_start,
            cached_statistic_bytes=sum(
                value.numel() * value.element_size()
                for value in self.full_statistics + self.half_statistics),
            candidate_calls=[],
            information_boundary="Supports only; no query, bank, semantic labels, or audit metadata",
        )

    def _synchronize(self):
        if self.steps.device.type == "cuda":
            torch.cuda.synchronize(self.steps.device)

    def _statistics(self, source, targets):
        """Return [slot, step, frequency], matching merged_steps' objective."""
        if source.ndim != 2 or any(target.shape != source.shape for target in targets):
            raise ValueError("Source and target coefficient dimensions differ")
        statistics = torch.empty(
            (self.slot_count, len(self.steps), source.shape[0]),
            device=source.device, dtype=source.dtype)
        for start in range(0, len(self.steps), 32):
            stop = min(start + 32, len(self.steps))
            grid = self.steps[start:stop, None, None]
            templates = (source[None] / grid).round() * grid
            energy = templates.square().sum(-1)
            for g, target in enumerate(targets):
                statistics[g, start:stop] = (
                    energy - 2 * torch.einsum("qfn,fn->qf", templates, target))
            del templates, energy
        if not torch.isfinite(statistics).all():
            raise ValueError("Nonfinite per-content block statistics")
        return statistics

    @staticmethod
    def _ordered_sum(values, include):
        total = values[include[0]].clone()
        for index in include[1:]:
            total = total + values[index]
        return total

    @staticmethod
    def _validate_include(include):
        indices = list(include)
        if (len(indices) not in (7, 8)
                or any(isinstance(k, bool) or not isinstance(k, numbers.Integral)
                       for k in indices)):
            raise ValueError("include must list seven or eight integer content indices")
        indices = [int(k) for k in indices]
        if len(set(indices)) != len(indices) or any(k < 0 or k >= 8 for k in indices):
            raise ValueError("Content indices must be unique and in 0..7")
        return indices

    @torch.no_grad()
    def candidates(self, include):
        """Return one four-family dict per anonymous slot, in fixed key order."""
        indices = self._validate_include(include)
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        full_errors = self._ordered_sum(self.full_statistics, indices)
        full_count = sum(self.source_full[k].shape[-1] for k in indices)
        full_errors = full_errors / full_count
        half_errors = self._ordered_sum(self.half_statistics, indices)
        for plane in range(3):
            count = sum(self.source_half[k][plane].shape[-1] for k in indices)
            half_errors[:, plane] = half_errors[:, plane] / count
        if not torch.isfinite(full_errors).all() or not torch.isfinite(half_errors).all():
            raise ValueError("Nonfinite aggregated block objective")
        full_steps = self.steps[full_errors.argmin(dim=1)]
        half_steps = self.steps[half_errors.argmin(dim=2)]
        output = []
        for g in range(self.slot_count):
            full = ("spatial", {"block_quantization_steps": full_steps[g]})
            half = ("spatial", {"half_chroma_steps": half_steps[g]})
            full_offsets, half_offsets = [], []
            for k in indices:
                # These are precisely the full/half branches reached by
                # replay.apply; cached source transforms avoid recomputing them.
                prediction_full = self.replay.quantized_prediction(
                    self.source_full[k], full_steps[g], self.height, self.width)
                prediction_half = self.replay.predict_multires(
                    self.source_half[k], half_steps[g], self.height, self.width)
                target = self.targets[k][g]
                full_offsets.append(((target - prediction_full) * 255).mean((1, 2)))
                half_offsets.append(((target - prediction_half) * 255).mean((1, 2)))
            full_offset = torch.stack(full_offsets).mean(0).round()
            half_offset = torch.stack(half_offsets).mean(0).round()
            if not torch.isfinite(full_offset).all() or not torch.isfinite(half_offset).all():
                raise ValueError("Nonfinite RGB offsets")
            output.append({
                "block_full": full,
                "block_half": half,
                "block_full_offset": ("spatial", {
                    "block_quantization_steps": full_steps[g],
                    "block_rgb_offset_8bit": full_offset}),
                "block_half_offset": ("spatial", {
                    "half_chroma_steps": half_steps[g],
                    "block_rgb_offset_8bit": half_offset}),
            })
        self._synchronize()
        self.report["candidate_calls"].append(dict(
            include=indices, fit_count=len(indices),
            wall_seconds=time.perf_counter() - wall_start,
            cpu_process_seconds=time.process_time() - cpu_start,
        ))
        return output

