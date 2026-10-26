"""Build a support-only pool of fixed-base shared residual corrections.

No query, candidate history, semantic metadata, or endpoint enters this module.
The fixed bases already saw all supports: this is conditional fitting, not OOF.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import torch

from chainprint_shared_residual import (
    ResidualReplay, features, fit_coefficients,
)

PARETO_TOLERANCE = 1e-10


def _frame(tag, payload):
    return tag + len(payload).to_bytes(8, "big") + payload


def _parameter_bytes(value):
    """Typed, ordered serialization; never use output equality for deduplication."""
    if isinstance(value, torch.nn.Module):
        if type(value).__name__ != "QuantizedSeparableResponse":
            raise TypeError("Unsupported saved module: " + type(value).__name__)
        identity = type(value).__module__ + "." + type(value).__qualname__
        return _frame(b"U", _parameter_bytes((identity, value.training, dict(value.state_dict()))))
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError("Parameter hash requires a dense, nonquantized tensor")
        tensor = value.detach().cpu().contiguous()
        metadata = (str(tensor.dtype) + ":" + repr(tuple(tensor.shape))).encode()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return _frame(b"T", _frame(b"M", metadata) + _frame(b"B", raw))
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported parameter values")
        array = np.ascontiguousarray(value)
        metadata = (array.dtype.str + ":" + repr(tuple(array.shape))).encode()
        return _frame(b"A", _frame(b"M", metadata) + _frame(b"B", array.tobytes()))
    if isinstance(value, np.generic):
        return _frame(b"N", value.dtype.str.encode() + b":" + value.tobytes())
    if value is None:
        return _frame(b"0", b"")
    if isinstance(value, bool):
        return _frame(b"b", b"1" if value else b"0")
    if isinstance(value, int):
        return _frame(b"i", str(value).encode())
    if isinstance(value, float):
        return _frame(b"f", struct.pack(">d", value))
    if isinstance(value, str):
        return _frame(b"s", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _frame(b"y", value)
    if isinstance(value, (list, tuple)):
        return _frame(b"l" if isinstance(value, list) else b"t",
                      b"".join(_parameter_bytes(item) for item in value))
    if isinstance(value, dict):
        items = sorted((_parameter_bytes(k), _parameter_bytes(v))
                       for k, v in value.items())
        return _frame(b"d", b"".join(_frame(b"k", k) + _frame(b"v", v)
                                    for k, v in items))
    raise TypeError("Unsupported parameter type: " + type(value).__name__)


def parameter_hash(value):
    return hashlib.sha256(_parameter_bytes(value)).hexdigest()


def _pareto_indices(losses):
    """Keep points not strictly dominated under the declared absolute tolerance."""
    retained = []
    for index, candidate in enumerate(losses):
        dominated = any(
            other != index
            and np.all(losses[other] <= candidate + PARETO_TOLERANCE)
            and np.any(losses[other] < candidate - PARETO_TOLERANCE)
            for other in range(len(losses))
        )
        if not dominated:
            retained.append(index)
    if not retained:
        # Tolerance-based dominance can in general contain cycles. Do not silently
        # insert a query-dependent fallback or claim an empty pool is valid.
        raise ValueError("No Pareto survivors under the frozen tolerance")
    return retained


@torch.no_grad()
def build_support_correction_pool(replay, before, after, default_ops):
    """Return retained_models, report, collapsed_model, support_best_model.

    Each leave-one tuple omits the same content from every adopted correction.
    Base operators and uncorrected slots stay fixed. Full-support refitting is
    recorded for consistency checking; it does not replace the saved baseline.
    """
    before_shape, after_shape = tuple(before.shape), tuple(after.shape)
    defaults = list(default_ops)
    if len(before_shape) != 4 or before_shape[0] != 8 or before_shape[1] != 3:
        raise ValueError("Expected eight RGB supports: before[8,3,H,W]")
    if after_shape != (8, len(defaults), *before_shape[1:]) or not defaults:
        raise ValueError("Expected after[8,G,3,H,W] with matching default slots")
    count, slots = 8, len(defaults)
    images = [replay.image(before[k]) for k in range(count)]
    targets = [[replay.image(after[k][g]) for g in range(slots)]
               for k in range(count)]
    adopted = [g for g, op in enumerate(defaults) if op[0] == "shared_residual"]
    correction_folds, full_checks = {}, []
    for g in adopted:
        underlying, saved = defaults[g][1]
        if not isinstance(saved, torch.Tensor) or tuple(saved.shape) != (31, 3):
            raise ValueError("Saved shared correction must be a 31x3 tensor")
        if not torch.isfinite(saved).all():
            raise ValueError("Saved correction contains nonfinite coefficients")
        predictions = [replay.apply(image, underlying) for image in images]
        designs = [features(image, prediction)
                   for image, prediction in zip(images, predictions)]
        residuals = [((targets[k][g].to(torch.float64) - predictions[k]) * 255)
                     .reshape(3, -1).T for k in range(count)]
        refit = fit_coefficients(designs, residuals, list(range(count)))
        difference = float((refit - saved.to(device=refit.device,
                                             dtype=refit.dtype)).abs().max())
        full_checks.append(dict(
            slot=g, max_absolute_coefficient_difference=difference,
            saved_dtype=str(saved.dtype), refit_dtype=str(refit.dtype),
            shape=list(saved.shape),
            policy="Recorded only; runner determines reproduction acceptance.",
        ))
        correction_folds[g] = [
            fit_coefficients(designs, residuals,
                             [k for k in range(count) if k != held])
            for held in range(count)
        ]
        del predictions, designs, residuals

    proposed = [(defaults, dict(kind="saved_default", fit_contents=list(range(count))))]
    if adopted:
        for held in range(count):
            model = list(defaults)
            for g in adopted:
                model[g] = ("shared_residual",
                            (defaults[g][1][0], correction_folds[g][held]))
            proposed.append((model, dict(
                kind="leave_one_content_correction", held_content=held,
                fit_contents=[k for k in range(count) if k != held],
            )))

    models, records, seen = [], [], {}
    for model, origin in proposed:
        fingerprint = parameter_hash(model)
        if fingerprint in seen:
            records[seen[fingerprint]]["sources"].append(origin)
        else:
            index = len(models)
            seen[fingerprint] = index
            models.append(model)
            records.append(dict(model_index=index, parameter_sha256=fingerprint,
                                sources=[origin]))
    default_index = seen[parameter_hash(defaults)]

    wrapped = ResidualReplay(replay)
    operator_losses = {}
    losses = np.empty((len(models), count), dtype=np.float64)
    for index, model in enumerate(models):
        per_slot = []
        for g, operator in enumerate(model):
            key = (g, parameter_hash(operator))
            if key not in operator_losses:
                values = []
                for k in range(count):
                    prediction = wrapped.apply(images[k], operator).to(torch.float64)
                    target = targets[k][g].to(torch.float64)
                    values.append(float(((prediction - target) * 255).square().mean()))
                operator_losses[key] = np.asarray(values, dtype=np.float64)
            per_slot.append(operator_losses[key])
        losses[index] = np.mean(per_slot, axis=0)
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite support loss; no retention decision produced")

    retained_indices = _pareto_indices(losses)
    retained = [models[index] for index in retained_indices]
    means = losses.mean(axis=1)
    best_index = min(retained_indices, key=lambda index: (means[index], index))
    collapsed = list(defaults)
    for g in adopted:
        coefficients = [model[g][1][1] for model in retained]
        if len(coefficients) == 1:
            averaged = coefficients[0].clone()
        else:
            averaged = torch.stack(coefficients).mean(dim=0)
        collapsed[g] = ("shared_residual", (defaults[g][1][0], averaged))
    for index, record in enumerate(records):
        record["support_mse_8bit_by_content"] = losses[index].tolist()
        record["support_mean_mse_8bit"] = float(means[index])
        record["retained"] = index in retained_indices

    report = dict(
        version="fixed-base-correction-pool-v1",
        support_count=count, operation_count=slots, adopted_slots=adopted,
        proposed_count=len(proposed), distinct_parameter_tuples=len(models),
        retained_count=len(retained), retained_indices=retained_indices,
        default_model_index=default_index, default_retained=default_index in retained_indices,
        support_best_model_index=best_index,
        full_refit_checks=full_checks,
        full_refit_max_absolute_coefficient_difference=max(
            [entry["max_absolute_coefficient_difference"] for entry in full_checks],
            default=0.0),
        records=records, support_losses_mse_8bit=losses.tolist(),
        loss_rule="Per content: mean over all slots of whole-image 8-bit MSE.",
        pareto_tolerance_mse_8bit=PARETO_TOLERANCE,
        pareto_rule="other <= candidate + tolerance everywhere and < candidate - tolerance somewhere",
        deduplication="Complete typed parameter tuple SHA256; no support-output deduplication",
        collapsed_rule="Equal mean of retained distinct coefficient tuples, with bases fixed",
        weights="Uniform over retained parameter-distinct tuples, independent of source multiplicity",
        information_boundary="Only operation before/after supports and saved default operations",
        validation_scope="Fixed bases saw all eight contents; conditional refitting, NOT strict OOF",
        coverage="Finite empirical alternatives, not a confidence set or exhaustive operator coverage",
    )
    return dict(retained_models=retained, report=report, collapsed_model=collapsed,
                support_best_model=models[best_index])
