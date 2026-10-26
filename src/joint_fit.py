"""Joint support fitting in the frozen replay library; no query inputs.

Parameter objectives and terminal-refit masks follow the frozen library. Family
selection uses equal-content full-image MSE, as the pooled baseline does.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch

FAMILIES = (
    "spatial_linear", "spatial_gain", "spatial_deadzone", "color_affine",
    "pointwise_quantizer", "block_full", "block_half", "block_full_offset",
    "block_half_offset", "resampling", "jpeg_codec", "quantized_separable",
)
FIT_VERSION = "opweave-joint-support-v1"
_GRID_CACHE_LIMIT = 32
_GRID_STATS_CACHE = OrderedDict()
_GRID_STATS_OWNER = None


def _clear_grid_cache():
    """Drop bounded, support-only grid statistics (also useful for cold timing)."""
    global _GRID_STATS_OWNER
    _GRID_STATS_CACHE.clear()
    _GRID_STATS_OWNER = None


def _grid_statistics(replay, image, source, target_group, observed, family, grid, sample):
    """Cache per-parameter/per-slot MSE, never a fit or a held-fold aggregate.

    Entries use every raw support/target pixel, including unselected pixels and
    all slots. Reusing a seven-content fold therefore only reads its seven own
    statistics. Division and accumulation remain the caller's original loops.
    """
    global _GRID_STATS_OWNER
    if _GRID_STATS_OWNER is not replay:
        _clear_grid_cache()
        _GRID_STATS_OWNER = replay
    digest = hashlib.sha256()
    # Include raw dtype and shape, before byte rounding or pixel subsampling.
    for value in (source, *target_group):
        value = np.ascontiguousarray(value.detach().cpu().numpy())
        header = repr((value.dtype.str, value.shape)).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.tobytes())
    digest.update(repr((family, grid, tuple(replay.filters.items()),
                        "pillow-jpeg:optimize=False,progressive=False")).encode())
    digest.update(repr((sample.dtype.str, sample.shape)).encode())
    digest.update(sample.tobytes())
    # Renderer identity prevents stale reuse after a method is replaced. The
    # owner guard avoids assuming equal configuration across replay instances.
    methods = tuple(getattr(getattr(replay, name), "__func__", getattr(replay, name))
                    for name in ("pil_image", "roundtrip", "codec_bytes"))
    key = (methods, digest.digest())
    if key in _GRID_STATS_CACHE:
        _GRID_STATS_CACHE.move_to_end(key)
        return _GRID_STATS_CACHE[key]
    statistics = np.empty((len(grid), len(target_group)), dtype=np.float64)
    for index, parameter in enumerate(grid):
        if family == "resampling":
            name, size = parameter
            predicted = np.asarray(replay.roundtrip(image, size, name)).transpose(2, 0, 1)
        else:
            predicted = replay.codec_bytes(image, *parameter)
        values = predicted.reshape(3, -1)[:, sample].astype(np.float64)
        statistics[index] = np.square(values[None] - observed).mean((1, 2))
    statistics.setflags(write=False)
    _GRID_STATS_CACHE[key] = statistics
    if len(_GRID_STATS_CACHE) > _GRID_CACHE_LIMIT:
        _GRID_STATS_CACHE.popitem(last=False)
    return statistics


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite " + label)
    return value


def merged_ridge(designs, responses, ridge=1e-3, center=False):
    """Concatenated-row fit using per-image sufficient statistics.

    Ridge is applied to the mean Gram, exactly as frozen spatial/color fit.
    Returns coefficient, feature mean, response mean; means are zero if uncentred.
    """
    if not designs or len(designs) != len(responses):
        raise ValueError("Paired design/response groups required")
    dim, out = designs[0].shape[1], responses[0].shape[1]
    kw = dict(dtype=designs[0].dtype, device=designs[0].device)
    gram, rhs = torch.zeros((dim, dim), **kw), torch.zeros((dim, out), **kw)
    sx, sy = torch.zeros(dim, **kw), torch.zeros(out, **kw)
    count = 0
    for x, y in zip(designs, responses):
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("Aligned row matrices required")
        gram += x.T @ x
        rhs += x.T @ y
        sx += x.sum(0)
        sy += y.sum(0)
        count += x.shape[0]
    if not count:
        return rhs, sx, sy
    xm, ym = sx / count, sy / count
    gram, rhs = gram / count, rhs / count
    if center:
        gram -= xm[:, None] * xm[None, :]
        rhs -= xm[:, None] * ym[None, :]
    else:
        xm, ym = torch.zeros_like(xm), torch.zeros_like(ym)
    coef = torch.linalg.solve(gram + ridge * torch.eye(dim, **kw), rhs)
    return _finite(coef, "joint ridge coefficients"), xm, ym


def merged_quantizer(replay, images, targets):
    """Frozen all-pixel quantizer objective, merged across support contents."""
    counts, sums = np.zeros(256), np.zeros(256)
    for before, after in zip(images, targets):
        index = (before * 255).round().long().cpu().numpy().ravel()
        target = (after * 255).round().cpu().numpy().ravel()
        counts += np.bincount(index, minlength=256)
        sums += np.bincount(index, weights=target, minlength=256)
    count = torch.as_tensor(counts, device=replay.device, dtype=images[0].dtype)
    total = torch.as_tensor(sums, device=replay.device, dtype=images[0].dtype)
    errors = (replay.luts.square() * count - 2 * replay.luts * total).sum(-1)
    return int(errors.argmin())


def merged_steps(replay, sources, targets):
    """Shared DCT step per frequency; same 1..255 grid for all supports.

    Target-square terms cancel as in frozen estimate_steps. Source rows are
    frequencies and columns are blocks. Images are never tiled into a mosaic.
    """
    errors, count = None, 0
    for source, target in zip(sources, targets):
        if source.shape != target.shape:
            raise ValueError("Source/target coefficient shape differs")
        templates = (source[None] / replay.steps[:, None, None]).round() * replay.steps[:, None, None]
        current = templates.square().sum(-1) - 2 * torch.einsum("qfn,fn->qf", templates, target)
        errors = current if errors is None else errors + current
        count += source.shape[-1]
    if not count:
        raise ValueError("No DCT blocks")
    return replay.steps[_finite(errors / count, "step losses").argmin(0)]


def _merged_gain(delta, target):
    denominator = sum(float(x.square().sum()) for x in delta)
    if denominator <= 1e-12:
        return 1.0
    return max(0.0, sum(float((x * y).sum()) for x, y in zip(delta, target)) / denominator)


def _merged_deadzone(delta, target):
    # Global sorting is required; averaging separately fitted thresholds is wrong.
    x = np.concatenate([v.cpu().numpy() for v in delta])
    y = np.concatenate([v.cpu().numpy() for v in target])
    if not len(x):
        return 1.0, 0.0
    order = np.argsort(-np.abs(x), kind="stable")
    x, y = x[order], y[order]
    magnitude = np.abs(x)
    xy, xx = np.cumsum(x * y), np.cumsum(x * x)
    ends = np.flatnonzero(np.r_[magnitude[:-1] != magnitude[1:], True])
    ends = ends[xx[ends] > 1e-12]
    if not len(ends):
        return 1.0, 0.0
    explained = np.maximum(xy[ends], 0) ** 2 / xx[ends]
    best = ends[np.flatnonzero(explained == explained.max())[-1]]
    gain = max(0.0, float(xy[best] / xx[best]))
    threshold = float((magnitude[best] + magnitude[best + 1]) / 2) if best + 1 < len(x) else 0.0
    return gain, threshold


def _continuous(replay, images, targets, bases, colors, requested):
    result = {}
    if requested & {"spatial_linear", "spatial_gain", "spatial_deadzone"}:
        eligible = [(t > 0) & (t < 1) for t in targets]
        designs = [b[:, m].T for b, m in zip(bases, eligible)]
        responses = [(t - x)[m][:, None] for x, t, m in zip(images, targets, eligible)]
        coefficient = merged_ridge(designs, responses)[0][:, 0]
        result["spatial_linear"] = ("spatial", coefficient)
        delta = [torch.einsum("k,kchw->chw", coefficient, b)[m] for b, m in zip(bases, eligible)]
        target = [y[:, 0] for y in responses]
        result["spatial_gain"] = ("spatial", coefficient * _merged_gain(delta, target))
        gain, threshold = _merged_deadzone(delta, target)
        result["spatial_deadzone"] = ("spatial", {"coef": coefficient, "gain": gain, "threshold": threshold})
    if "color_affine" in requested:
        eligible = [((t > 0) & (t < 1)).all(0) for t in targets]
        designs = [f[:, m].T for f, m in zip(colors, eligible)]
        responses = [(t - x)[:, m].T for x, t, m in zip(images, targets, eligible)]
        coefficient, xm, ym = merged_ridge(designs, responses, center=True)
        result["color_affine"] = ("spatial", {"color_coef": coefficient, "feature_mean": xm, "delta_mean": ym})
    if "pointwise_quantizer" in requested:
        result["pointwise_quantizer"] = ("pointwise", merged_quantizer(replay, images, targets))
    return result


def _block_families(replay, images, targets, source_full, source_half, requested):
    result = {}
    target_colors = [replay.to_color(y) for y in targets]
    for half, label, key in ((False, "block_full", "block_quantization_steps"),
                             (True, "block_half", "half_chroma_steps")):
        if not requested & {label, label + "_offset"}:
            continue
        if half:
            target_coefs = [replay.target_coefficients(y) for y in target_colors]
            steps = torch.stack([merged_steps(replay, [s[p] for s in source_half],
                                             [t[p] for t in target_coefs]) for p in range(3)])
        else:
            steps = merged_steps(replay, source_full, [replay.coefficients(y) for y in target_colors])
        operator = ("spatial", {key: steps})
        result[label] = operator
        if label + "_offset" in requested:
            predictions = [replay.apply(x, operator) for x in images]
            offset = torch.stack([((t - p) * 255).mean((1, 2)) for t, p in zip(targets, predictions)]).mean(0).round()
            result[label + "_offset"] = ("spatial", {key: steps, "block_rgb_offset_8bit": offset})
    return result


def _grid_fits(replay, images, target_groups, requested):
    """Search common image-level grids once for all anonymous output slots."""
    h, w = images[0].shape[-2:]
    yy, xx = np.indices((h, w))
    train = (yy + xx) % 4 != 0
    pil = [replay.pil_image(x.cpu().numpy()) for x in images]
    targets = [np.rint(np.stack([y.cpu().numpy() for y in group]) * 255) for group in target_groups]
    slots = len(target_groups[0])
    fitted, records = {}, {}
    for family in ("resampling", "jpeg_codec"):
        if family not in requested:
            continue
        if family == "resampling":
            sample = np.linspace(0, h*w - 1, min(1024, h*w), dtype=np.int64)
            sample = sample[train.ravel()[sample]]
            grid, seen = [], set()
            for name in replay.filters:
                for latent in range(1, max(h, w)):
                    scale = latent / max(h, w)
                    size = (max(1, round(w * scale)), max(1, round(h * scale)))
                    if (name, size) not in seen:
                        grid.append((name, size))
                        seen.add((name, size))
        else:
            sample = np.flatnonzero(train.ravel())
            sample = sample[np.linspace(0, len(sample) - 1, min(1024, len(sample)), dtype=np.int64)]
            grid = [(quality, subsampling) for subsampling in (0, 1, 2) for quality in range(1, 101)]
        if not len(sample) or not grid:
            raise ValueError("No fitting pixels or hypotheses for " + family)
        observed = [y.reshape(slots, 3, -1)[:, :, sample] for y in targets]
        statistics = [_grid_statistics(replay, image, source, group, target, family, grid, sample)
                      for image, source, group, target in zip(pil, images, target_groups, observed)]
        best, parameters = np.full(slots, np.inf), [None] * slots
        for index, parameter in enumerate(grid):
            losses = np.zeros(slots)
            for per_content in statistics:
                losses += per_content[index] / len(images)
            for g in np.flatnonzero(losses < best):
                best[g], parameters[g] = losses[g], parameter
        operators = []
        for parameter in parameters:
            if family == "resampling":
                name, size = parameter
                value = {"resampling_size": list(size), "resampling_source_size": [w, h], "resampling_filter": name}
            else:
                quality, subsampling = parameter
                value = {"codec_quality": quality, "codec_subsampling": subsampling}
            operators.append(("spatial", value))
        fitted[family] = operators
        records[family] = {"grid_size": len(grid), "fit_pixels_per_content": len(sample), "train_mse_8bit": best.tolist()}
    return fitted, records


def _quantized_separable(replay, images, targets):
    """Shared terminal Adam iterate; same surrogate/budget, not a global optimum."""
    from history_recovery.chainprint_quantized_separable import QuantizedSeparableResponse

    model = QuantizedSeparableResponse(replay.device)
    xs = [(x.to(torch.float32) * 255).round() for x in images]
    ys = [(y.to(torch.float32) * 255).round() for y in targets]
    h, w = images[0].shape[-2:]
    yy = torch.arange(h, device=replay.device)[:, None]
    xx = torch.arange(w, device=replay.device)[None, :]
    mask = ((yy + xx) % 4 != 0).expand_as(xs[0])
    with torch.no_grad():
        deltas = [model.lowpass(x) - x for x in xs]
        denominator = sum(d[mask].square().sum() for d in deltas).clamp_min(1e-12)
        numerator = sum((d[mask] * (y - x)[mask]).sum() for x, y, d in zip(xs, ys, deltas))
        model.gain.copy_(numerator / denominator)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    with torch.enable_grad():
        for _ in range(128):
            optimizer.zero_grad(set_to_none=True)
            terminal_loss = 0.0
            for x, y in zip(xs, ys):
                prediction = model(x, surrogate=True)
                loss = (prediction[mask] - y[mask]).square().mean() / len(xs)
                loss.backward()
                terminal_loss += float(loss.detach())
            optimizer.step()
            with torch.no_grad():
                model.threshold.clamp_(min=0)
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        _finite(parameter, "quantized separable parameters")
    return ("spatial", {"quantized_separable_model": model}), terminal_loss


def _signature(images, targets):
    digest = hashlib.sha256()
    for x, group in zip(images, targets):
        for image in (x, *group):
            array = image.detach().cpu().numpy().astype("<f8", copy=False)
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _read_support(replay, before, after, allowed_counts=(8,)):
    if len(before) not in allowed_counts or len(after) != len(before):
        raise ValueError("Joint support count must be " + repr(allowed_counts))
    images = [replay.image(x) for x in before]
    targets = [[replay.image(y) for y in group] for group in after]
    slots = len(targets[0])
    if not slots or any(len(group) != slots for group in targets):
        raise ValueError("Aligned nonempty anonymous output slots required")
    if any(y.shape != images[0].shape for y in images + [y for group in targets for y in group]):
        raise ValueError("All supports must have the same CHW shape")
    return images, targets


@torch.no_grad()
def _fit_joint(replay, before, after, families):
    images, targets = _read_support(replay, before, after, allowed_counts=(7, 8))
    requested = set(families)
    if not requested or requested - set(FAMILIES) or len(requested) != len(families):
        raise ValueError("Nonempty unique frozen family names required")
    order = [family for family in FAMILIES if family in requested]
    blocks = requested & {"block_full", "block_half", "block_full_offset", "block_half_offset"}
    h, w = images[0].shape[-2:]
    if blocks and (h % 16 or w % 16):
        raise ValueError("Frozen block library requires dimensions divisible by 16")
    continuous = requested & {"spatial_linear", "spatial_gain", "spatial_deadzone", "color_affine"}
    bases = [replay.basis(x) for x in images] if continuous else []
    colors = [replay.color_features(x, b) for x, b in zip(images, bases)] if "color_affine" in requested else []
    image_colors = [replay.to_color(x) for x in images] if blocks else []
    source_full = [replay.coefficients(x) for x in image_colors] if requested & {"block_full", "block_full_offset"} else []
    source_half = [replay.source_coefficients(x) for x in image_colors] if requested & {"block_half", "block_half_offset"} else []
    grids, grid_records = _grid_fits(replay, images, targets, requested) if requested & {"resampling", "jpeg_codec"} else ({}, {})
    selected, records = [], []
    signature = _signature(images, targets)
    for g in range(len(targets[0])):
        ys = [group[g] for group in targets]
        candidates = _continuous(replay, images, ys, bases, colors, requested)
        if blocks:
            candidates.update(_block_families(replay, images, ys, source_full, source_half, requested))
        candidates.update({name: operators[g] for name, operators in grids.items()})
        surrogate_loss = None
        if "quantized_separable" in requested:
            candidates["quantized_separable"], surrogate_loss = _quantized_separable(replay, images, ys)
        losses = [[float(((replay.apply(x, candidates[name]).to(torch.float64) - y) * 255).square().mean())
                   for x, y in zip(images, ys)] for name in order]
        panel = np.asarray(losses)
        if not np.isfinite(panel).all():
            raise ValueError("Nonfinite joint family MSE")
        best = int(panel.mean(1).argmin())
        selected.append(candidates[order[best]])
        records.append({"fit_version": FIT_VERSION, "slot": g, "support_count": len(images),
                        "support_signature_sha256": signature, "family": order[best], "families": order,
                        "selected": best, "support_mse_8bit": losses,
                        "mean_support_mse_8bit": panel.mean(1).tolist(),
                        "selection_rule": "lowest equal-content full-image 8-bit MSE; canonical fixed-order ties",
                        "complete_library": tuple(order) == FAMILIES,
                        "grid_diagnostics": {name: {**data, "train_mse_8bit": data["train_mse_8bit"][g]}
                                             for name, data in grid_records.items()},
                        "quantized_separable_terminal_surrogate_loss": surrogate_loss,
                        "optimizer_scope": "Analytic/same-grid joint fits; quantized separable is a 128-step nonconvex STE approximation; see JOINT_FIT_NOTES.md",
                        "selection_scope": "Support fitting/selection only, not independent validation"})
    return selected, records


def fit_joint_operators(replay, before, after, *, families: Sequence[str] = FAMILIES):
    """K=8 joint fit; return replay-compatible operators and JSON records.

    Reduced family sets are only for narrow tests/diagnostics and visibly marked
    complete_library=False. Registered comparisons must use the defaults.
    """
    if len(before) != 8:
        raise ValueError("The comparison is fixed at K=8 supports")
    return _fit_joint(replay, before, after, families)


def _residual_module():
    # Runner resolves the import to the frozen deployed helper: same feature map,
    # pseudoinverse tolerance, round/clamp behavior, and residual capacity.
    import chainprint_shared_residual
    return chainprint_shared_residual


@torch.no_grad()
def _residual_fit(replay, images, targets, operator):
    residual = _residual_module()
    predictions = [replay.apply(x, operator) for x in images]
    xs = [residual.features(x, prediction) for x, prediction in zip(images, predictions)]
    ys = [((target - prediction) * 255).reshape(3, -1).T for target, prediction in zip(targets, predictions)]
    return residual.fit_coefficients(xs, ys, list(range(len(images))))


@torch.no_grad()
def calibrate_joint_operators(replay, before, after, joint, joint_records, *, families=None):
    """Same-capacity correction, rebuilding every held-content fold from seven.

    Held source/target is only scored AFTER that fold's base operator and residual
    coefficients have been fitted. These eight losses select adoption, not a test.
    """
    images, targets = _read_support(replay, before, after)
    if len(joint) != len(targets[0]) or len(joint_records) != len(joint):
        raise ValueError("Joint operators/records must match anonymous slots")
    signature = _signature(images, targets)
    inferred = tuple(joint_records[0]["families"])
    families = inferred if families is None else tuple(families)
    if any(record["support_signature_sha256"] != signature or record["support_count"] != 8
           or tuple(record["families"]) != families for record in joint_records):
        raise ValueError("Joint fitting inputs or library changed before calibration")
    residual = _residual_module()
    baseline, corrected = [[] for _ in joint], [[] for _ in joint]
    fold_records = [[] for _ in joint]
    for held in range(8):
        include = [j for j in range(8) if j != held]
        fold_before, fold_after = [before[j] for j in include], [after[j] for j in include]
        fold_operators, fold_diagnostics = _fit_joint(replay, fold_before, fold_after, families)
        training_images = [images[j] for j in include]
        for g, operator in enumerate(fold_operators):
            coefficients = _residual_fit(replay, training_images, [targets[j][g] for j in include], operator)
            prediction = replay.apply(images[held], operator)
            fixed = residual.corrected(prediction, residual.features(images[held], prediction), coefficients)
            target = targets[held][g]
            baseline[g].append(float(((prediction.to(torch.float64) - target) * 255).square().mean()))
            corrected[g].append(float(((fixed.to(torch.float64) - target) * 255).square().mean()))
            fold_records[g].append({"held_content": held, "fit_contents": include,
                                    "fit_count": 7, "family": fold_diagnostics[g]["family"],
                                    "training_signature_sha256": fold_diagnostics[g]["support_signature_sha256"],
                                    "family_mean_support_mse_8bit": fold_diagnostics[g]["mean_support_mse_8bit"]})
    output, records = list(joint), []
    for g, operator in enumerate(joint):
        base, new = np.asarray(baseline[g]), np.asarray(corrected[g])
        if not np.isfinite(base).all() or not np.isfinite(new).all():
            raise ValueError("Nonfinite leave-one-content-out losses")
        adopt = bool(np.all(new <= base) and new.mean() < base.mean())
        coefficient_record = None
        if adopt:
            coefficients = _residual_fit(replay, images, [group[g] for group in targets], operator)
            output[g] = ("shared_residual", (operator, coefficients))
            coefficient_record = coefficients.cpu().tolist()
        records.append({"adopted": adopt, "held_baseline_mse_8bit": baseline[g],
                        "held_corrected_mse_8bit": corrected[g], "coefficients": coefficient_record,
                        "features": 31, "base_parameter_fit_count_per_fold": 7,
                        "leave_one_content_out": fold_records[g],
                        "validation_scope": "Each base model and family selection is refitted on seven contents; same 31-feature residual uses those seven only. Eight held folds select adoption and are not an independent test.",
                        "adoption_rule": "all eight corrected losses <= baseline; strictly smaller arithmetic mean; no tolerance"})
    return output, records
