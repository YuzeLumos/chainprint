"""CPU fitting, composition and ranking without author-machine path assumptions.

The frozen numerical modules are distributed unchanged. ``paper`` fitting uses
OpWeaveEstimator; ``diagnostic-joint`` is a visibly restricted, inexpensive
diagnostic from its existing operation library, not the paper estimator result.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import time

import numpy as np
from PIL import Image
import torch


class MissingEditorAsset(RuntimeError):
    """An AI-containing history needs a caller-provided frozen editor backend."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_array(value):
    """Read RGB path, HWC/CHW uint8, or normalized floating-point pixels."""
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as image:
            value = np.asarray(image.convert("RGB")).copy()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError("An RGB CHW or HWC image is required")
    if array.shape[0] != 3 and array.shape[-1] == 3:
        array = array.transpose(2, 0, 1)
    if array.shape[0] != 3 or min(array.shape[1:]) < 16:
        raise ValueError("Expected RGB with height and width at least 16")
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255
    elif not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Use uint8 [0,255] or floating-point [0,1] pixels")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Pixels must be finite and in [0,1]")
    return np.ascontiguousarray(array, dtype=np.float32)


def save_image(path, pixels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image_array(pixels)
    Image.fromarray(np.rint(array.transpose(1, 2, 0) * 255).astype(np.uint8)).save(path)


def _cpu_runtime(threads=1):
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    from gain_block import GainReplay
    return GainReplay(torch.device("cpu"))


def _to_cpu(value):
    if isinstance(value, torch.nn.Module):
        return value.to("cpu").eval().requires_grad_(False)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


@dataclass
class ModelLibrary:
    operators: list
    metadata: dict

    def __len__(self):
        return len(self.operators)


def _asset_root(asset_root):
    if asset_root is None:
        asset_root = os.environ.get("CHAINPRINT_ASSET_ROOT")
    return Path(asset_root) if asset_root else Path(__file__).parent / "assets"


def load_library(cell=0, *, asset_root=None):
    """Load one of the six bundled banks after byte and numerical-state checks."""
    from final_local_extension import final_parameter_hash
    if type(cell) is not int or cell not in range(6):
        raise ValueError("cell must be an integer from 0 to 5")
    root = _asset_root(asset_root)
    manifest = json.loads((root / "bank_manifest.json").read_text(encoding="utf-8"))
    entry = next(row for row in manifest["banks"] if row["cell"] == cell)
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Model path escapes asset root")
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Model checksum mismatch: {path.name}")
    # These verified bundled files contain the original typed operation objects.
    operators = torch.load(path, map_location="cpu", weights_only=False)
    if final_parameter_hash(operators) != entry["model_hash"]:
        raise ValueError("Model numerical state differs from the frozen fit")
    return ModelLibrary(_to_cpu(operators), {
        "mode": "paper-frozen-bank", "cell": cell,
        "method_version": manifest["method_version"], "source": entry,
        "scope": manifest["scope"], "state_hash_verified": True,
    })


def save_library(library, path):
    """Save fitted operators and a SHA-bound JSON receipt; never overwrite."""
    from final_local_extension import final_parameter_hash
    path = Path(path)
    receipt = path.with_suffix(path.suffix + ".json")
    if path.exists() or receipt.exists():
        raise FileExistsError("Choose a new output file for fitted parameters")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(library.operators, path)
    receipt.write_text(json.dumps({
        "schema": "chainprint-fitted-library-v1", "sha256": sha256(path),
        "model_hash": final_parameter_hash(library.operators),
        "metadata": library.metadata,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_fitted(path):
    from final_local_extension import final_parameter_hash
    path = Path(path)
    receipt = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if receipt["schema"] != "chainprint-fitted-library-v1" or sha256(path) != receipt["sha256"]:
        raise ValueError("Fitted library checksum mismatch")
    operators = torch.load(path, map_location="cpu", weights_only=False)
    if final_parameter_hash(operators) != receipt["model_hash"]:
        raise ValueError("Fitted numerical-state checksum mismatch")
    return ModelLibrary(_to_cpu(operators), receipt["metadata"])


def _supports(before, after):
    before = np.stack([image_array(x) for x in before])
    after = np.stack([np.stack([image_array(x) for x in row]) for row in after])
    if len(before) != 8 or after.shape[0] != 8:
        raise ValueError("The frozen fit requires exactly eight independent paired contents")
    if after.ndim != 5 or after.shape[2:] != before.shape[1:] or not after.shape[1]:
        raise ValueError("Expected before[8,3,H,W], after[8,G,3,H,W]")
    if len({hashlib.sha256(x.tobytes()).hexdigest() for x in before}) != 8:
        raise ValueError("Eight distinct support contents are required; do not repeat a shot")
    return before, after


class _AnalyticalSupportCandidate:
    """The fitting-only slice of VariableHistoryCandidate.prepare_support.

    The frozen estimator reads only fits and diagnostics; its old neural
    fallback is never used by that path. No neural checkpoint is substituted.
    """
    def __init__(self):
        from history_recovery.chainprint_quantized_separable_likelihood import QuantizedSeparableLikelihood
        self.replay = QuantizedSeparableLikelihood(torch.device("cpu"))

    def prepare_support(self, *, support_before, support_after):
        fits, diagnostics = [], []
        for before, after in zip(support_before, support_after):
            fitted, diagnostic = self.replay.fit(before, after)
            fits.append(fitted)
            diagnostics.append(diagnostic)
        return SimpleNamespace(fits=fits, diagnostics=diagnostics)


class _SupportMemoReplay:
    def __init__(self, base, before, prior, outputs, available, refused):
        from ai_operator import CheckerRejectedSample
        self.base, self.device = base, base.device
        self.error = CheckerRejectedSample
        self.memo = {}
        for k, image in enumerate(before):
            key = hashlib.sha256(image.tobytes()).hexdigest()
            for j, spec in enumerate(prior.specs):
                self.memo[(key, spec.fingerprint)] = (
                    outputs[k, j] if available[k, j] else None, bool(refused[k, j]))

    def image(self, value):
        return self.base.image(value)

    def apply(self, image, operator):
        if operator[0] != "ai_inpaint":
            return self.base.apply(image, operator)
        pixels = np.ascontiguousarray(image.detach().cpu().numpy(), dtype=np.float32)
        key = (hashlib.sha256(pixels.tobytes()).hexdigest(), operator[1].fingerprint)
        if key not in self.memo:
            raise MissingEditorAsset("No exact support/prior replay; no editor is called implicitly")
        value, refused = self.memo[key]
        if refused:
            raise self.error("Caller-supplied cache records an unchanged checker refusal")
        if value is None:
            raise MissingEditorAsset("Missing replay is not a zero loss or a checker refusal")
        return self.image(value)


def load_prior(path):
    """Read a declared finite prior with explicit box masks or relative mask files."""
    from ai_operator import DeclaredPrior, InpaintSpec, box_mask
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for row in record["candidates"]:
        values = dict(row)
        if "box" in values:
            box = values.pop("box")
            mask = box_mask(*box, feather_pixels=values.pop("feather_pixels", 8))
        else:
            mask = np.load(path.parent / values.pop("mask_npy"), allow_pickle=False)
        specs.append(InpaintSpec(mask=mask, **values))
    return DeclaredPrior(tuple(specs), record["declaration_id"])


def fit(before, after, *, mode="paper", prior=None, prior_cache=None,
        threads=1, maximum_seconds=600):
    """Fit new anonymous operations from paired pixels, without query inputs.

    paper: exact frozen support hierarchy; requires 8 x 4 (or 8) 256px outputs,
    an explicit DeclaredPrior and exact existing support/prior replay cache.
    diagnostic-joint: the original cheap analytical joint-fitting diagnostic,
    restricted to linear/color/pointwise families; it is not the paper method.
    """
    from final_local_extension import final_parameter_hash
    if not np.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be positive and finite")
    before, after = _supports(before, after)
    base = _cpu_runtime(threads)
    started = time.monotonic()
    if mode == "diagnostic-joint":
        from joint_fit import fit_joint_operators
        families = ("spatial_linear", "color_affine", "pointwise_quantizer")
        operators, records = fit_joint_operators(base, before, after, families=families)
        metadata = {"mode": mode, "families": list(families), "records": records,
                    "scope": "Reduced original-library diagnostic; no full hierarchy, AI prior, or paper-performance claim"}
    elif mode == "paper":
        from opweave_frozen_20260916 import OpWeaveEstimator, VERSION
        if before.shape[-2:] != (256, 256) or after.shape[1] not in (4, 8):
            raise ValueError("Paper mode needs 256x256 pixels and four or eight anonymous slots")
        if prior is None or prior_cache is None:
            raise ValueError("Paper mode requires a declared prior and exact prior replay cache")
        if isinstance(prior, (str, os.PathLike)):
            prior = load_prior(prior)
        with np.load(prior_cache, allow_pickle=False) as cache:
            if str(cache["prior_sha256"].item()) != prior.manifest()["sha256"]:
                raise ValueError("Replay cache belongs to a different declared prior")
            cached_before = np.asarray(cache["before"], dtype=np.float32)
            outputs = np.asarray(cache["after"], dtype=np.float32)
            available = np.asarray(cache["available"], dtype=bool)
            refused = np.asarray(cache["refused"], dtype=bool)
        expected = (8, len(prior.specs))
        if not np.array_equal(before, cached_before) or outputs.shape != (*expected, 3, 256, 256):
            raise ValueError("Prior replay cache must bind exact before pixels and declared prior order")
        if available.shape != expected or refused.shape != expected or np.any(available & refused):
            raise ValueError("Invalid per-request availability/refusal mask")
        if np.any(~available & ~refused):
            raise MissingEditorAsset("Every request must have real cached pixels or a recorded checker refusal")
        if not np.isfinite(outputs[available]).all() or (outputs[available].size and
                (outputs[available].min() < 0 or outputs[available].max() > 1)):
            raise ValueError("Cached accepted pixels must be finite in [0,1]")
        replay = _SupportMemoReplay(base, before, prior, outputs, available, refused)
        estimator = OpWeaveEstimator(base, replay, _AnalyticalSupportCandidate(), prior)
        operators, records = [], []
        def budget():
            if time.monotonic() - started > maximum_seconds:
                raise TimeoutError("Support fitting exceeded its caller-defined time budget")
        for start in range(0, after.shape[1], 4):
            fitted, record = estimator.fit(before, np.ascontiguousarray(after[:, start:start+4]), check_budget=budget)
            operators.extend(fitted)
            records.append(record)
        metadata = {"mode": mode, "method_version": VERSION, "records": records,
                    "prior": prior.manifest(), "prior_cache_sha256": sha256(prior_cache),
                    "scope": "Frozen fitting hierarchy on caller-supplied new paired pixels; not a rerun of the historical benchmark"}
    else:
        raise ValueError("mode must be paper or diagnostic-joint")
    operators = _to_cpu(operators)
    metadata.update(support_count=8, operation_count=len(operators), seconds=time.monotonic()-started,
                    support_before_sha256=hashlib.sha256(before.tobytes()).hexdigest(),
                    support_after_sha256=hashlib.sha256(after.tobytes()).hexdigest(),
                    model_hash=final_parameter_hash(operators))
    return ModelLibrary(operators, metadata)


class _NoEditor:
    device = torch.device("cpu")
    def __call__(self, *_args, **_kwargs):
        raise MissingEditorAsset("AI operator requires a supplied frozen editor backend; none is downloaded or called implicitly")


def _replay(editor_backend=None, threads=1):
    from ai_operator import AIReplay
    from chainprint_shared_residual import ResidualReplay
    return ResidualReplay(AIReplay(editor_backend or _NoEditor(), _cpu_runtime(threads)))


def _words(candidates, count):
    result = []
    for word in candidates:
        word = list(word)
        if not word or any(type(g) is not int or g not in range(count) for g in word):
            raise ValueError("A history must be a nonempty list of valid anonymous integer slot IDs")
        result.append(word)
    if not result:
        raise ValueError("At least one candidate history is required")
    return result


def _operator_needs_editor(operator):
    return operator[0] == "ai_inpaint" or (
        operator[0] == "shared_residual" and _operator_needs_editor(operator[1][0]))


def _predict_candidates(before, candidates, library, editor_backend=None, threads=1):
    from ai_operator import CheckerRejectedSample
    replay = _replay(editor_backend, threads)
    # GainReplay.image is the original float64 convention. AIReplay.image has
    # a separate 256px editor constraint and must not coerce traditional inputs.
    initial = replay.base.base.image(image_array(before))
    words = _words(candidates, len(library))
    cache, reasons = {(): initial}, {}
    with torch.no_grad():
        for word in words:
            for depth, slot in enumerate(word, 1):
                prefix = tuple(word[:depth])
                if prefix in cache:
                    continue
                parent = prefix[:-1]
                if cache[parent] is None:
                    cache[prefix], reasons[prefix] = None, reasons[parent]
                    continue
                if editor_backend is None and _operator_needs_editor(library.operators[slot]):
                    cache[prefix] = None
                    reasons[prefix] = "MissingEditorAsset: no frozen editor backend was supplied"
                    continue
                try:
                    value = replay.apply(cache[parent], library.operators[slot])
                except (MissingEditorAsset, CheckerRejectedSample) as error:
                    cache[prefix], reasons[prefix] = None, f"{type(error).__name__}: {error}"
                    continue
                if value.shape != initial.shape or not torch.isfinite(value).all():
                    raise ValueError("An operation returned invalid pixels")
                cache[prefix] = value
    return [cache[tuple(w)] for w in words], [reasons.get(tuple(w)) for w in words]


def compose(before, history, library, *, editor_backend=None, threads=1):
    """Compose anonymous operations in order, including repeated occurrences."""
    predictions, reasons = _predict_candidates(before, [history], library, editor_backend, threads)
    if predictions[0] is None:
        raise MissingEditorAsset(reasons[0])
    return predictions[0].detach().cpu().numpy()


def rank(before, observed_after, candidates, library, *, editor_backend=None, threads=1):
    """Rank candidate histories by the original endpoint pixel MSE.

    Missing editor assets remain unavailable. Candidate index resolves display
    order only; numerical ties are explicitly reported with the same rank.
    """
    words = _words(candidates, len(library))
    predictions, reasons = _predict_candidates(before, words, library, editor_backend, threads)
    target = image_array(observed_after).astype(np.float64)
    if target.shape != image_array(before).shape:
        raise ValueError("Before and observed endpoint must have the same shape")
    rows = []
    for index, (word, value, reason) in enumerate(zip(words, predictions, reasons)):
        mse = float(np.square(value.cpu().numpy().astype(np.float64)-target).mean()) if value is not None else None
        rows.append({"candidate_index": index, "history": word, "available": mse is not None,
                     "mse": mse, "mse_8bit": mse*255**2 if mse is not None else None,
                     "unavailable_reason": reason})
    ordered = sorted(rows, key=lambda row: (row["mse"] is None, row["mse"] if row["mse"] is not None else 0, row["candidate_index"]))
    finite = [r["mse"] for r in rows if r["mse"] is not None]
    complete = len(finite) == len(rows)
    for row in ordered:
        conditional_rank = 1 + sum(v < row["mse"]-1e-15 for v in finite) if row["available"] else None
        row["rank_within_available"] = conditional_rank
        row["rank"] = conditional_rank if complete else None
    conditional_best = [r["candidate_index"] for r in ordered if r["rank_within_available"] == 1]
    return {"metric": "endpoint pixel MSE on [0,1] (lower is better)",
            "complete_candidates": complete, "abstained": not complete, "ranking": ordered,
            "best_candidate_indices": conditional_best if complete else [],
            "conditional_best_available_indices": conditional_best,
            "decision_scope": "complete candidate set" if complete else "incomplete candidate set; available-subset ordering is diagnostic only",
            "scope": library.metadata.get("scope", library.metadata.get("mode"))}
