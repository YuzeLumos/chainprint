"""Reusable inpainting selected from a declared editor/configuration prior.

Renderer truth is never an input to fitting. The same finite, predeclared prior
is tested against every anonymous slot using support pixels. This is matched
prior selection, not unknown-editor discovery. See AI_OPERATOR_NOTES.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
import re
import threading
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
DEFAULT_MODEL_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
DIFFUSERS_VERSION = "0.36.0"
OPERATOR_KIND = "ai_inpaint"
IMAGE_SIZE, INTERNAL_SIZE = 256, 512


class CheckerRejectedSample(RuntimeError):
    """The pinned model checker rejected an image; no generated image is exposed."""
    pass


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def box_mask(x0: int, y0: int, x1: int, y1: int,
             feather_pixels: float = 0.0) -> np.ndarray:
    """256px rectangle; optional feathering is inward, so outside stays zero."""
    if any(type(v) is not int for v in (x0, y0, x1, y1)):
        raise ValueError("Box coordinates must be integer pixel coordinates")
    if not (0 <= x0 < x1 <= IMAGE_SIZE and 0 <= y0 < y1 <= IMAGE_SIZE):
        raise ValueError("Box must be nonempty and inside the 256px image")
    if not math.isfinite(feather_pixels) or feather_pixels < 0:
        raise ValueError("feather_pixels must be finite and nonnegative")
    yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    distance = np.minimum.reduce((xx + 0.5 - x0, x1 - xx - 0.5,
                                  yy + 0.5 - y0, y1 - yy - 0.5))
    return ((distance > 0).astype(np.float32) if feather_pixels == 0 else
            np.clip(distance / feather_pixels, 0, 1).astype(np.float32))


@dataclass(frozen=True, init=False)
class InpaintSpec:
    """Immutable function configuration; seed never varies by image or call."""
    prompt: str
    seed: int
    negative_prompt: str
    strength: float
    guidance_scale: float
    num_inference_steps: int
    _mask_bytes: bytes = field(repr=False)

    def __init__(self, prompt: str, mask, seed: int, negative_prompt: str = "",
                 strength: float = 1.0, guidance_scale: float = 7.5,
                 num_inference_steps: int = 25):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A fixed nonempty prompt is required")
        if not isinstance(negative_prompt, str):
            raise ValueError("negative_prompt must be a string")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        if not math.isfinite(strength) or not 0 < strength <= 1:
            raise ValueError("strength must be finite in (0, 1]")
        if int(num_inference_steps * strength) < 1:
            raise ValueError("strength and steps must produce at least one denoising step")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        alpha = np.asarray(mask, dtype="<f4")
        if alpha.shape != (IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(alpha).all():
            raise ValueError("mask must be a finite 256x256 array")
        if alpha.min() < 0 or alpha.max() > 1 or not np.any(alpha > 0):
            raise ValueError("mask must be in [0,1] with a nonzero region")
        values = dict(prompt=prompt, seed=seed, negative_prompt=negative_prompt,
                      strength=float(strength), guidance_scale=float(guidance_scale),
                      num_inference_steps=num_inference_steps,
                      _mask_bytes=alpha.tobytes(order="C"))
        for key, value in values.items():
            object.__setattr__(self, key, value)

    @property
    def mask(self) -> np.ndarray:
        # Bytes-backed views cannot be made writable; source-array edits are inert.
        return np.frombuffer(self._mask_bytes, dtype="<f4").reshape(IMAGE_SIZE, IMAGE_SIZE)

    def manifest(self) -> dict:
        return {"prompt": self.prompt, "negative_prompt": self.negative_prompt,
                "seed": self.seed, "strength": self.strength,
                "guidance_scale": self.guidance_scale,
                "num_inference_steps": self.num_inference_steps,
                "mask_sha256": hashlib.sha256(self._mask_bytes).hexdigest(),
                "mask_shape": [IMAGE_SIZE, IMAGE_SIZE],
                "mask_rule": "binary nonzero support for diffusion; fixed soft alpha for final blend"}

    @property
    def fingerprint(self) -> str:
        return _digest(self.manifest())


@dataclass(frozen=True)
class DeclaredPrior:
    """One predeclared finite bank shared by every anonymous slot.

    Identifiers/hashes are audit fields, not proof of preregistration. The runner
    must save the prior before rendering or observing an episode. Do not create
    this bank by inspecting that episode's renderer truth configurations.
    """
    specs: tuple[InpaintSpec, ...]
    declaration_id: str

    def __post_init__(self):
        object.__setattr__(self, "specs", tuple(self.specs))
        if not self.specs or not all(isinstance(s, InpaintSpec) for s in self.specs):
            raise ValueError("Declare at least one InpaintSpec")
        if not isinstance(self.declaration_id, str) or not self.declaration_id.strip():
            raise ValueError("A prior declaration identifier is required")
        if len({s.fingerprint for s in self.specs}) != len(self.specs):
            raise ValueError("Duplicate prior proposals are not allowed")

    def manifest(self) -> dict:
        payload = {"declaration_id": self.declaration_id,
                   "scope": "matched declared editor and finite configuration prior",
                   "same_prior_for_all_anonymous_slots": True,
                   "candidates": [spec.manifest() for spec in self.specs]}
        return {**payload, "sha256": _digest(payload)}


def _as_image(value, device=None) -> torch.Tensor:
    result = value.detach().clone() if isinstance(value, torch.Tensor) else torch.tensor(np.asarray(value))
    if tuple(result.shape) == (IMAGE_SIZE, IMAGE_SIZE, 3):
        result = result.permute(2, 0, 1)
    if tuple(result.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("A CHW or HWC RGB 256x256 image is required")
    if not result.is_floating_point():
        result = result.to(torch.float32)
    if not bool(torch.isfinite(result).all()) or result.min() < 0 or result.max() > 1:
        raise ValueError("Image pixels must be finite and in [0,1]")
    return result.to(device=device).contiguous() if device is not None else result.contiguous()


class DiffusersInpaintBackend:
    """Lazy pinned SD1.5 backend, offline by default; imports never fetch weights.

    pipeline_factory is a contract-test injection seam, explicitly marked in the
    manifest. The model safety checker stays enabled; flagged samples raise.
    """
    def __setattr__(self, name, value):
        immutable = {"revision", "device", "dtype", "local_files_only", "cache_dir", "_pipeline_factory"}
        if name in immutable and hasattr(self, name):
            raise AttributeError(f"Backend configuration {name!r} is fixed; create a new backend")
        object.__setattr__(self, name, value)

    def __init__(self, revision: str = DEFAULT_MODEL_REVISION, device="cuda:0",
                 dtype: str = "float16", local_files_only: bool = True,
                 cache_dir: str | None = None, pipeline_factory: Callable | None = None):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be an immutable 40-character lowercase commit hash")
        if dtype not in ("float16", "float32"):
            raise ValueError("dtype must be float16 or float32")
        self.revision, self.device, self.dtype = revision, torch.device(device), dtype
        if self.device.type == "cpu" and dtype != "float32":
            raise ValueError("CPU inference requires float32")
        self.local_files_only, self.cache_dir = bool(local_files_only), cache_dir
        self._pipeline_factory, self._pipeline = pipeline_factory, None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def manifest(self) -> dict:
        versions = {}
        for package in ("diffusers", "transformers", "accelerate", "safetensors", "torch", "numpy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return {"backend_kind": ("stable_diffusion_inpainting" if self._pipeline_factory is None
                                  else "injected_pipeline_test_only"),
                "model_id": MODEL_ID, "revision": self.revision,
                "required_diffusers_version": DIFFUSERS_VERSION, "weight_variant": "fp16",
                "use_safetensors": True, "device": str(self.device), "dtype": self.dtype,
                "local_files_only": self.local_files_only, "loaded": self.loaded,
                "internal_size": INTERNAL_SIZE, "external_size": IMAGE_SIZE,
                "rng": "fresh CPU torch.Generator with operation seed on every call",
                "deterministic_algorithms_required_for_real_backend": True,
                "torch_deterministic_algorithms_current": torch.are_deterministic_algorithms_enabled(),
                "versions": versions,
                "reproducibility_scope": "fixed runtime/device; cross-device bit identity is not guaranteed"}

    def _load(self):
        if self._pipeline is not None:
            return
        if self._pipeline_factory is None:
            installed = importlib.metadata.version("diffusers")
            if installed != DIFFUSERS_VERSION:
                raise RuntimeError(f"Require diffusers=={DIFFUSERS_VERSION}; found {installed}")
            from diffusers import StableDiffusionInpaintPipeline
            factory = StableDiffusionInpaintPipeline.from_pretrained
            # Runner should set the environment variable before initializing CUDA.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
                raise RuntimeError("CUBLAS_WORKSPACE_CONFIG conflicts with deterministic execution")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            factory = self._pipeline_factory
        pipe = factory(MODEL_ID, revision=self.revision, torch_dtype=getattr(torch, self.dtype),
                       variant="fp16", use_safetensors=True,
                       local_files_only=self.local_files_only, cache_dir=self.cache_dir)
        self._pipeline = pipe.to(self.device)
        self._pipeline.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def __call__(self, image_512: np.ndarray, mask_512: np.ndarray,
                 spec: InpaintSpec) -> np.ndarray:
        with self._lock:
            self._load()
            # Reusing one Generator would advance state and change the operation.
            generator = torch.Generator(device="cpu").manual_seed(spec.seed)
            result = self._pipeline(prompt=spec.prompt, negative_prompt=spec.negative_prompt,
                                    image=image_512.copy(), mask_image=mask_512.copy(),
                                    height=INTERNAL_SIZE, width=INTERNAL_SIZE,
                                    strength=spec.strength, guidance_scale=spec.guidance_scale,
                                    num_inference_steps=spec.num_inference_steps,
                                    num_images_per_prompt=1, eta=0.0, generator=generator,
                                    output_type="np", return_dict=True)
            flags = getattr(result, "nsfw_content_detected", None)
            if flags is not None and any(bool(flag) for flag in flags):
                raise CheckerRejectedSample("Inpainting sample flagged by model checker; no evidence emitted")
            output = np.asarray(result.images)
            if output.shape != (1, INTERNAL_SIZE, INTERNAL_SIZE, 3):
                raise ValueError("Backend must return one RGB 512x512 image")
            return output[0].copy()


@torch.no_grad()
def apply_inpaint(backend, before, spec: InpaintSpec, device=None) -> torch.Tensor:
    """Apply to the supplied current image, never substitute a stored endpoint."""
    if not isinstance(spec, InpaintSpec):
        raise TypeError("Expected an InpaintSpec")
    image = _as_image(before, device=device)
    # Fixed CPU resize path independent of the model device.
    source = image.detach().cpu().to(torch.float32)
    upsampled = F.interpolate(source[None], size=(INTERNAL_SIZE, INTERNAL_SIZE),
                              mode="bilinear", align_corners=False)[0]
    alpha = torch.tensor(spec.mask.copy(), dtype=torch.float32)
    hard = F.interpolate((alpha > 0).to(torch.float32)[None, None],
                          size=(INTERNAL_SIZE, INTERNAL_SIZE), mode="nearest")[0, 0]
    generated = np.asarray(backend(upsampled.permute(1, 2, 0).numpy(), hard.numpy(), spec))
    if generated.shape != (INTERNAL_SIZE, INTERNAL_SIZE, 3):
        raise ValueError("Backend output must have shape 512x512x3")
    if not np.isfinite(generated).all() or generated.min() < 0 or generated.max() > 1:
        raise ValueError("Backend output must be finite RGB in [0,1]")
    prediction = torch.tensor(generated, dtype=torch.float32).permute(2, 0, 1)
    prediction = F.interpolate(prediction[None], size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode="bilinear", align_corners=False, antialias=True)[0]
    prediction = prediction.to(device=image.device, dtype=image.dtype)
    alpha = alpha.to(device=image.device, dtype=image.dtype)[None]
    blended = image + alpha * (prediction - image)
    return torch.where(alpha == 0, image, blended.clamp(0, 1))


class AIRenderer:
    """Privileged rendering facade; do not pass this object to inference."""
    def __init__(self, backend):
        self.backend = backend

    def apply(self, image, true_spec: InpaintSpec) -> torch.Tensor:
        return apply_inpaint(self.backend, image, true_spec)


class AIReplay:
    """image()/apply() wrapper compatible with existing replay consumers."""
    def __init__(self, backend, base=None):
        self.backend, self.base = backend, base
        self.device = torch.device(getattr(base, "device", getattr(backend, "device", "cpu")))

    def image(self, value) -> torch.Tensor:
        return _as_image(value, self.device)

    def apply(self, before, operator) -> torch.Tensor:
        kind, value = operator
        if kind == OPERATOR_KIND:
            return apply_inpaint(self.backend, before, value, self.device)
        if self.base is None:
            raise ValueError(f"No base replay for operator kind {kind!r}")
        return self.base.apply(before, operator)


@dataclass(frozen=True)
class AICandidate:
    """sources lists EVERY support row used to create/tune this proposal.

    Empty sources means it was fixed independently, e.g. a declared prior. A fit
    from multiple rows must name all of them. Holdout excludes that whole fit.
    """
    operator: tuple
    sources: frozenset[int] = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "sources", frozenset(self.sources))
        if any(type(source) is not int or source < 0 for source in self.sources):
            raise ValueError("Sources must be nonnegative support-row indices")


def select_candidates(losses, candidates: Sequence[AICandidate], held_out: int | None = None,
                      available=None) -> int | None:
    """Equal-content mean, declared-order ties, general source/column exclusion."""
    matrix = np.asarray(losses, dtype=np.float64)
    mask = np.ones_like(matrix,dtype=bool) if available is None else np.asarray(available,dtype=bool)
    if (matrix.ndim != 2 or matrix.shape[0] != len(candidates) or not matrix.size
            or mask.shape != matrix.shape):
        raise ValueError("Expected finite nonnegative candidate-by-content losses")
    count = matrix.shape[1]
    if any(max(c.sources, default=-1) >= count for c in candidates):
        raise ValueError("Candidate provenance refers to a nonexistent support row")
    if held_out is not None and (type(held_out) is not int or not 0 <= held_out < count):
        raise ValueError("held_out must be a valid support-row index")
    columns = [j for j in range(count) if j != held_out]
    if not columns:
        raise ValueError("No independent content remains after exclusion")
    included_values = matrix[:,columns][mask[:,columns]]
    if not np.isfinite(included_values).all() or np.any(included_values < 0):
        raise ValueError("Included available losses must be finite and nonnegative")
    rows = [i for i, candidate in enumerate(candidates)
            if held_out not in candidate.sources and bool(mask[i,columns].all())]
    if not rows:
        return None
    return rows[int(np.argmin(matrix[np.ix_(rows, columns)].mean(axis=1)))]


@torch.no_grad()
def fit_ai_operators(replay: AIReplay, before: Iterable, after: Iterable,
                     declared_prior: DeclaredPrior):
    """Select one shared AI proposal per anonymous slot using support pixels.

    before=[support][RGB]; after=[support][anonymous_slot][RGB]. No operation
    names, true parameters, query endpoints, or expected answers are accepted.
    Inference cost is N*C calls for N supports, C proposals, shared across slots.
    """
    if not isinstance(declared_prior, DeclaredPrior):
        raise TypeError("An explicit DeclaredPrior is required")
    images = [replay.image(value) for value in before]
    observed = [[replay.image(value) for value in row] for row in after]
    if len(images) < 2 or len(observed) != len(images):
        raise ValueError("At least two aligned support rows are required for leave-one-content-out")
    slots = len(observed[0])
    if slots == 0 or any(len(row) != slots for row in observed):
        raise ValueError("Output rows must share a nonzero number of anonymous slots")
    candidates = [AICandidate((OPERATOR_KIND, spec)) for spec in declared_prior.specs]
    losses = np.zeros((slots, len(candidates), len(images)), dtype=np.float64)
    available = np.ones((len(candidates),len(images)),dtype=bool)
    for i, candidate in enumerate(candidates):
        for j, image in enumerate(images):
            try:
                predicted = replay.apply(image, candidate.operator).to(torch.float64)
            except CheckerRejectedSample:
                available[i,j] = False
                continue
            for g in range(slots):
                losses[g, i, j] = float(((predicted - observed[j][g].to(torch.float64)) * 255).square().mean())
    chosen, records = [], []
    prior_manifest = declared_prior.manifest()
    for matrix in losses:
        selected = select_candidates(matrix, candidates, available=available)
        held = [select_candidates(matrix, candidates, j, available=available) for j in range(len(images))]
        chosen.append(candidates[selected].operator if selected is not None else None)
        records.append({"prior_sha256": prior_manifest["sha256"],
                        "scope": "matched declared editor and finite configuration prior",
                        "candidates": [{"index": i, "candidate_sha256": s.fingerprint,
                                        "sources": []} for i, s in enumerate(declared_prior.specs)],
                        "support_mse_8bit": [[float(matrix[i,j]) if available[i,j] else None
                                               for j in range(len(images))] for i in range(len(candidates))],
                        "support_available": available.tolist(), "selected": selected,
                        "selection_rule": "lowest equal-content mean; declared-order ties",
                        "leave_one_content_out_selected": held,
                        "leave_one_content_out_mse_8bit": [float(matrix[i,j]) if i is not None and available[i,j] else None
                                                           for j,i in enumerate(held)],
                        "leave_one_content_out_scope": "Exclude held loss column, held availability, and all proposals whose source sets contain it. Availability is checked only on included columns. Declared proposals have no support sources. Support diagnostic, not independent query evidence.",
                        "new_global_training_updates": 0})
    return chosen, records
