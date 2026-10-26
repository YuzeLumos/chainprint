# Declared-prior AI operator adapter

Status: implemented and tested with an injected arithmetic fixture. No real AI
inference, model download, or remote execution was performed in this subtask.
Passing these contract tests supplies **no empirical evidence** for AI editing,
history recovery, robustness, visual quality, or unknown-editor generalization.

## Implemented boundary

`AIRenderer.apply(image, true_spec)` is the renderer-only entry point for a true
configuration. It returns a CHW torch RGB image. Renderer configuration tables,
operation names, query endpoints, and correct answers must remain outside the
fitter. Do not hand a renderer object to inference.

`fit_ai_operators(replay, before, after, declared_prior)` accepts `[support][RGB]`
inputs and `[support][anonymous_slot][RGB]` outputs. It applies every proposal in
the same finite `DeclaredPrior` to every input. Every anonymous slot receives the
same proposal bank. It selects one operation by equal-content mean squared error
in 8-bit units and returns `(chosen_operators, selection_records)`. Ties use the
predeclared proposal order. At least two support rows are required; there is no
hard-coded eight-row assumption.

The selected object is `('ai_inpaint', InpaintSpec(...))`. Its `apply` method
executes the generator on the **supplied current image**, including a new image
or intermediate image in a chain. It does not look up or paste an example's
previous output. Prediction calls are shared across output slots: fitting costs
`number_of_supports × number_of_prior_proposals` model calls.

This is a **matched editor and finite configuration prior**. Prompts, masks,
seeds, and parameter values may be included in a finite prior fixed in advance;
they are not revealed as the true configuration of an anonymous slot. A search
that includes the renderer's exact configuration is a matched-configuration
search. A successful result cannot establish unknown-prompt, unknown-mask,
unknown-seed, or unknown-editor recovery. If the actual function is outside the
prior, the method still chooses its nearest available proposal and may fail.

The runner must create and save the prior manifest before rendering or examining
an episode. `declaration_id` and a SHA-256 fingerprint make later changes
detectable; they do not prove that the prior was honestly declared in advance.
The Python API is a data-flow boundary, not an adversarial security sandbox.

For mixed traditional/AI dictionaries, offer these AI proposals to every slot
and compare them with existing support-fitted traditional proposals under the
same selection rule. Do not branch on true slot labels or inject AI only at a
known true AI position. `AIReplay(backend, base=existing_replay)` delegates
non-AI operators to the existing replay. The existing `ResidualReplay` can wrap
an `AIReplay` to execute already fitted residual operators, but this module does
not fit or adopt residual corrections and does not change the frozen calibrator.

## Fixed function and renderer convention

An `InpaintSpec` freezes prompt, negative prompt, mask, seed, denoising steps,
strength, and guidance scale. Its mask is stored as immutable float32 bytes;
editing the array used to construct it cannot change the operation. Backend
configuration fields are also fixed after construction.

Input is a finite `[0,1]` RGB 256×256 torch tensor or NumPy array in CHW or HWC
form. Output is a CHW torch tensor on the replay device, preserving the floating
input dtype. RGB bytes in `[0,255]` must be explicitly normalized beforehand.

Processing is fixed: CPU bilinear resize to 512, nearest-neighbor upsampling of
the mask's nonzero support to a binary diffusion mask, inpainting at 512, CPU
antialiased bilinear resize to 256, then fixed-alpha blending with the original
current image. At every exactly zero alpha pixel, the final output is an exact
copy of the input, including input values that are not 8-bit multiples. Soft
mask values control final blending; diffusion itself receives a binary mask.
`box_mask(x0, y0, x1, y1, feather_pixels=...)` uses 256-image pixel coordinates
and feathers inward, preserving an exactly zero exterior.

Each backend call creates a **fresh CPU `torch.Generator`** and sets it to the
operation seed. The seed never depends on support index, content, chain stage,
or call count. A lock serializes pipeline calls because scheduler state is
mutable. A process uses deterministic torch algorithms, disables cuDNN
benchmarking and TF32, and uses a deterministic CUDA workspace setting when a
real pipeline loads. These are explicit process-wide runtime effects.

The runner should set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before initializing CUDA
and preserve the complete environment/version record. The code does not claim
bit-identical results across GPU architectures, software versions, devices, or
precision settings. Repeated-image behavior should be checked on the actual
deployed runtime before collecting data.

## Model and software pins

- Model: `stable-diffusion-v1-5/stable-diffusion-inpainting`.
- Default revision: `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`, verified from the
  [official repository commit](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/commit/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb).
- Loader: `StableDiffusionInpaintPipeline`, exactly `diffusers==0.36.0`.
- Weights: `variant='fp16'`, `use_safetensors=True`; the pinned model contains
  those safetensors. Runtime arithmetic can be float16 or float32; CPU requires
  float32. The scheduler remains the one in the pinned model configuration.
- `local_files_only=True` is the default. Construction, manifest inspection,
  imports, and fixture tests do not fetch weights. Explicitly setting it false
  permits lazy loading to download missing files when the caller has authorized
  that separate deployment step.
- `backend.manifest()` records model/revision, precision, device, loading policy,
  dependency versions, and current deterministic-algorithm status. Record it
  again after loading, together with the deployment environment lock and GPU.
- A custom revision must be a complete 40-character lowercase commit hash;
  mutable revisions such as `main` are rejected. No fallback editor is loaded.
- The model checker stays enabled. A flagged output raises; it is not silently
  counted as generated evidence. Report such failures in the evaluation ledger.

The official [versioned inpainting API](https://huggingface.co/docs/diffusers/v0.36.0/en/api/pipelines/stable_diffusion/inpaint)
defines the RGB/mask interface and recommends inpainting-specific checkpoints.
The official [reproducibility guide](https://huggingface.co/docs/diffusers/v0.36.0/en/using-diffusers/reusing_seeds)
explains CPU generators and why reusing a consumed generator changes later
outputs; it also states the limits of cross-platform reproducibility.

## Executable finite-prior pattern

```python
from itertools import product
from ai_operator import (
    InpaintSpec, DeclaredPrior, DiffusersInpaintBackend, AIReplay,
    box_mask, fit_ai_operators,
)

# Save prior.manifest() before rendering or reading the episode outputs.
# These values are a declared search space, never read from renderer truth.
prompts = ("a patch of red flowers", "a small blue ceramic vase")
masks = (box_mask(48, 64, 144, 176), box_mask(96, 80, 208, 208, 8))
seeds = (112, 317)
prior = DeclaredPrior(
    tuple(InpaintSpec(p, m, seed=s, num_inference_steps=25)
          for p, m, s in product(prompts, masks, seeds)),
    declaration_id="editor-prior-v1-fixed-before-episode",
)
backend = DiffusersInpaintBackend(local_files_only=True)
replay = AIReplay(backend, base=existing_replay)
chosen, records = fit_ai_operators(replay, before, after, prior)
next_image = replay.apply(current_image, chosen[anonymous_slot_index])
```

`before`, `after`, `existing_replay`, and `current_image` are supplied by the
experiment runner. The snippet is a construction pattern and creates no data.
The eight proposals above cost 64 inpainting calls for eight supports. A large
prompt/mask/seed cross product quickly becomes expensive. Any search budget,
candidate ordering, mask bank, and sampling rule should be fixed before running
the target evaluation and reported with results.

## Leave-one-content-out and leakage

The built-in finite prior has empty support provenance because its proposals
are fixed before the episode. Each leave-one-content-out selection removes the
held loss column. The reported held loss then evaluates that selected operator
on the excluded row. These folds are support diagnostics; they are not separate
query tests and must not be reported as query generalization evidence.

`AICandidate(operator, sources=frozenset(...))` and `select_candidates(...)`
also support arbitrary support-derived proposals. Record **all** rows used to
fit, generate, tune, or select a proposal, including indirect dependencies.
Holding out row `j` removes every candidate whose source set contains `j` as
well as column `j`. A fit from all support rows is therefore ineligible in every
fold; callers must refit without the held row to evaluate that fitting method.
When nothing independent remains, selection raises rather than leak the row.
This adapter does not presently create support-derived masks, tune prompts,
fit a residual network, or learn new global parameters.

## Local verification

Run `python -B -m unittest test_ai_operator -v` in this directory. Seven tests
passed on local torch 2.9.1 and NumPy 2.1.3 with no model download:

- Lazy loading and immutable model revision; injected runs marked test-only.
- Fresh generator, repeated-input equality, dependence on current input, and
  exact outside-mask backfill through the 256/512 resize path.
- Immutable operation/mask parameters and nontrivial soft edges.
- Equal-content common selection and equivariance to anonymous-slot permutation.
- Fitter rejects true-configuration, name, and query metadata arguments.
- Leave-one exclusion handles multi-row candidate provenance and loss columns.
- Existing replay delegation and rejection of invalid image shape/range.

The fixture uses a deterministic arithmetic color blend and a seeded random
color. It is deliberately **not a diffusion model or simulated AI experiment**.
Real-model success, visual inspection, transfer to independent images, chain
composition, running time, and experimental effect sizes remain to be measured
by the separate authorized deployment/evaluation workflow.
