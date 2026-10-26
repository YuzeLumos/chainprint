# Reproducibility boundary

## Two explicit fitting modes

| Mode | Implementation | Inputs | Verification scope |
|---|---|---|---|
| `paper` | Original `OpWeaveEstimator`, all frozen stages, same prior at every slot | Eight 256x256 RGB origins; four or eight anonymous outputs; declared prior; exact support/prior replay cache | Source bytes preserved; original parameter banks load and execute; no complete new-support refit rerun |
| `diagnostic-joint` | Original `joint_fit.fit_joint_operators` restricted to `spatial_linear`, `color_affine`, `pointwise_quantizer` | Eight distinct RGB origins and G anonymous outputs | Fresh generated RGB fit, save/reload, composition and order ranking verified |

The diagnostic is a small, usable path through genuine numerical primitives,
not a renamed full ChainPrint experiment. Its metadata states the restriction.
The original complete estimator remains available through `paper` mode and
direct import. Fitting uses paired supports only; query pixels, true histories,
renderer labels and operation names are not accepted by `fit`.

The portable support adapter extracts the fitting-only part of the original
`VariableHistoryCandidate.prepare_support`: it calls the exact
`QuantizedSeparableLikelihood.fit` for each content and exposes its `fits` and
`diagnostics`. Those are the only fields consumed by `MixedEstimator`. The old
neural fallback and query-scoring machinery are not used by this fitting path.

## Exact offline prior cache

`examples/paper_prior.json` reconstructs the original eight declared prompt,
mask and seed configurations. A different finite declaration may be supplied,
but must be fixed before observing any query or renderer truth. All slots use
the same declaration.

The caller supplies a numeric NPZ (loaded with `allow_pickle=False`):

| Key | Shape / meaning |
|---|---|
| `prior_sha256` | Scalar string equal to `load_prior(path).manifest()['sha256']` |
| `before` | float32 `[8,3,256,256]`; byte-exact normalized supports |
| `after` | float32 `[8,C,3,256,256]`; actual pre-existing outputs in declared prior order |
| `available` | bool `[8,C]`; true only for actual accepted pixel arrays |
| `refused` | bool `[8,C]`; true only for recorded, unchanged editor-checker refusals |

Every request must be either available or a recorded refusal. Unknown/missing
requests abort fitting; they are not treated as refusals, losses of zero, or
generated substitutes. Accepted outputs must be finite RGB in [0,1]. The before
pixels and prior digest are checked before fitting. Retain separate provenance
for the editor revision, its environment, image hashes and request status.
The supplied cache is a required input, not created by this release.

CLI `--maximum-seconds` passes the original budget callback. Original hierarchy
sections check it at stage boundaries; some internal sections are not
preemptible. This is a cooperative fitting bound, not a process watchdog.

## CPU composition and ranking

The package uses the original `GainReplay`, `AIReplay` and `ResidualReplay`.
Operations compose sequentially with prefix caching. The same fitted object is
reused for repeated occurrences. Pixel MSE is computed after composition.
AI-containing candidates with no supplied editor remain unavailable. The
`rank` result separates unavailability from a finite loss and reports tied
best candidates; index order only resolves display order. A global winner is
reported only if every candidate is available (`complete_candidates=true`).
Otherwise `abstained=true`, global best indices and ranks are empty/null, and
`conditional_best_available_indices` plus `rank_within_available` are explicitly
limited to the available subset.

All wrappers construct CPU replay objects explicitly. Some unchanged generic
classes have historical CUDA defaults, but these defaults are not used by the
portable API. No GPU model fitting or editor invocation was performed in this
verification. The optional editor CLI constructs a CPU float32 backend and uses
only an existing local cache for the original immutable revision.

## What the assets establish

Each shipped bank is checked against the original `aggregate.json` and strict
G=8 `RUN_CONTRACT.json`. The state hash is checked after CPU deserialization.
One real fitted operator was executed on generated RGB pixels and its output
compared with a direct call to the original model class. This validates loading
and numerical plumbing; it is not an independent accuracy experiment.

The generated diagnostic query is a distinct image from all eight supports.
Its true composition is used only to render an observed endpoint and check the
demo result. Neither the true operation parameters nor the true history is
passed to the fitter or ranker.

The 2D control is an independent finite-grid theoretical illustration. Its
five numerical CSVs match the submitted files byte for byte. This does not
substitute for the real-image experiments.

## Full benchmark assets still needed

Raw source images and derivative caches are omitted. Historical replay needs
the exact source-disjoint design, all support pools and availability masks,
query/candidate images, editor revision/weights/cache, and baseline adaptation
checkpoints. Some historical AI requests were refused, and the original
contracts explicitly preserve those missing results. They must not be filled
by rerunning or substituting requests when reconstructing the frozen study.

The original 180-source cohort, complete paper-mode refitting, baseline
training and AI editor generation were not rerun here. The candidate should be
described using those limits, even after all portability checks pass.
