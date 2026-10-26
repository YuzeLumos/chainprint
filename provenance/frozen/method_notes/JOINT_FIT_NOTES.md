# Joint support fit and matched residual control

This new module does not modify the frozen method. The runner must import the original `SupportCodecReplay` and the frozen deployed `chainprint_shared_residual` helper. Core source modules read during implementation were checked against the 2026-09-10 closeout `frozen_runtime`; the seven checked fitting/prediction modules matched SHA-256 exactly. The deployed pooled and residual helper source files were read separately.

## Runner API

```python
from joint_fit import fit_joint_operators, calibrate_joint_operators
from chainprint_shared_residual import ResidualReplay

joint, joint_records = fit_joint_operators(replay, before, after)
joint_corrected, joint_residual_records = calibrate_joint_operators(
    replay, before, after, joint, joint_records)
wrapped = ResidualReplay(replay)
# replay.apply(image, joint[g])
# wrapped.apply(image, joint_corrected[g])
```

`before` has eight CHW [0, 1] RGB images; `after[k][g]` is anonymous slot `g` output for support `k`. Images must share shape; the full block library requires dimensions divisible by 16. Fit APIs accept no query images, query outputs, operation identities, or true parameters. The reduced `families` keyword exists only for bounded tests and is marked `complete_library=False`; the four-method comparison uses the default complete library. Diagnostics are JSON serializable; operator objects contain tensors and, where selected, a torch module.

Keep the four methods: frozen pooled base, frozen pooled plus residual, joint base, and joint plus the identical-capacity residual. Both base methods remain available regardless of residual adoption. This provides the missing comparison between selecting a separately fitted operator and estimating shared parameters from all supports. It does not equalize training computation between those approaches.

## Library and fitting objectives

| Family | Joint fit and frozen conventions retained |
|---|---|
| Spatial linear | 14 orbit features; concatenate eligible pixel equations through summed sufficient statistics, divide Gram/RHS by total eligible count, ridge 0.001; exclude target bytes 0 and 255. |
| Spatial gain | The same joint spatial kernel, then one nonnegative gain from global dot products; denominator guard 1e-12. |
| Spatial dead zone | The same joint kernel, then globally sort all eligible absolute deltas, use the frozen cumulative-statistic threshold/gain rule and last-best tie convention. |
| Cross-channel affine | 45 features, shared feature and response means, pooled centered Gram/RHS, ridge 0.001; include pixels only if all three target channels are interior. |
| Pointwise quantizer | Shared levels 2 through 256; sum per-image byte-bin counts/target sums; same LUT, rounding, and first-index tie. |
| Full block quantization | Shared 192 frequency steps on the original 1 through 255 grid; aggregate the original coefficient-space squared-error objective across all image blocks. Same YCbCr transform and inverse. |
| Half-chroma block quantization | Same joint grid for each of three planes (64 steps each), original downsample and least-squares latent chroma inverse, original bilinear restoration. |
| Full/half block with RGB offset | Joint steps as above, then one shared RGB mean residual rounded to integer bytes; same three-parameter capacity. |
| Resampling | Both original filters, exhaustive original latent sizes, unique-size order, original at-most-1024 linspace pixel sample restricted to `(row+column)%4 != 0`. Minimize mean sample loss across all supports. |
| JPEG codec | Shared quality 1 through 100 and subsampling 0, 1, 2; original order, original at-most-1024 subsample of training pixels, same Pillow encoder settings; equal-support objective. |
| Quantized separable response | One shared five-logit kernel/gain/threshold model; same float32 byte inputs, checkerboard mask, initial gain objective, Adam learning rate 0.03 and 128 updates. Sum all per-image gradients before each shared update. Same STE, clamps and integer prediction. |

The general linear/color/quantizer/block families use their original **terminal full-refit** target mask (all pixels, subject to original saturation exclusions). JPEG, resampling and quantized-separable use their original prescribed training/sample mask because those methods do not perform a full-pixel refit in frozen code. All predictions, clipping, color conversions and output quantization delegate to the frozen replay's `apply`. No average of independently estimated parameters is substituted for a shared fit.

Every fitted family then predicts every training image. Select the family with lowest arithmetic mean of full-image 8-bit MSE, with fixed family order for ties. Diagnostics store the complete family-by-content loss matrix and mean losses. The joint baseline evaluates all eligible library families; it does not reproduce the frozen per-support hierarchy that conditionally suppresses later families. This changed family selection is explicit and requested: pooled selection already compares whole-image losses across fitted candidates. The joint family search is therefore broader at the *selection* stage, even though its model parameterizations and family fitting rules come from the same library.

## Exactness and fairness limitations

- Spatial/color fits solve their pooled penalized surrogate objectives. They are not global optima of final clipped, byte-quantized MSE. The original estimator is also a surrogate fit.
- Gain and dead-zone fitting are sequential given the ridge-fitted kernel; they are not a simultaneous nonlinear kernel/gain/threshold optimizer. This preserves the original family estimator.
- DCT quantization searches are exact for each frequency in the original coefficient-space objective, not joint RGB output MSE after inverse transform/clipping. RGB offsets are fitted after steps, not jointly with them.
- Quantized-separable fitting is a **nonconvex approximate joint optimizer**, using the original fixed 128-step budget. No global-optimality claim is justified. Sharing gradients is a direct objective extension, not an exact analytic sufficient-statistic reduction. With K=8, each update processes eight images; equal update counts do not imply equal total FLOPs to single-image fitting.
- The JPEG candidate need not have exact support-byte agreement to enter the joint family comparison, matching the pooled method's additional codec-proxy candidates. This is not an assertion of an exact recovered codec.
- The original per-support internal validation gates are not re-created as joint hierarchy gates. Final family selection is full-support MSE. Treat this as the declared same-library joint baseline, not an exact reimplementation of frozen single-support selection.
- All samples have equal shape. Eligible pixel weighting for regression follows the original normal equations; full-image family selection uses equal-content weighting. Saturation counts can differ, so those are distinct weighting conventions by design.

No family is silently omitted in the default run. `complete_library=True` states coverage, not exact optimization. Report the optimizer caveats above with results.

## Matched residual gate and leakage boundary

Each of eight content folds physically slices the raw support lists to the other seven images before calling the entire joint fitter, including family selection and all discrete searches. The residual is fitted only on predictions/targets for those same seven images. Only after both fits finish does the code apply that fold's operator and correction to the held source image and evaluate its target. Training-content indices, seven-content data signatures, selected families and training family MSE are recorded.

The residual implementation is imported directly from the frozen helper: 31 features, a 31-by-3 coefficient matrix, Gram pseudoinverse with `rtol=1e-10`, no added hyperparameters, and identical clipping and rounding. Adopt iff all eight corrected losses are no greater than their matching base losses and the mean strictly decreases. If adopted, refit residual coefficients on all eight supports around the already stored all-eight joint base. Retain that base separately. Held folds select adoption and are **not independent test estimates**. Query outcomes must never control adoption.

## Bounded verification and computational budget

`test_joint_fit.py` contains three focused tests: pooled sufficient statistics versus direct concatenated objectives (centered/un-centered ridge, quantizer, and DCT grid); an actual eight-fold fitting audit which poisons held source and target and checks that its excluded fold's base parameters and residual coefficients stay identical; and bitwise equivalence of cached grid statistics and uncached Pillow evaluation, including all eight leave-one-content-out folds. Tests use tiny synthetic inputs and a restricted library for the leakage audit; they do not claim full benchmark validation. Run in the original environment with frozen source and deployed helper on `PYTHONPATH`:

```text
python test_joint_fit.py
```

Full fitting performs one eight-support fit. Corrected joint fitting adds eight seven-support fits, plus one residual regression per slot/fold and an all-eight residual refit only when accepted. Grid predictions are shared across slots within each fit; the support-only grid statistics described below can also be reused between fits. Nothing reuses a model fitted on the held content. No gradient/sweep budget is increased after inspecting results. Wall time should be reported separately for base fitting and the optional nested correction.

At authoring time, the local bundled Python lacked torch; numerical tests are supplied for execution by the orchestrating runner in the original torch environment. A local syntax check can run without importing torch. Record actual execution results separately rather than interpreting test availability as a pass.

## Bounded support-statistic reuse (2026-09-15)

JPEG and resampling now retain one float64 grid-by-slot MSE matrix per source image and family. A 32-entry LRU holds only these small matrices, rather than rendered images, fitted operators, fold aggregates, or query values. Its key includes every original source and target pixel across all anonymous slots, raw dtype/shape, ordered parameter grid, exact sampled pixel indices, filter configuration, and renderer functions. Switching replay instances clears the cache. Changing even an unused target pixel forces that content's statistics to be recomputed.

Every fit still visits grid parameters in the same order and then visits its included supports in the same order. It divides each single-content MSE by that fit's support count before adding it to the float64 accumulator, exactly as before. It does not subtract held losses from an eight-content sum or average previously fitted parameters. All eight leave-one-out fits only consume the seven included contents' entries, so a cached held target cannot influence their fitted objectives. The full-eight cold fit can populate the sixteen entries reused by all folds; in this case the number of Pillow grid renders falls from 64 content-fits per family (8 + 8×7) to 8, while all other fitting and validation work remains unchanged. This is a render-count reduction, not a measured end-to-end speedup.

Local execution with `D:/Anaconda/python.exe` (torch 2.9.1, CPU, one torch thread) passed all three focused checks in 1.962 seconds. The added check compares every cached parameter/slot loss against direct Pillow arithmetic, verifies unchanged full-fit outputs and all eight seven-content fold outputs, verifies zero additional renders on warm fits, and poisons a held target to confirm cache invalidation and exclusion. No benchmark, remote run, or accuracy claim is implied by these checks.
