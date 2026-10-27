# ChainPrint: anonymous image processing-chain forensics by example

## Public release candidate

`v0.1.0-rc1` publishes the tested core code and companion model/state assets.
The locally executed checks and coverage limits are recorded in `VERIFY.json`
and below; complete raw-data-to-paper-table reproduction is not claimed.
License status is recorded in `NOTICE`.


This candidate packages the **current paired-operation fitting method** from
*Anonymous Image Processing-Chain Forensics by Example* (ICLR 2027 submission
7543). It contains the frozen numerical implementation and its dependency
closure, six original K=8 anonymous operation banks, and new portable CPU
interfaces for fitting, sequential composition, and endpoint ranking.

The old A10/v2 neural training branch is not the entry point to this package.
The original numerical source files are unchanged. Their hashes and origin
paths are recorded in `provenance/SOURCE_MANIFEST.json`.

## Install and try actual RGB images

Python 3.11 or newer is required. The verification environment used Python
3.13.5, NumPy 2.1.3, Pillow 11.1.0, and PyTorch 2.9.1 on CPU.

```sh
python -m venv .venv
# Activate .venv using your platform's usual command.
python -m pip install .
python -m chainprint demo --output demo-output
```

The demo generates eight distinct RGB support images and two anonymous outputs
for each. It fits real analytical primitives from the original library, then
ranks four ordered histories on a new RGB query. It also executes an original
submitted operator from bank 0 on a separate 256x256 image. Inspect the PNGs,
`pairs.json`, fitted parameters, and `result.json` in the output directory.

The fast demo uses the explicitly restricted `diagnostic-joint` fitter. It
demonstrates image inputs, parameter fitting and order-sensitive composition;
it is not a claim to reproduce the paper's full estimator or reported accuracy.

## Fit, compose, and rank your own paired images

Create a JSON file containing exactly eight distinct paired contents. Each row
has one origin and the same number of anonymous output slots. Paths resolve
against the JSON file, so the folder can be moved to another machine.

```json
{
  "supports": [
    {"before": "images/source0.png", "after": ["images/slot0_source0.png", "images/slot1_source0.png"]}
  ]
}
```

The example above illustrates one row; provide eight rows. RGB PNG/JPEG files,
CHW/HWC uint8 arrays, and normalized floating-point arrays are supported. An
NPZ with `before[8,3,H,W]` and `after[8,G,3,H,W]` is also accepted.

```sh
python -m chainprint fit --mode diagnostic-joint --pairs demo-output/pairs.json --output my-model.pt
python -m chainprint compose --library my-model.pt --before demo-output/query_before.png --history "[0,1]" --output predicted.png
python -m chainprint rank --library my-model.pt --before demo-output/query_before.png --after demo-output/query_after.png --candidates demo-output/candidates.json --output ranking.json
```

Fitted model files have an adjacent SHA-bound JSON receipt. Output files and
demo directories must be fresh. Anonymous slot IDs belong to that fitted bank;
they are not global operation labels. Repeated operation IDs are allowed in a
history. Ranking uses endpoint pixel MSE and reports numerical ties explicitly.

## Original submitted operation banks

```sh
python -m chainprint inspect --cell 0
python -m chainprint compose --cell 0 --before demo-output/frozen_bank_input.png --history "[0]" --output original-bank-prediction.png
```

All six original files are included, totaling **6,571,774 bytes**. Both file
SHA-256 and numerical parameter hashes are checked at load time. The package
includes the original Python classes needed to deserialize them. These are
support-specific operation libraries, not generic neural detector checkpoints.

The default assets ship with the installed package. `--asset-root DIR` or
`CHAINPRINT_ASSET_ROOT` can select a relocated directory containing
`bank_manifest.json` and `models/`. No server path is needed to use this API.

Some slots invoke the declared Stable Diffusion inpainting prior. Its weights
are not included. A missing editor makes that candidate unavailable; it never
receives a fabricated zero loss. If any candidate is unavailable, the full-set
decision abstains and has no declared winner. A separately named conditional
best among available candidates is diagnostic only. Conventional slots run
without an editor.
For explicit CPU execution with an existing pinned Hugging Face cache, install
the `ai` extra and pass `--editor-cache DIR`. This stays offline and uses the
original checker and pinned revision. AI execution has not been rerun here.

## Complete frozen fitting hierarchy

`fit(..., mode="paper")` wraps the unmodified
`opweave_frozen_20260916.OpWeaveEstimator`. It preserves the pooled, threshold,
gain, unit-gain, and integer-local stages and the support-only selection rules.
It requires exactly eight 256x256 before images, four or eight anonymous output
slots, a declared finite editor prior, and exact pre-existing editor outputs for
those same supports. No editor is silently called or downloaded during fitting.

```sh
python -m chainprint fit --mode paper --pairs my-pairs.npz --prior examples/paper_prior.json --prior-cache my-exact-prior-cache.npz --output fitted-paper.pt --maximum-seconds 600
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the cache contract and the
difference between the two fit modes. This hierarchy and the original banks
are provided; a full new-support refit and the complete image benchmark have
not been rerun as part of this candidate verification.

Python API:

```python
from chainprint import fit, compose, rank, load_library

library = load_library(cell=0)
prediction = compose(before_rgb, [0, 2, 0], library)
result = rank(before_rgb, observed_rgb, [[0, 2], [2, 0]], library)
# New paired RGB arrays; this mode is the reduced diagnostic, explicitly:
fitted = fit(before_supports, after_supports, mode="diagnostic-joint")
```

## Supplementary two-dimensional control

This is separate from the RGB image demo and estimator.

```sh
python -m chainprint control --no-plots --output direction-results
```

The numerical CSVs are checked against the submitted reference files. Plotting
is optional (`pip install .[plots]`). The only control-code changes are a relative
helper import, optional plotting and a fresh-output guard; grids, seeds,
enumeration, and numerical rules are unchanged.

## Verification and remaining scope

```sh
python -m pip install .[test]
python -m pytest
```

`VERIFY.json` records model checks, CPU execution, installation and tests.
The full historical benchmark still needs the licensed FiveK/Flickr8k/COCO
images, exact support/query manifests and caches, editor assets, and baseline
adaptations. The historical frozen contracts are retained unchanged under
`provenance/frozen/`; they are evidence, not portable executable configs.

This repository publishes a core research release candidate. The original
repository license remains unselected; see `NOTICE` for the current status.
