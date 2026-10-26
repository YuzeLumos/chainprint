# Frozen evidence and portable interfaces

`SOURCE_MANIFEST.json` maps packaged files to their exact source paths, SHA-256
digests and transformations. Forty frozen scientific modules are copied byte
for byte. They contain the current operation-function fitter and its runtime
dependency closure, rather than the old v2 training repository.

`frozen/aggregate.json`, `RUN_CONTRACT.json`, `SLOT_INVARIANCE.json` and
`SHARED_POOL_MANIFEST.json` are original eight-slot fit/support records.
`frozen/final_closeout/` retains the original strict G=8 runner, protocols,
contract and report unchanged. `frozen/runners/` contains historical fitting
entry points. These records can contain original author-machine absolute
paths; they are archival evidence and are not rewritten or executed by the
portable package.

Portable execution uses installed package modules, explicit input paths and
`--asset-root` (or packaged defaults). The top-level import names of numerical
modules and the `history_recovery` subset are preserved because the original
parameter files refer to those exact Python class names.

The 2D control reference is taken from the citation-updated supplementary
archive accompanying the local submission PDF. Its portable variant changes
the helper import, makes plotting optional, and requires a fresh output folder.
Its numerical procedure is unchanged.
