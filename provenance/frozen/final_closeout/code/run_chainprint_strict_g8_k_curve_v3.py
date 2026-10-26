#!/usr/bin/env python3
"""Prospective zero-training ChainPrint G=8 conventional K-curve closeout.

This script scores the frozen ChainPrint independent-fit operators on the same
conventional L=8/N=16 source/query/candidate episodes and the same eight-slot
support prefixes used by the existing comparison methods.  It performs no
editor calls, no fitting, no weight updates, and retains every registered
support-prefix failure.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


KS = (2, 3, 4, 6, 8, 10, 12, 16)
DATASETS = ("FiveK", "Flickr8k", "COCO")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_record = read(args.protocol)
    if "parent_protocol" in protocol_record:
        parent_protocol = Path(protocol_record["parent_protocol"])
        if sha256(parent_protocol) != protocol_record["parent_protocol_sha256"]:
            raise ValueError("Parent protocol bytes differ from the recovery contract")
        protocol = read(parent_protocol)
        protocol.update(protocol_record["overrides"])
        protocol["engineering_recovery"] = {
            key: value for key, value in protocol_record.items()
            if key not in {"overrides"}
        }
    else:
        protocol = protocol_record
    base = Path(protocol["base"])
    root = Path(protocol["root"])
    output = args.output
    if output.exists():
        raise FileExistsError("Never overwrite the prospective G=8 closeout")
    if sha256(Path(__file__)) != protocol["code_sha256"]:
        raise ValueError("Runner bytes differ from frozen protocol")
    for path, expected in protocol["input_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    if shutil.disk_usage(base).free < protocol["min_free_bytes"]:
        raise OSError("Preserve the frozen disk reserve")

    output.mkdir(parents=False)
    started = time.monotonic()

    def bounds() -> None:
        if time.monotonic() - started > protocol["maximum_seconds"]:
            raise TimeoutError("Frozen four-hour wall-clock bound")
        if shutil.disk_usage(base).free < protocol["min_free_bytes"]:
            raise OSError("Preserve the frozen disk reserve")
        size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if size > protocol["output_cap_bytes"]:
            raise OSError("Frozen output cap exceeded")

    original = read(Path(read(root / "stage1_protocol.json")["original_protocol"]))
    repo = Path(original["repo"])
    code = base / "code"
    sys.path[:0] = [
        str(code), str(root / "code"), str(root), *original["helper_roots"],
        str(repo / "scripts"), str(repo / "src"),
    ]
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )

    import numpy as np
    import torch
    from ai_operator import AIReplay
    from available_inference import MemoryAvailableAIReplay, available_distances, compose_available
    from chainprint_shared_residual import ResidualReplay
    from final_local_extension import final_parameter_hash
    from gain_block import GainReplay
    from native_k_reference_bindings_v1 import column_keys, rebind_columns
    from native_k_support_bindings_v1 import prefix_available, validate_banks
    from run_real_author_k_reference_v1 import load_registered_pools
    from run_same_endpoint_span_cpu_v1 import ranked

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)

    class NoEditorBackend:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = 0

        def manifest(self):
            return {
                "kind": "NO_EDITOR_CALLS_CONVENTIONAL_ONLY",
                "loaded": False,
                "torch_deterministic_algorithms_current": True,
            }

        def __call__(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("Frozen conventional candidate bank attempted an AI editor call")

    manifest = read(base / "manifest_v2/PRIMARY_EVALUATOR_MANIFEST_v2.json")
    if tuple(manifest["dataset_order"]) != DATASETS:
        raise ValueError("Dataset order changed")
    designs = {row["group"]: row for row in manifest["designs_evaluator_only"]}
    if set(designs) != set(range(60)):
        raise ValueError("Expected the frozen 60 source groups")

    before16, banks, masks, pixel_receipts = load_registered_pools(base)
    support_fit_path = Path(protocol["support_fit_aggregate"])
    fits = read(support_fit_path)
    validate_banks(fits["banks"], masks)
    if fits["masks"] != {str(cell): value for cell, value in masks.items()}:
        raise ValueError("Support availability mask changed")

    fitted = {}
    model_file_sha256 = {}
    model_state_before = {}
    for row in fits["banks"]:
        if row["method"] != "ChainPrint" or not row["support_prefix_available"]:
            continue
        path = Path(row["model_path"])
        if sha256(path) != row["model_file_sha256"]:
            raise ValueError(f"Frozen support-fit file changed: {path}")
        models = torch.load(path, map_location="cpu", weights_only=False)
        if final_parameter_hash(models) != row["model_hash"]:
            raise ValueError("Frozen support-fit numerical state changed")
        key = (row["K"], row["cell"])
        fitted[key] = (models, row["column_pixel_keys"])
        model_file_sha256[str(path)] = row["model_file_sha256"]
        model_state_before[f"K{row['K']}_cell{row['cell']}"] = row["model_hash"]

    original_fit_summary = read(Path(protocol["k8_fit_aggregate"]))
    k8_cache = {}
    for bank in original_fit_summary["banks"]:
        path = Path(bank["model_path"])
        if sha256(path) != bank["model_file_sha256"]:
            raise ValueError(f"Frozen K=8 model file changed: {path}")
        models = torch.load(path, map_location="cpu", weights_only=False)
        if final_parameter_hash(models) != bank["model_hash"]:
            raise ValueError("Frozen K=8 numerical state changed")
        for key, model in zip(bank["column_pixel_keys"], models):
            if key in k8_cache and final_parameter_hash(k8_cache[key]) != final_parameter_hash(model):
                raise ValueError("Duplicate support pixels map to different K=8 models")
            k8_cache[key] = model
        model_file_sha256[str(path)] = bank["model_file_sha256"]
        model_state_before[f"K8_cell{bank['cell']}"] = bank["model_hash"]

    parent_path = Path(protocol["chainprint_parent_records"])
    comparator_path = Path(protocol["g8_comparator_records"])
    parent = {}
    for row in jsonl(parent_path):
        if row["condition"] == "conventional" and row["span"] == 8 and row["K"] == 8:
            key = (row["dataset"], row["group"])
            if key in parent:
                raise ValueError("Duplicate ChainPrint parent episode")
            parent[key] = row
    comparator = {}
    for row in jsonl(comparator_path):
        if (
            row["method"] == protocol["comparator_method"]
            and row["condition"] == "conventional"
            and row["span"] == 8
            and row["K"] == 8
        ):
            key = (row["dataset"], row["group"])
            if key in comparator:
                raise ValueError("Duplicate G=8 comparator episode")
            comparator[key] = row
    expected_keys = {(dataset, group) for dataset in DATASETS for group in range(60)}
    if set(parent) != expected_keys or set(comparator) != expected_keys:
        raise ValueError("Exactly 180 frozen parent/comparator episodes required")
    for key in sorted(expected_keys):
        left, right = parent[key], comparator[key]
        for field in ("dataset", "group", "source_id", "source_root", "config_cell", "candidate_words", "truth_index"):
            if left[field] != right[field]:
                raise ValueError(f"Parent/comparator mismatch in {field}: {key}")
        right_query_hash = right.get("query_npz_sha256", right.get("query_asset_sha256"))
        query = base / f"chainprint_conventional_formal_v2/query_{key[0]}_{key[1]:03d}_L8.npz"
        if sha256(query) != right_query_hash:
            raise ValueError(f"Comparator query receipt changed: {key}")

    support_binding = {
        "before16_pixels_sha256": hashlib.sha256(before16.tobytes()).hexdigest(),
        "banks": pixel_receipts,
        "masks": masks,
        "prefix_indices": {str(k): list(range(k)) for k in KS},
        "no_fill_drop_retry": True,
    }
    write(output / "SUPPORT_PREFIX_PIXEL_BINDING.json", support_binding)
    contract = {
        "schema": "chainprint-strict-g8-conventional-k-curve-v1",
        "protocol": protocol,
        "protocol_sha256": sha256(args.protocol),
        "runner_sha256": sha256(Path(__file__)),
        "started_utc": now(),
        "pid": os.getpid(),
        "device": "CPU_ONLY",
        "expected_records": 180 * len(KS),
        "expected_available_records": 1080,
        "expected_failure_records": 360,
        "editor_call_budget": 0,
        "new_fits": 0,
        "new_weight_updates": 0,
        "same_episode_parent_sha256": sha256(parent_path),
        "same_g8_comparator_sha256": sha256(comparator_path),
        "support_binding_sha256": sha256(output / "SUPPORT_PREFIX_PIXEL_BINDING.json"),
        "model_file_sha256": model_file_sha256,
        "no_paper_edits": True,
    }
    write(output / "RUN_CONTRACT.json", contract)

    no_editor = NoEditorBackend()
    replay = ResidualReplay(
        MemoryAvailableAIReplay(AIReplay(no_editor, GainReplay(torch.device("cpu"))), capacity=256)
    )
    rows = []
    parameter_hashes = {}
    actual_score_calls = 0
    records_path = output / "episode_records.jsonl"
    with torch.no_grad(), records_path.open("x", encoding="utf-8") as handle:
        for dataset in DATASETS:
            for group in range(60):
                bounds()
                source = parent[(dataset, group)]
                cell = source["config_cell"]
                mapping = designs[group]["canonical_to_anonymous_evaluator_only"]
                if sorted(mapping) != list(range(8)):
                    raise ValueError("Frozen local-ID mapping is not a bijection")
                legal_conventional_ids = set(mapping[:4])
                used_ids = {value for word in source["candidate_words"] for value in word}
                if not used_ids <= legal_conventional_ids:
                    raise ValueError("Conventional bank attempted to use an AI distractor operation")
                query = base / f"chainprint_conventional_formal_v2/query_{dataset}_{group:03d}_L8.npz"
                with np.load(query, allow_pickle=False) as payload:
                    early = payload["earlier"].copy()
                    late = payload["later"].copy()
                if early.shape != (3, 256, 256) or late.shape != early.shape:
                    raise ValueError("Frozen query geometry changed")

                for k in KS:
                    bounds()
                    available = prefix_available(masks[cell], k)
                    missing = [
                        [i, j]
                        for i in range(k)
                        for j, value in enumerate(masks[cell][i])
                        if not value
                    ]
                    model_hash = None
                    if not available:
                        scores = [None] * 16
                    else:
                        canonical = np.stack([np.stack(row) for row in banks[cell][:k]])
                        desired = column_keys(before16[:k], canonical, mapping)
                        if k == 8:
                            operators = [k8_cache[key] for key in desired]
                        else:
                            models, keys = fitted[(k, cell)]
                            operators = rebind_columns(models, keys, desired, final_parameter_hash)
                        model_hash = final_parameter_hash(operators)
                        predictions, _ = compose_available(replay, early, operators, source["candidate_words"])
                        losses, valid = available_distances(predictions, late)
                        if not bool(valid.all()):
                            raise ValueError("Conventional-only candidate unexpectedly invoked an unavailable editor")
                        scores = [-float(value) for value in losses]
                        actual_score_calls += 1
                        if final_parameter_hash(operators) != model_hash:
                            raise ValueError("Operator state changed during inference")
                    metrics = ranked(scores, source["truth_index"])
                    row = dict(source)
                    row.update(
                        K=k,
                        scores=scores,
                        **metrics,
                        support_dictionary_G=8,
                        support_prefix_indices=list(range(k)),
                        support_prefix_available=available,
                        missing_support_pairs=missing,
                        operator_parameter_hash=model_hash,
                        exact_parent_K8_score_reuse=False,
                        strict_interface_parity=True,
                        parent_chainprint_g4_records_sha256=sha256(parent_path),
                        g8_comparator_records_sha256=sha256(comparator_path),
                        support_prefix_pixel_binding_sha256=contract["support_binding_sha256"],
                        generation_complete=True,
                        failure=None if available else "REGISTERED_G8_SUPPORT_PREFIX_INCOMPLETE",
                        encoder_updates_at_test=0,
                        weight_updates_at_inference=0,
                        new_fit_calls=0,
                        new_editor_calls=0,
                        formal_efficiency=False,
                        query_seconds=None,
                        raw_ranking=None,
                        origin="FROZEN_CHAINPRINT_G8_SUPPORT_OPERATOR_ENDPOINT_REPLAY",
                    )
                    key = f"{dataset}|{group}|K{k}"
                    parameter_hashes[key] = model_hash
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                write(output / "STATUS.json", {
                    "status": "SCORING_STRICT_G8_CHAINPRINT_CONVENTIONAL_K",
                    "records": len(rows),
                    "expected_records": 180 * len(KS),
                    "dataset": dataset,
                    "group": group,
                    "actual_score_calls": actual_score_calls,
                    "new_editor_calls": no_editor.calls,
                    "utc": now(),
                })

    expected_complete = sum(prefix_available(masks[designs[group]["config_cell"]], k)
                            for _dataset in DATASETS for group in range(60) for k in KS)
    if len(rows) != 1440 or expected_complete != 1080 or actual_score_calls != expected_complete:
        raise ValueError("Frozen record/coverage denominator changed")
    if no_editor.calls != 0:
        raise ValueError("Zero-editor-call contract violated")
    for key, expected in model_state_before.items():
        k_part, cell_part = key.split("_")
        k = int(k_part[1:])
        cell = int(cell_part[4:])
        if k == 8:
            # K=8 cache is rebound across all cells; the per-cell file was already
            # checked above and no numerical mutation can occur under no_grad.
            continue
        models, _keys = fitted[(k, cell)]
        if final_parameter_hash(models) != expected:
            raise ValueError("Frozen support-fit state changed after inference")

    write(output / "QUERY_PARAMETER_BINDING.json", parameter_hashes)
    result = {
        "status": "COMPLETE_STRICT_G8_CHAINPRINT_CONVENTIONAL_K_FAILURES_RETAINED",
        "records": len(rows),
        "records_sha256": sha256(records_path),
        "protocol_sha256": sha256(args.protocol),
        "run_contract_sha256": sha256(output / "RUN_CONTRACT.json"),
        "support_binding_sha256": sha256(output / "SUPPORT_PREFIX_PIXEL_BINDING.json"),
        "query_parameter_binding_sha256": sha256(output / "QUERY_PARAMETER_BINDING.json"),
        "actual_score_calls": actual_score_calls,
        "complete_records": expected_complete,
        "failure_records": len(rows) - expected_complete,
        "new_editor_calls": 0,
        "new_fit_calls": 0,
        "new_weight_updates": 0,
        "strict_interface_parity": True,
        "no_paper_edits": True,
        "finished_utc": now(),
    }
    write(output / "aggregate.json", result)
    write(output / "STATUS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--output") + 1])
            if path.exists():
                write(path / "FAILED.json", {
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "utc": now(),
                })
        raise
