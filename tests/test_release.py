from pathlib import Path
import json
import shutil
import numpy as np
import pytest
import torch

from chainprint import (MissingEditorAsset, compose, fit, load_fitted,
                        load_library, rank, save_library)
from chainprint.api import image_array, sha256
from chainprint.cli import read_pairs
from chainprint.demo import demo_arrays, run_demo

ROOT = Path(__file__).resolve().parents[1]


def test_all_six_real_model_banks_match_original_bytes_and_numerical_state():
    for cell in range(6):
        library = load_library(cell)
        assert len(library) == 8
        assert library.metadata["state_hash_verified"]
        assert library.metadata["source"]["support_count"] == 8


def test_real_operator_predicts_pixels_and_roundtrips_exactly():
    library = load_library(0)
    image = np.random.default_rng(17).integers(0, 256, (3, 256, 256), dtype=np.uint8)
    prediction = compose(image, [0], library)
    model = library.operators[0][1]["quantized_separable_model"]
    direct = model.predict(torch.as_tensor(image/255, dtype=torch.float64)).numpy()
    np.testing.assert_array_equal(prediction, direct)
    np.testing.assert_array_equal(prediction, compose(image, [0], library))
    assert prediction.shape == (3, 256, 256) and np.isfinite(prediction).all()
    assert np.square(prediction-image/255).mean() > 1e-4
    result = rank(image, prediction, [[0], [2], [3]], library)
    assert result["best_candidate_indices"] == [0]


def test_missing_ai_asset_never_receives_a_fabricated_zero_loss():
    library = load_library(0)
    image = np.full((3, 256, 256), 0.5, dtype=np.float32)
    result = rank(image, image, [[1], [5]], library)
    assert result["abstained"] and not result["best_candidate_indices"]
    assert all(row["mse"] is None and not row["available"] for row in result["ranking"])
    with pytest.raises(MissingEditorAsset):
        compose(image, [1], library)


def test_mixed_missing_ai_candidate_prevents_a_complete_set_winner():
    library = load_library(0)
    image = np.random.default_rng(12).random((3, 256, 256), dtype=np.float32)
    endpoint = compose(image, [0], library)
    result = rank(image, endpoint, [[0], [1]], library)
    assert not result["complete_candidates"] and result["abstained"]
    assert result["best_candidate_indices"] == []
    assert result["conditional_best_available_indices"] == [0]
    assert all(row["rank"] is None for row in result["ranking"])
    assert result["ranking"][0]["rank_within_available"] == 1
    assert result["ranking"][1]["mse"] is None


def test_small_traditional_input_with_missing_ai_candidate_still_abstains_cleanly():
    library = load_library(0)
    image = np.random.default_rng(19).random((3, 32, 32), dtype=np.float32)
    endpoint = compose(image, [0], library)
    result = rank(image, endpoint, [[0], [1]], library)
    assert result["abstained"] and not result["complete_candidates"]
    assert result["conditional_best_available_indices"] == [0]
    assert result["ranking"][1]["unavailable_reason"].startswith("MissingEditorAsset")


def test_relocated_asset_root_and_tamper_detection(tmp_path):
    assets = ROOT/"src/chainprint/assets"
    moved = tmp_path/"a new asset root"
    shutil.copytree(assets, moved)
    assert load_library(3, asset_root=moved).metadata["cell"] == 3
    path = moved/"models/cell_3_anonymous_models.pt"
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_library(3, asset_root=moved)


def test_new_image_pairs_fit_and_rank_without_author_paths(tmp_path, monkeypatch):
    output = tmp_path/"fresh generated images"
    result = run_demo(output)
    monkeypatch.chdir(tmp_path)
    before_paths, after_paths = read_pairs(output/"pairs.json")
    library = fit(before_paths, after_paths, mode="diagnostic-joint")
    before, after, query, endpoint = demo_arrays()
    assert all(path.is_file() for path in before_paths)
    assert library.metadata["mode"] == "diagnostic-joint"
    restored = load_fitted(output/"fitted.pt")
    candidates = json.loads((output/"candidates.json").read_text())
    a = rank(output/"query_before.png", output/"query_after.png", candidates, library)
    b = rank(query, endpoint, candidates, restored)
    assert a["best_candidate_indices"] == b["best_candidate_indices"] == [0]
    assert library.metadata["model_hash"] == restored.metadata["model_hash"] == result["model_hash"]
    assert [r["mse"] for r in a["ranking"]] == [r["mse"] for r in b["ranking"]]


def test_validation_and_fitted_receipt(tmp_path):
    before, after, query, endpoint = demo_arrays()
    with pytest.raises(ValueError, match="distinct"):
        fit(np.repeat(before[:1], 8, axis=0), after, mode="diagnostic-joint")
    with pytest.raises(ValueError, match="four or eight"):
        fit(before, after, mode="paper")
    with pytest.raises(ValueError, match="valid anonymous"):
        compose(query, [999], load_library(0))
    with pytest.raises(ValueError, match="finite"):
        image_array(np.full((3, 32, 32), np.nan))
    library = fit(before, after, mode="diagnostic-joint")
    path = save_library(library, tmp_path/"models.pt")
    with pytest.raises(FileExistsError):
        save_library(library, path)
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_fitted(path)


def test_identical_candidates_are_reported_as_tied():
    library = load_library(0)
    image = np.random.default_rng(21).random((3, 32, 32), dtype=np.float32)
    prediction = compose(image, [0], library)
    result = rank(image, prediction, [[0], [0]], library)
    assert result["best_candidate_indices"] == [0, 1]
    assert [r["rank"] for r in result["ranking"]] == [1, 1]


def test_source_bytes_remain_bound_to_the_recorded_frozen_modules():
    manifest = json.loads((ROOT/"provenance/SOURCE_MANIFEST.json").read_text())
    original = [row for row in manifest["files"] if row.get("module")]
    assert len(original) == 40
    for row in original:
        assert sha256(ROOT/row["path"]) == row["sha256"]


def test_2d_control_numerics_are_unchanged_and_need_no_plotting(tmp_path):
    from chainprint.control import run
    output = tmp_path/"control"
    run(output, plots=False)
    for name in ["cases.csv", "summary.csv", "sampling_sensitivity.csv", "design_balance.csv", "counterexamples.csv"]:
        assert (output/name).read_bytes() == (ROOT/"examples/direction_reference"/name).read_bytes()
