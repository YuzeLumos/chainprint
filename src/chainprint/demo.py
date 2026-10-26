"""Fresh RGB support images and an unseen-content order-inference example."""
from pathlib import Path
import json
import numpy as np

from .api import compose, fit, load_library, rank, save_image, save_library


def _render(image, slot):
    # Rendering is isolated from fitting: no operation labels or true parameters
    # are passed to fit(), which only receives the paired RGB pixels.
    if slot == 0:
        matrix = np.array([[0.70, 0.15, 0.05], [0.10, 0.75, 0.05], [0.05, 0.10, 0.70]])
        value = np.einsum("ij,jhw->ihw", matrix, image) + np.array([0.08, 0.03, 0.12])[:, None, None]
        return np.round(value.clip(0, 1)*255).astype(np.float32)/255
    # Same declared generic quantizer family as the original replay library.
    levels = 5
    lut = np.array([int(round(v/(255/(levels-1)))*(255/(levels-1))) for v in range(256)])
    return lut[np.rint(image*255).astype(np.int64)].astype(np.float32)/255


def demo_arrays(seed=1971, size=32):
    rng = np.random.default_rng(seed)
    before = rng.integers(0, 256, (8, 3, size, size), dtype=np.uint8).astype(np.float32)/255
    after = np.stack([np.stack([_render(image, slot) for slot in range(2)]) for image in before])
    query = rng.integers(0, 256, (3, size, size), dtype=np.uint8).astype(np.float32)/255
    observed = _render(_render(query, 0), 1)
    return before, after, query, observed


def run_demo(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh demo directory")
    output.mkdir(parents=True)
    before, after, query, observed = demo_arrays()
    supports = []
    for k in range(8):
        before_path = f"inputs/support_{k}_before.png"
        after_paths = [f"inputs/support_{k}_slot_{g}.png" for g in range(2)]
        save_image(output/before_path, before[k])
        for g, path in enumerate(after_paths):
            save_image(output/path, after[k, g])
        supports.append({"before": before_path, "after": after_paths})
    (output/"pairs.json").write_text(json.dumps({"supports": supports}, indent=2)+"\n")
    candidates = [[0, 1], [1, 0], [0, 0], [1, 1]]
    (output/"candidates.json").write_text(json.dumps(candidates)+"\n")
    save_image(output/"query_before.png", query)
    save_image(output/"query_after.png", observed)
    library = fit(before, after, mode="diagnostic-joint")
    save_library(library, output/"fitted.pt")
    result = rank(query, observed, candidates, library)
    save_image(output/"predicted.png", compose(query, [0, 1], library))
    if result["best_candidate_indices"] != [0]:
        raise AssertionError("Fresh image order-recovery diagnostic did not prefer the generating history")
    # Exercise the actual submitted parameter object on a separate generated
    # 256px input. This checks compatibility; it is not held-out accuracy.
    bank = load_library(0)
    pixels = np.random.default_rng(99).integers(0, 256, (3, 256, 256), dtype=np.uint8)
    frozen_prediction = compose(pixels, [0], bank)
    save_image(output/"frozen_bank_input.png", pixels)
    save_image(output/"frozen_bank_prediction.png", frozen_prediction)
    result.update({"demo_scope": "Fresh generated RGB images; reduced original analytical joint diagnostic, not the full paper benchmark",
                   "paired_supports": 8, "query_content_not_in_supports": True,
                   "model_hash": library.metadata["model_hash"],
                   "frozen_bank_check": {"cell": 0, "slot": 0,
                       "finite": bool(np.isfinite(frozen_prediction).all()),
                       "prediction_mse_from_input": float(np.square(frozen_prediction-pixels/255).mean()),
                       "scope": "Original parameter loading and real operator execution; no accuracy claim"}})
    (output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result
