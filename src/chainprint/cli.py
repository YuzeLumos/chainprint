"""Command line interface with explicit assets and caller-owned input paths."""
from pathlib import Path
import argparse
import json
import numpy as np

from .api import (compose, fit, load_fitted, load_library, rank, save_image,
                  save_library)


def read_pairs(path):
    """JSON paths resolve against the JSON file, never an author's machine."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return data["before"].copy(), data["after"].copy()
    record = json.loads(path.read_text(encoding="utf-8"))
    rows = record["supports"]
    return ([path.parent / row["before"] for row in rows],
            [[path.parent / value for value in row["after"]] for row in rows])


def _library(args):
    return load_fitted(args.library) if args.library else load_library(args.cell, asset_root=args.asset_root)


def _write(result, path=None):
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if path:
        path = Path(path)
        if path.exists():
            raise FileExistsError("Choose a new result path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ChainPrint CPU fitting, composition and ranking")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Verify and inspect one bundled bank")
    fit_cmd = commands.add_parser("fit", help="Fit new paired supports")
    fit_cmd.add_argument("--pairs", required=True, type=Path)
    fit_cmd.add_argument("--mode", required=True, choices=["paper", "diagnostic-joint"])
    fit_cmd.add_argument("--prior", type=Path)
    fit_cmd.add_argument("--prior-cache", type=Path)
    fit_cmd.add_argument("--output", required=True, type=Path)
    fit_cmd.add_argument("--maximum-seconds", type=float, default=600)
    fit_cmd.add_argument("--threads", type=int, default=1)
    compose_cmd = commands.add_parser("compose", help="Predict an ordered candidate history")
    compose_cmd.add_argument("--before", required=True, type=Path)
    compose_cmd.add_argument("--history", required=True, help="JSON list, e.g. [0,2,0]")
    compose_cmd.add_argument("--output", required=True, type=Path)
    rank_cmd = commands.add_parser("rank", help="Rank candidate histories against an observed endpoint")
    rank_cmd.add_argument("--before", required=True, type=Path)
    rank_cmd.add_argument("--after", required=True, type=Path)
    rank_cmd.add_argument("--candidates", required=True, type=Path, help="JSON array of anonymous-ID histories")
    rank_cmd.add_argument("--output", type=Path)
    for item in (inspect, compose_cmd, rank_cmd):
        item.add_argument("--cell", type=int, default=0)
        item.add_argument("--asset-root", type=Path)
        item.add_argument("--library", type=Path, help="A fitted .pt with its adjacent receipt")
        item.add_argument("--threads", type=int, default=1)
        item.add_argument("--editor-cache", type=Path, help="Existing pinned Hugging Face cache; CPU/offline editor execution is explicit")
    demo = commands.add_parser("demo", help="Generate RGB pairs, fit real library primitives, and rank unseen image histories")
    demo.add_argument("--output", type=Path, required=True)
    control = commands.add_parser("control", help="Recompute the supplementary 2D control separately")
    control.add_argument("--output", required=True, type=Path)
    control.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "fit":
        before, after = read_pairs(args.pairs)
        library = fit(before, after, mode=args.mode, prior=args.prior,
                      prior_cache=args.prior_cache, threads=args.threads,
                      maximum_seconds=args.maximum_seconds)
        save_library(library, args.output)
        _write({"output": str(args.output.resolve()), "mode": library.metadata["mode"],
                "operation_count": len(library), "model_hash": library.metadata["model_hash"]})
    elif args.command == "compose":
        if args.output.exists():
            raise FileExistsError("Choose a new predicted-image path")
        pixels = compose(args.before, json.loads(args.history), _library(args), threads=args.threads,
                         editor_backend=_editor(args))
        save_image(args.output, pixels)
        _write({"output": str(args.output.resolve()), "shape": list(pixels.shape)})
    elif args.command == "rank":
        candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
        _write(rank(args.before, args.after, candidates, _library(args), threads=args.threads,
                    editor_backend=_editor(args)), args.output)
    elif args.command == "inspect":
        library = _library(args)
        _write({"metadata": library.metadata, "operators": [
            {"slot": i, "kind": op[0], "needs_editor": _needs_editor(op)}
            for i, op in enumerate(library.operators)]})
    elif args.command == "demo":
        from .demo import run_demo
        _write(run_demo(args.output))
    elif args.command == "control":
        from .control import run
        run(args.output, plots=not args.no_plots)
    return 0


def _needs_editor(operator):
    return operator[0] == "ai_inpaint" or (operator[0] == "shared_residual" and _needs_editor(operator[1][0]))


def _editor(args):
    if args.editor_cache is None:
        return None
    if not args.editor_cache.is_dir():
        raise FileNotFoundError("An existing editor cache directory is required")
    from ai_operator import DiffusersInpaintBackend
    return DiffusersInpaintBackend(device="cpu", dtype="float32", local_files_only=True,
                                  cache_dir=str(args.editor_cache.resolve()))


if __name__ == "__main__":
    raise SystemExit(main())
