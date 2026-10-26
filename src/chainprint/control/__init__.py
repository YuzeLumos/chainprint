"""Portable supplementary 2D control; separate from the image-estimator demo."""
from pathlib import Path

def run(output, *, plots=True):
    from . import experiment
    experiment.ROOT = Path(output).resolve()
    experiment.main(plots=plots)
