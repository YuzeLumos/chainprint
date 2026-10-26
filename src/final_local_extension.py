"""Final support-only integer-local candidate selection; old models stay frozen."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch

from extended_mixed_estimator import mixed_parameter_hash
from integer_local_response import IntegerLocalResponse

TOL = 1e-10
SUPPORT_COUNT = 8
SLOT_COUNT = 4


def final_parameter_hash(value):
    """Extend the old hash only for IntegerLocalResponse's complete identity."""
    def normalize(item):
        if isinstance(item, IntegerLocalResponse):
            return ("__type__", "integer_local_response.IntegerLocalResponse:v1",
                    item.training, dict(item.state_dict()), item.parameter_record())
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, tuple):
            return tuple(normalize(child) for child in item)
        if isinstance(item, dict):
            return {normalize(key): normalize(child) for key, child in item.items()}
        return item
    return mixed_parameter_hash(normalize(value))


def extend_integer_local(base, before, after, defaults, default_held,
                         default_available, *, check_budget=None):
    """Fit from anonymous paired supports and saved support-only defaults.

    before is [8,3,H,W], after is [8,4,3,H,W]. Every proposal has exactly one
    source content; held h excludes source h in both proposal and mean selection.
    """
    if base.device.type != "cpu":
        raise ValueError("The frozen final extension requires CPU replay")
    before_shape, after_shape = tuple(before.shape), tuple(after.shape)
    if len(before_shape) != 4 or before_shape[:2] != (SUPPORT_COUNT, 3):
        raise ValueError("Expected eight CHW RGB support inputs")
    if after_shape != (SUPPORT_COUNT, SLOT_COUNT, *before_shape[1:]):
        raise ValueError("Expected four anonymous outputs per support input")
    if any(len(value) != SLOT_COUNT for value in (defaults, default_held, default_available)):
        raise ValueError("Defaults must describe four anonymous slots")
    clean_held, available = [], []
    for losses, declared in zip(default_held, default_available):
        if len(losses) != SUPPORT_COUNT:
            raise ValueError("Expected eight saved held losses per slot")
        clean = []
        for value in losses:
            if value is None:
                clean.append(None)
            elif not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError("Invalid default held-content MSE8")
            else:
                clean.append(float(value))
        clean_held.append(clean)
        available.append(bool(declared) and all(value is not None for value in clean))

    def budget():
        if check_budget is not None:
            check_budget()

    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    default_hash = final_parameter_hash(defaults)
    models = list(defaults)
    slots, selected_held = [], []
    try:
        budget()
        with torch.inference_mode(False):
            with torch.no_grad():
                images = [base.image(before[k]) for k in range(SUPPORT_COUNT)]
                targets = [[base.image(after[k, g]) for g in range(SLOT_COUNT)]
                           for k in range(SUPPORT_COUNT)]
                values = images + [image for row in targets for image in row]
                if any(x.device.type != "cpu" or x.dtype != torch.float64 for x in values):
                    raise ValueError("Expected existing CPU float64 replay inputs")
                if not all(bool(torch.isfinite(x).all()) for x in values):
                    raise ValueError("Non-finite paired support arrays")
                height, width = before_shape[-2:]
                mask = ((torch.arange(height)[:, None] + torch.arange(width)[None, :]) % 4 != 0).expand_as(images[0])

            for g in range(SLOT_COUNT):
                slot_started = time.perf_counter()
                proposals, proposal_records = [], []
                losses = np.empty((SUPPORT_COUNT, SUPPORT_COUNT), dtype=np.float64)
                for k in range(SUPPORT_COUNT):
                    budget()
                    fit_started = time.perf_counter()
                    with torch.enable_grad():
                        trained, diagnostic = IntegerLocalResponse.fit_support(
                            before[k], after[k, g], mask, torch.device("cpu"),
                            check_budget=check_budget)
                    budget()
                    if not isinstance(trained, IntegerLocalResponse) or not isinstance(diagnostic, dict):
                        raise TypeError("Expected IntegerLocalResponse and a diagnostic dictionary")
                    json.dumps(diagnostic, allow_nan=False)
                    trained_hash = final_parameter_hash(trained)
                    candidate = copy.deepcopy(trained).eval().requires_grad_(False)
                    for name, value in candidate.state_dict().items():
                        if not torch.equal(value, trained.state_dict()[name]):
                            raise ValueError("Freezing the candidate changed a fitted parameter")
                    if final_parameter_hash(trained) != trained_hash:
                        raise ValueError("Returned trained model was changed in place")
                    operator = ("spatial", {"quantized_separable_model": candidate})
                    proposals.append(operator)
                    prediction_started = time.perf_counter()
                    with torch.no_grad():
                        for j, image in enumerate(images):
                            prediction = base.apply(image, operator)
                            loss = float(((prediction - targets[j][g]) * 255).square().mean())
                            if not np.isfinite(loss) or loss < 0:
                                raise ValueError("Invalid integer-local support prediction loss")
                            losses[k, j] = loss
                    budget()
                    proposal_records.append(dict(
                        proposal_index=k, source_contents=[k], diagnostic=diagnostic,
                        trained_model_hash=trained_hash,
                        operator_hash=final_parameter_hash(operator),
                        parameters=candidate.parameter_record(),
                        fit_seconds=prediction_started - fit_started,
                        prediction_seconds=time.perf_counter() - prediction_started))

                held_losses, folds = [], []
                for h in range(SUPPORT_COUNT):
                    include = [k for k in range(SUPPORT_COUNT) if k != h]
                    means = losses[np.ix_(include, include)].mean(axis=1)
                    source = include[int(np.argmin(means))]
                    held_losses.append(float(losses[source, h]))
                    folds.append(dict(held_content=h, fitting_contents=include,
                                      eligible_proposal_sources=include,
                                      eligible_mean_support_mse8=means.tolist(),
                                      selected_proposal_source=source, held_mse8=held_losses[-1]))
                candidate_mean = float(np.mean(held_losses))
                default_mean = float(np.mean(clean_held[g])) if available[g] else None
                adopted = bool(available[g] and default_mean - candidate_mean > TOL)
                full_means = losses.mean(axis=1)
                full_source = int(np.argmin(full_means))
                if adopted:
                    models[g] = proposals[full_source]
                selected_held.append(held_losses if adopted else list(clean_held[g]))
                slots.append(dict(
                    slot=g, proposals=proposal_records, support_loss_matrix_mse8=losses.tolist(),
                    matrix_axes=["proposal_source_content", "evaluated_support_content"], folds=folds,
                    integer_local_held_mse8=held_losses, integer_local_mean_held_mse8=candidate_mean,
                    default_held_mse8=clean_held[g], default_mean_held_mse8=default_mean,
                    default_held_available=available[g], adopted=adopted,
                    improvement_mse8=default_mean - candidate_mean if available[g] else None,
                    status="INTEGER_LOCAL_ADOPTED" if adopted else (
                        "DEFAULT_RETAINED_NO_HELD_IMPROVEMENT" if available[g]
                        else "DEFAULT_RETAINED_INCOMPLETE_HELD_REFERENCE"),
                    full_mean_support_mse8=full_means.tolist(), full_selected_proposal_source=full_source,
                    returned_proposal_source=full_source if adopted else None,
                    default_operator_hash=final_parameter_hash(defaults[g]),
                    returned_operator_hash=final_parameter_hash(models[g]),
                    seconds=time.perf_counter() - slot_started))
                budget()
        if final_parameter_hash(defaults) != default_hash:
            raise ValueError("Default models changed during the final extension")
        record = dict(
            version="anonymous-integer-local-support-extension-v1", family="integer_local_response",
            slots=slots, default_held_losses_mse8=clean_held, default_held_available=available,
            selected_held_losses_mse8=selected_held,
            selected_held_available=[True if slot["adopted"] else available[g]
                                     for g, slot in enumerate(slots)],
            default_model_hash=default_hash, model_hash=final_parameter_hash(models),
            default_models_unchanged=True, fitting_device="cpu", fitting_threads=4,
            fitting="Eight single-content proposals for every slot; IntegerLocalResponse selects its fixed internal alternatives using only that proposal's source and training mask",
            held_selection="Held h excludes proposal source h and scoring content h during selection on the other seven; h is scored only after choosing that fold's proposal",
            proposal_ties="Exact mean-loss ties select the lowest proposal-source index",
            adoption="Require more than 1e-10 improvement in mean held MSE8; ties and incomplete default evidence retain the complete default operator",
            full_selection="Select the new proposal by mean MSE8 over all eight supports after the held comparison",
            information_boundary="Only anonymous support pairs and saved support-derived defaults/held losses; no query, labels, recipe, candidate histories, or AI calls",
            validation_scope="Held losses select the candidate family and are not independent validation; the final candidate was motivated by development diagnosis",
            seconds=time.perf_counter() - started)
        budget()
        return models, record
    finally:
        torch.set_num_threads(previous_threads)
