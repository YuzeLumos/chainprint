"""One support-only unit-gain filtering family, with content-excluded selection.

Every anonymous slot receives the same eight independent single-support fits.
No default model supplies parameters to the new proposals. The defaults and
their previously computed held losses are used only for the final comparison.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from extended_mixed_estimator import mixed_parameter_hash
from history_recovery.chainprint_quantized_separable import QuantizedSeparableResponse

TOL = 1e-10
SUPPORT_COUNT = 8
SLOT_COUNT = 4


def extend_unit_gain(base, before, after, defaults, default_held, default_available, *, check_budget=None):
    """Return models and a JSON-ready record; receive anonymous supports only.

    before: [8, 3, H, W]; after: [8, 4, 3, H, W]. default_held is
    [4, 8] with missing values represented by None. Missing default evidence
    retains that complete default operator without imputing any held loss.
    check_budget is an optional cancellation callback, never fitting evidence.
    """
    if base.device.type != "cpu":
        raise ValueError("The fixed unit-gain extension requires CPU replay")
    before_shape, after_shape = tuple(before.shape), tuple(after.shape)
    if len(before_shape) != 4 or before_shape[:2] != (SUPPORT_COUNT, 3):
        raise ValueError("Expected eight CHW RGB supports")
    if after_shape != (SUPPORT_COUNT, SLOT_COUNT, *before_shape[1:]):
        raise ValueError("Expected four anonymous output slots per support")
    if len(defaults) != SLOT_COUNT or len(default_held) != SLOT_COUNT or len(default_available) != SLOT_COUNT:
        raise ValueError("Defaults must describe exactly four anonymous slots")
    clean_held, available = [], []
    for row, declared in zip(default_held, default_available):
        if len(row) != SUPPORT_COUNT:
            raise ValueError("Expected eight default held-content losses per slot")
        clean = []
        for value in row:
            if value is None:
                clean.append(None)
            elif not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError("Invalid default held-content loss")
            else:
                clean.append(float(value))
        clean_held.append(clean)
        available.append(bool(declared) and all(value is not None for value in clean))

    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    default_hash = mixed_parameter_hash(defaults)
    models = list(defaults)
    slots = []
    selected_held = []
    try:
        # fit_support creates trainable tensors before its own enable_grad block;
        # disable any caller inference context for their complete lifetime.
        with torch.inference_mode(False):
            with torch.no_grad():
                images = [base.image(before[k]) for k in range(SUPPORT_COUNT)]
                targets = [[base.image(after[k, g]) for g in range(SLOT_COUNT)]
                           for k in range(SUPPORT_COUNT)]
                if any(image.device.type != "cpu" or image.dtype != torch.float64 for image in images):
                    raise ValueError("Expected the existing CPU float64 replay input convention")
                if not all(bool(torch.isfinite(value).all()) for value in images + [x for row in targets for x in row]):
                    raise ValueError("Support arrays contain non-finite values")
                height, width = before_shape[-2:]
                mask = ((torch.arange(height)[:, None] + torch.arange(width)[None, :]) % 4 != 0).expand_as(images[0])

            for g in range(SLOT_COUNT):
                slot_started = time.perf_counter()
                proposals, proposal_records = [], []
                losses = np.empty((SUPPORT_COUNT, SUPPORT_COUNT), dtype=np.float64)
                for k in range(SUPPORT_COUNT):
                    if check_budget is not None:
                        check_budget()
                    fit_started = time.perf_counter()
                    with torch.enable_grad():
                        trained, train_loss = QuantizedSeparableResponse.fit_support(
                            before[k], after[k, g], mask, torch.device("cpu"))
                    if check_budget is not None:
                        check_budget()
                    trained_hash = mixed_parameter_hash(trained)
                    candidate = copy.deepcopy(trained)
                    with torch.no_grad():
                        candidate.gain.fill_(1)
                    # Freeze only the clone; neither defaults nor the original
                    # independently trained model are edited in place.
                    candidate.eval().requires_grad_(False)
                    for name, value in candidate.state_dict().items():
                        if name != "gain" and not torch.equal(value, trained.state_dict()[name]):
                            raise ValueError("Unit-gain construction changed a non-gain parameter")
                    if mixed_parameter_hash(trained) != trained_hash:
                        raise ValueError("Original single-support trained model changed")
                    if not np.isfinite(train_loss):
                        raise ValueError("Non-finite terminal training loss")
                    operator = ("spatial", {"quantized_separable_model": candidate})
                    proposals.append(operator)
                    prediction_started = time.perf_counter()
                    with torch.no_grad():
                        for j, image in enumerate(images):
                            prediction = base.apply(image, operator)
                            loss = float(((prediction - targets[j][g]) * 255).square().mean())
                            if not np.isfinite(loss) or loss < 0:
                                raise ValueError("Invalid unit-gain support prediction loss")
                            losses[k, j] = loss
                    proposal_records.append(dict(
                        proposal_index=k, source_contents=[k],
                        trained_model_hash=trained_hash,
                        unit_gain_operator_hash=mixed_parameter_hash(operator),
                        trained_gain=float(trained.gain.detach()), unit_gain=1.0,
                        terminal_training_loss_mse8=float(train_loss),
                        fit_seconds=prediction_started - fit_started,
                        prediction_seconds=time.perf_counter() - prediction_started,
                    ))

                held_losses, folds = [], []
                for h in range(SUPPORT_COUNT):
                    include = [k for k in range(SUPPORT_COUNT) if k != h]
                    # Row k depends only on content k. Exclude k=h even though
                    # its precomputed predictions are present in this matrix.
                    means = losses[np.ix_(include, include)].mean(axis=1)
                    source = include[int(np.argmin(means))]
                    held_losses.append(float(losses[source, h]))
                    folds.append(dict(
                        held_content=h, fitting_contents=include,
                        eligible_proposal_sources=include,
                        eligible_mean_support_mse8=means.tolist(),
                        selected_proposal_source=source,
                        held_mse8=held_losses[-1],
                    ))
                unit_mean = float(np.mean(held_losses))
                default_mean = float(np.mean(clean_held[g])) if available[g] else None
                adopted = available[g] and default_mean - unit_mean > TOL
                # This full-support choice is made after the held comparison.
                full_means = losses.mean(axis=1)
                full_source = int(np.argmin(full_means))
                if adopted:
                    models[g] = proposals[full_source]
                selected_held.append(held_losses if adopted else list(clean_held[g]))
                slots.append(dict(
                    slot=g, proposals=proposal_records,
                    support_loss_matrix_mse8=losses.tolist(),
                    matrix_axes=["proposal_source_content", "evaluated_support_content"],
                    folds=folds, unit_gain_held_mse8=held_losses,
                    unit_gain_mean_held_mse8=unit_mean,
                    default_held_mse8=clean_held[g], default_mean_held_mse8=default_mean,
                    default_held_available=available[g], adopted=bool(adopted),
                    improvement_mse8=default_mean - unit_mean if available[g] else None,
                    status="UNIT_GAIN_ADOPTED" if adopted else (
                        "DEFAULT_RETAINED_NO_HELD_IMPROVEMENT" if available[g]
                        else "DEFAULT_RETAINED_INCOMPLETE_HELD_REFERENCE"),
                    full_mean_support_mse8=full_means.tolist(),
                    full_selected_proposal_source=full_source,
                    returned_proposal_source=full_source if adopted else None,
                    default_operator_hash=mixed_parameter_hash(defaults[g]),
                    returned_operator_hash=mixed_parameter_hash(models[g]),
                    seconds=time.perf_counter() - slot_started,
                ))
        if mixed_parameter_hash(defaults) != default_hash:
            raise ValueError("Default models changed during extension")
        record = dict(
            version="anonymous-unit-gain-support-extension-v1",
            family="quantized_separable_unit_gain", slots=slots,
            default_held_losses_mse8=clean_held,
            default_held_available=available,
            selected_held_losses_mse8=selected_held,
            selected_held_available=[True if slot["adopted"] else available[g]
                                     for g, slot in enumerate(slots)],
            default_model_hash=default_hash, model_hash=mixed_parameter_hash(models),
            default_models_unchanged=True, fitting_device="cpu", fitting_threads=4,
            fitting="Eight independent one-content fits per anonymous slot; unchanged repository fit_support, 128 Adam steps and (row+column)%4!=0 mask; clone then replace only gain by 1",
            held_selection="For held h, exclude proposal source h; select by mean error on the other seven contents, then score once on h",
            proposal_ties="Exact mean-loss ties choose the lowest support source index via argmin",
            adoption="Replace the complete default slot only when its mean held MSE8 improves by more than 1e-10; incomplete default evidence retains the default",
            full_selection="After the held comparison, select the unit-gain proposal by mean MSE8 over all eight contents",
            information_boundary="Receives only anonymous paired supports, support-derived defaults and their held-loss records; same candidate family at every slot; no query, condition, recipe, true type, candidate history, or AI call",
            validation_scope="Held losses select the family and are not an independent performance estimate; this candidate family was motivated by post-evaluation development diagnosis",
            seconds=time.perf_counter() - started,
        )
        return models, record
    finally:
        torch.set_num_threads(previous_threads)
