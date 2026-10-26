"""Support-only pooled -> threshold -> gain hierarchy for new anonymous slots.

The common AI prior belongs to the estimator configuration. fit receives only
eight paired before/after supports, never query images, conditions, recipes,
candidate histories, true operation names, or generator parameters.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from gain_block import GAIN_FAMILIES, GainBlockCache, GainReplay
from mixed_estimator import MixedEstimator
from threshold_block import THRESHOLD_FAMILIES, ThresholdBlockCache

TOL = 1e-10
METHODS = ("pooled_corrected", "threshold_point", "gain_corrected_candidates")


def mixed_parameter_hash(value):
    """Extend the frozen serializer only for a complete immutable InpaintSpec.

    Traditional values retain their exact old serialization/hash. The typed
    AI tuple includes every dataclass field, including original float32 mask
    bytes; it does not rely on display names, object identity, or a partial
    manifest. No editor is initialized or invoked for hashing.
    """
    from dataclasses import fields
    from ai_operator import InpaintSpec
    from support_correction_pool import parameter_hash

    def normalize(item):
        if isinstance(item, InpaintSpec):
            return (
                "__opweave_parameter_type__",
                "ai_operator.InpaintSpec:v1",
                tuple((field.name, normalize(getattr(item, field.name)))
                      for field in fields(item)),
            )
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, tuple):
            return tuple(normalize(child) for child in item)
        if isinstance(item, dict):
            return {normalize(key): normalize(child) for key, child in item.items()}
        return item

    return parameter_hash(normalize(value))


class ExtendedMixedEstimator:
    def __init__(self, traditional_replay, mixed_replay, candidate, prior):
        if not isinstance(traditional_replay, GainReplay):
            raise TypeError("ExtendedMixedEstimator requires GainReplay as the traditional replay")
        if traditional_replay.device.type != "cpu":
            raise ValueError("The frozen threshold/gain extension uses the CPU fitting environment")
        self.base, self.replay = traditional_replay, mixed_replay
        self.candidate, self.prior = candidate, prior
        self.baseline = MixedEstimator(
            "pooled_corrected", traditional_replay, mixed_replay, candidate, prior)

    @staticmethod
    def _default_held(models, record):
        corrections = record["correction"]
        if len(corrections) != len(models):
            raise ValueError("Pooled correction records do not match anonymous slots")
        values, available = [], []
        for model, correction in zip(models, corrections):
            adopted = bool(correction["adopted"])
            if adopted != (model[0] == "shared_residual"):
                raise ValueError("Pooled correction adoption disagrees with the returned model")
            key = "held_corrected_mse_8bit" if adopted else "held_baseline_mse_8bit"
            row = correction[key]
            if len(row) != 8:
                raise ValueError("Expected eight pooled held-content losses")
            clean = []
            for value in row:
                if value is None:
                    clean.append(None)
                elif not np.isfinite(float(value)) or float(value) < 0:
                    raise ValueError("Invalid pooled held-content loss")
                else:
                    clean.append(float(value))
            values.append(clean)
            available.append(all(value is not None for value in clean))
        return values, available

    @torch.no_grad()
    def _extend(self, before, after, defaults, default_held, default_available,
                cache_class, families, name):
        import chainprint_shared_residual as res

        started = time.perf_counter()
        labels = ["default"] + [f + "/raw" for f in families] + [f + "/corrected" for f in families]
        slot_count = len(defaults)
        # Null is preserved, never imputed as zero or infinite fitting evidence.
        held = [[[None] * 8 for _ in labels] for _ in range(slot_count)]
        for g in range(slot_count):
            held[g][0] = list(default_held[g])
        active = [g for g in range(slot_count) if default_available[g]]
        models = list(defaults)
        selected = [0] * slot_count
        decisions = []
        cache = None
        if active:
            cache = cache_class(self.base, before, after)
            images, targets = cache.images, cache.targets

            def corrected_operator(operator, predictions, slot, include):
                designs = [res.features(images[k], predictions[k]) for k in include]
                responses = [((targets[k][slot] - predictions[k]) * 255).reshape(3, -1).T
                             for k in include]
                coefficient = res.fit_coefficients(designs, responses, list(range(len(include))))
                return ("shared_residual", (operator, coefficient))

            for held_index in range(8):
                include = [k for k in range(8) if k != held_index]
                fold = cache.candidates(include)
                for g in active:
                    for family in families:
                        operator = fold[g][family]
                        predictions = [self.base.apply(image, operator) for image in images]
                        corrected = corrected_operator(operator, predictions, g, include)
                        coefficient = corrected[1][1]
                        adjusted = res.corrected(
                            predictions[held_index],
                            res.features(images[held_index], predictions[held_index]), coefficient)
                        for label, prediction in ((family + "/raw", predictions[held_index]),
                                                  (family + "/corrected", adjusted)):
                            loss = float(((prediction - targets[held_index][g]) * 255).square().mean())
                            if not np.isfinite(loss) or loss < 0:
                                raise ValueError("Invalid new held-content loss")
                            held[g][labels.index(label)][held_index] = loss

            # All model-family decisions precede the full-eight refit.
            for g in active:
                means = [float(np.mean(row)) for row in held[g]]
                best = 0
                for index in range(1, len(labels)):
                    if means[index] < means[best] - TOL:
                        best = index
                selected[g] = best

            if any(selected[g] != 0 for g in active):
                full = cache.candidates(list(range(8)))
                for g in active:
                    index = selected[g]
                    if index == 0:
                        continue
                    family, kind = labels[index].split("/")
                    operator = full[g][family]
                    if kind == "raw":
                        models[g] = operator
                    else:
                        predictions = [self.base.apply(image, operator) for image in images]
                        models[g] = corrected_operator(operator, predictions, g, list(range(8)))

        selected_held = []
        for g in range(slot_count):
            selected_held.append(list(held[g][selected[g]]))
            decisions.append(dict(
                slot=g, selected_label=labels[selected[g]],
                status="SELECTED_ON_EIGHT_HELD_LOSSES" if default_available[g]
                else "DEFAULT_RETAINED_INCOMPLETE_HELD_REFERENCE",
                default_held_available=bool(default_available[g]),
                mean_held_mse8={label: float(np.mean(row)) if all(v is not None for v in row) else None
                                for label, row in zip(labels, held[g])},
            ))
        available = [all(v is not None for v in row) for row in selected_held]
        record = dict(
            method=name, families=list(families), labels=labels,
            default_held_losses_mse8=default_held, held_losses_mse8=held,
            selected_held_losses_mse8=selected_held,
            selected_held_available=available, decisions=decisions,
            cache=cache.report if cache is not None else None,
            default_model_hash=mixed_parameter_hash(defaults), model_hash=mixed_parameter_hash(models),
            seconds=time.perf_counter() - started,
            fitting="Every new fold refits q, b, optional gain, offset, and residual from exactly seven contents; selected alternatives refit on all eight",
            selection="Equal-content held MSE; default, then raw, then corrected; ties within 1e-10 prefer earlier",
            incomplete_reference="Keep that slot's default unchanged through later stages; no zero/inf imputation or partial-content mean",
            validation_scope="Saved held losses are used for model selection; they are not independent validation estimates",
        )
        return models, selected_held, available, record

    def fit(self, before, after):
        started = time.perf_counter()
        if len(before) != 8 or len(after) != 8:
            raise ValueError("The hierarchy requires eight paired anonymous supports")
        # Preserve the baseline's original support and AI availability semantics.
        pooled_start = time.perf_counter()
        pooled, pooled_record = self.baseline.fit(before, after)
        pooled_seconds = time.perf_counter() - pooled_start
        held, available = self._default_held(pooled, pooled_record)
        # This conversion is only for the established caches' shape interface;
        # their replay.image conversion preserves the original float64 convention.
        before_array, after_array = np.asarray(before), np.asarray(after)
        threshold, threshold_held, threshold_available, threshold_record = self._extend(
            before_array, after_array, pooled, held, available,
            ThresholdBlockCache, THRESHOLD_FAMILIES, "threshold_point")
        gain, gain_held, gain_available, gain_record = self._extend(
            before_array, after_array, threshold, threshold_held, threshold_available,
            GainBlockCache, GAIN_FAMILIES, "gain_corrected_candidates")
        methods = dict(pooled_corrected=pooled, threshold_point=threshold,
                       gain_corrected_candidates=gain)
        record = dict(
            version="anonymous-mixed-support-hierarchy-v1", methods=list(METHODS),
            baseline=pooled_record, baseline_selected_held_losses_mse8=held,
            baseline_selected_held_available=available,
            threshold=threshold_record, gain=gain_record,
            final_selected_held_losses_mse8=gain_held,
            final_selected_held_available=gain_available,
            model_hashes={name: mixed_parameter_hash(model) for name, model in methods.items()},
            seconds=dict(pooled_corrected=pooled_seconds, threshold_extension=threshold_record["seconds"],
                         gain_extension=gain_record["seconds"], total=time.perf_counter() - started),
            prior=self.prior.manifest(),
            fitting_device="cpu", support_count=8,
            information_boundary="fit accepts only anonymous before/after support pairs; constructor receives the same declared AI prior at every slot; no query, condition, recipe, audit labels, or truth history",
            ai_reuse="Only the original pooled mixed fitting invokes AI candidates; threshold and gain alternatives use CPU block replay and reuse the saved support-derived default losses",
            comparison_scope="Same anonymous supports, common prior, family grids, correction capacity and frozen hierarchy; compare all three returned models on unchanged later inputs",
        )
        return methods, record
