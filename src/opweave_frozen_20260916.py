"""Frozen OpWeave development method, selected by the final paired comparison.

The integer-local family is enabled globally. Each anonymous slot retains the
support-only family-selection rule; condition, query, history and semantic
operation names never route this configuration.
"""
from __future__ import annotations

import time

import numpy as np

from extended_mixed_estimator import ExtendedMixedEstimator, mixed_parameter_hash
from unit_gain_extension import extend_unit_gain

VERSION = "opweave-dev-20260916-final"
INCLUDE_INTEGER_LOCAL = True  # Frozen globally on 2026-09-16.


class OpWeaveEstimator:
    def __init__(self, traditional_replay, mixed_replay, candidate, prior):
        if type(INCLUDE_INTEGER_LOCAL) is not bool:
            raise RuntimeError(
                "Invalid frozen configuration: INCLUDE_INTEGER_LOCAL must be boolean. "
                "Use the recorded frozen configuration.")
        self.include_integer_local = INCLUDE_INTEGER_LOCAL
        self.traditional_replay = traditional_replay
        self.prior = prior
        self.base = ExtendedMixedEstimator(traditional_replay, mixed_replay, candidate, prior)
        if self.include_integer_local:
            from final_local_extension import extend_integer_local, final_parameter_hash
            self._integer_extension = extend_integer_local
            self._parameter_hash = final_parameter_hash
            self._hash_schema = "final_parameter_hash"
        else:
            self._integer_extension = None
            self._parameter_hash = mixed_parameter_hash
            self._hash_schema = "mixed_parameter_hash"

    def fit(self, before, after, *, check_budget=None):
        """Fit one operator list from exactly eight anonymous paired supports."""
        if check_budget is not None and not callable(check_budget):
            raise TypeError("check_budget must be callable or None")
        before_array, after_array = np.asarray(before), np.asarray(after)
        if before_array.ndim != 4 or before_array.shape[:2] != (8, 3):
            raise ValueError("Expected eight CHW RGB support inputs")
        if after_array.shape != (8, 4, *before_array.shape[1:]):
            raise ValueError("Expected four anonymous output slots for each support input")

        def budget():
            if check_budget is not None:
                check_budget()

        started = time.perf_counter()
        stage_seconds, stage_records = {}, {}
        budget()
        tick = time.perf_counter()
        methods, base_record = self.base.fit(before_array, after_array)
        stage_seconds["original_hierarchy"] = time.perf_counter() - tick
        stage_records["original_hierarchy"] = base_record
        models = methods["gain_corrected_candidates"]
        held = base_record["final_selected_held_losses_mse8"]
        available = base_record["final_selected_held_available"]
        budget()

        tick = time.perf_counter()
        models, unit_record = extend_unit_gain(
            self.traditional_replay, before_array, after_array, models, held, available,
            check_budget=check_budget)
        stage_seconds["unit_gain"] = time.perf_counter() - tick
        stage_records["unit_gain"] = unit_record
        held = unit_record["selected_held_losses_mse8"]
        available = unit_record["selected_held_available"]
        final_stage = unit_record
        budget()

        if self.include_integer_local:
            tick = time.perf_counter()
            models, local_record = self._integer_extension(
                self.traditional_replay, before_array, after_array, models, held, available,
                check_budget=check_budget)
            stage_seconds["integer_local"] = time.perf_counter() - tick
            stage_records["integer_local"] = local_record
            held = local_record["selected_held_losses_mse8"]
            available = local_record["selected_held_available"]
            final_stage = local_record
            budget()

        model_hash = self._parameter_hash(models)
        if model_hash != final_stage["model_hash"]:
            raise ValueError("Final returned model differs from its support-selection record")
        stage_seconds["total"] = time.perf_counter() - started
        record = dict(
            version=VERSION,
            configuration=dict(include_integer_local=self.include_integer_local,
                               stage_order=list(stage_records), selection_scope="single global method constant"),
            stages=stage_records, seconds=stage_seconds,
            final_selected_held_losses_mse8=held,
            final_selected_held_available=available,
            model_hash=model_hash, hash_schema=self._hash_schema,
            declared_prior=self.prior.manifest(),
            support_count=8, anonymous_slot_count=4,
            information_boundary="fit receives only anonymous before/after supports and an optional cancellation callback; the constructor receives the same declared prior and estimator components; no condition, semantic slot type, query, recipe, candidate bank, or true history routes the method",
            parameter_updates="Only support-specific operation models are fitted; global pretrained representations and AI editor weights are not updated",
            output="One final operator list and its complete support-only selection record",
            validation_scope="Held-content losses select support-specific candidates; these losses are not independent evaluation or evidence that this global family choice generalizes",
            budget_scope="Original hierarchy checked immediately before/after fit; unit-gain and optional integer-local stages receive their existing internal budget callbacks",
        )
        return models, record
