"""Support-constrained response bounds, independent of the production estimator.

All radii are declared deterministic constraints, not fitted confidence levels.
The linear bound concerns an affine target on that parameter set. Applying it
to a nonlinear/quantized history requires separately valid remainder bounds.
"""
from __future__ import annotations

import math
import numpy as np


def linear_response_bound(gram, direction, epsilon, parameter_radius=math.inf,
                          eigen_rtol=1e-12):
    """Maximise g.T @ delta under delta.T @ G @ delta <= epsilon**2,
    ||delta|| <= parameter_radius. Return an original-set feasible witness and a
    dual upper bound. Small eigenvalues are removed for solving, enlarging the
    set; the original-set and relaxed-set gaps are reported separately.

    A symmetric eigendecomposition followed by a scalar bisection solves the
    two-active-constraint case. No empirical noise calibration is performed.
    Values are floating-point diagnostics, not interval-arithmetic certificates.
    """
    G = np.asarray(gram, dtype=np.float64)
    g = np.asarray(direction, dtype=np.float64)
    if G.ndim != 2 or G.shape != (g.size, g.size) or g.ndim != 1 or not g.size:
        raise ValueError("Expected square Gram and matching nonempty direction")
    if not np.isfinite(G).all() or not np.isfinite(g).all():
        raise ValueError("Nonfinite Gram or direction")
    eps, radius = float(epsilon), float(parameter_radius)
    if not math.isfinite(eps) or eps < 0 or math.isnan(radius) or radius < 0:
        raise ValueError("Nonnegative finite response radius and valid parameter radius required")
    if not 0 <= eigen_rtol < 1:
        raise ValueError("Invalid eigenvalue tolerance")
    values, vectors = np.linalg.eigh((G + G.T) / 2)
    scale = max(float(np.abs(values).max()), np.finfo(float).tiny)
    threshold = eigen_rtol * scale
    if values.min() < -max(threshold, 32*np.finfo(float).eps*scale):
        raise ValueError("Gram is not positive semidefinite")
    dropped = (values > 0) & (values <= threshold)
    lam = np.where(values > threshold, values, 0.0)
    positive = lam > 0
    z = vectors.T @ g
    null_norm = float(np.linalg.norm(z[~positive]))
    info = dict(rank=int(positive.sum()), dimension=len(g),
                eigenvalue_threshold=threshold,
                discarded_positive_eigenvalues=int(dropped.sum()),
                direction_norm=float(np.linalg.norm(g)),
                null_direction_norm=null_norm,
                epsilon=eps, parameter_radius=radius,
                interpretation="Declared-set affine response; no statistical coverage")

    def result(upper, w, route, alpha=None, beta=None):
        witness = None if w is None else vectors @ w
        relaxed_primal = None if w is None else float(z @ w)
        if witness is not None:
            original_response = math.sqrt(max(0., float(witness @ ((G+G.T)/2) @ witness)))
            if original_response > eps:
                witness *= eps/original_response
            primal = float(g @ witness)
            original_response = math.sqrt(max(0., float(witness @ ((G+G.T)/2) @ witness)))
        else:
            primal, original_response = None, None
        relaxed_gap = None if relaxed_primal is None else max(0., float(upper)-relaxed_primal)
        tolerance = 1e-9*max(abs(float(upper)), np.finfo(float).tiny)
        return {**info, "upper_bound": float(upper), "witness": witness,
                "witness_value": primal, "route": route,
                "dual_alpha": alpha, "dual_beta": beta,
                "duality_gap": None if primal is None else max(0.0, float(upper)-primal),
                "relaxed_duality_gap": relaxed_gap,
                "relaxed_solver_converged": None if relaxed_gap is None else relaxed_gap <= tolerance,
                "bound_scope": "eigenvalue-relaxed Gram" if dropped.any() else "input Gram within floating-point tolerance",
                "witness_response_norm": original_response,
                "witness_parameter_norm": None if witness is None else float(np.linalg.norm(witness))}

    norm = float(np.linalg.norm(z))
    if norm == 0 or radius == 0:
        return result(0.0, np.zeros_like(z), "zero")
    if eps == 0:
        if null_norm == 0:
            return result(0.0, np.zeros_like(z), "exact_support")
        if math.isinf(radius):
            return result(math.inf, None, "unbounded_null_direction")
        w = np.where(positive, 0.0, z) * (radius/null_norm)
        return result(radius*null_norm, w, "nullspace_ball")
    if math.isinf(radius):
        # Do not suppress even a tiny nonzero null component and silently claim
        # a finite bound without an explicit parameter-neighborhood assumption.
        if null_norm > 0:
            return result(math.inf, None, "unbounded_null_direction")
        inverse = np.zeros_like(z)
        inverse[positive] = z[positive]/lam[positive]
        q = float(z @ inverse)
        w = eps*inverse/math.sqrt(q)
        return result(eps*math.sqrt(q), w, "support_only",
                      math.sqrt(q)/(2*eps), 0.0)

    ball = radius*z/norm
    if float(np.sum(lam*ball*ball)) <= eps*eps:
        return result(radius*norm, ball, "parameter_ball", 0.0, norm/(2*radius))

    if null_norm == 0:
        inverse = np.zeros_like(z)
        inverse[positive] = z[positive]/lam[positive]
        q = float(z @ inverse)
        support = eps*inverse/math.sqrt(q)
        if np.linalg.norm(support) <= radius:
            return result(eps*math.sqrt(q), support, "support_only",
                          math.sqrt(q)/(2*eps), 0.0)

    # Both constraints active. For t=beta/alpha, a_i=g_i/(lambda_i+t).
    # The Rayleigh ratio sum(lambda*a^2)/sum(a^2) increases with t.
    target = (eps/radius)**2

    def rayleigh(t):
        a = z/(lam+t)
        a /= max(float(np.linalg.norm(a)), np.finfo(float).tiny)
        return float(np.sum(lam*a*a))

    lo, hi = 0.0, float(max(lam.max(), np.finfo(float).tiny))
    for _ in range(1024):
        if rayleigh(hi) >= target:
            break
        hi *= 2
        if not math.isfinite(hi):
            raise ArithmeticError("Unable to bracket active constraints")
    else:
        raise ArithmeticError("Unable to bracket active constraints")
    for _ in range(160):
        mid = (lo+hi)/2
        if mid == lo or mid == hi:
            break
        if rayleigh(mid) <= target:
            lo = mid
        else:
            hi = mid
    # Upper endpoint supplies positive dual multipliers regardless of whether
    # the corresponding witness is exactly feasible in floating point.
    t = hi
    a = z/(lam+t)
    alpha = float(np.linalg.norm(a)/(2*radius))
    beta = float(t*alpha)
    dual = alpha*eps*eps + beta*radius*radius + float(np.sum(z*z/(alpha*lam+beta)))/4
    w = radius*a/np.linalg.norm(a)
    response = math.sqrt(float(np.sum(lam*w*w)))
    if response > eps:
        w *= eps/response
    return result(np.nextafter(dual, math.inf), w, "both_constraints", alpha, beta)


def shared_occurrence_bound(gram, occurrence_directions, epsilon,
                            parameter_radius=math.inf):
    """Same uncertainty set, shared across all occurrences in a comparison.

    Summing occurrence contributions before taking a support function is exact
    for the shared affine target. Summing their individual bounds is a relaxed
    per-occurrence bound, and can be strictly larger.
    """
    directions = np.asarray(occurrence_directions, dtype=np.float64)
    if directions.ndim != 2 or not len(directions):
        raise ValueError("At least one occurrence direction required")
    shared = linear_response_bound(gram, directions.sum(axis=0),
                                   epsilon, parameter_radius)
    separate = [linear_response_bound(gram, d, epsilon, parameter_radius)
                for d in directions]
    return dict(shared=shared,
                independent_upper_bound=float(sum(r["upper_bound"] for r in separate)),
                occurrence_count=len(directions))


def finite_pool_comparison(losses, reference, eligible=None,
                           endpoint_tolerance_squared=None, tie_atol=1e-12):
    """Evaluate coherent model tuples on all candidates without smoothness.

    losses[m,h] must use the SAME per-operation models at every occurrence and
    across all h in row m. This semantic requirement is the caller's contract.
    Eligibility must be decided from operation examples, before query scoring.
    The sampled/finite pool is not a confidence region over arbitrary operators.
    """
    panel = np.asarray(losses, dtype=np.float64)
    if panel.ndim != 2 or not panel.shape[0] or panel.shape[1] < 2:
        raise ValueError("Need a nonempty model-by-candidate loss panel")
    if not np.isfinite(panel).all() or (panel < 0).any():
        raise ValueError("Losses must be finite and nonnegative")
    if not 0 <= reference < panel.shape[1] or tie_atol < 0:
        raise ValueError("Invalid reference or tie tolerance")
    model_rows = np.arange(len(panel))
    if eligible is not None:
        mask = np.asarray(eligible, dtype=bool)
        if mask.shape != (panel.shape[0],):
            raise ValueError("Eligibility does not align with model rows")
        panel = panel[mask]
        model_rows = model_rows[mask]
    if len(panel) == 0:
        raise ValueError("Empty supported pool cannot certify any candidate")
    differences = panel-panel[:, [reference]]
    lower = differences.min(axis=0)
    rivals = np.arange(panel.shape[1]) != reference
    result = dict(reference=int(reference), model_count=len(panel),
                  minimum_paired_margins=lower.tolist(),
                  stable_in_evaluated_pool=bool(np.all(lower[rivals] > tie_atol)),
                  counterexample_model_rows=[
                      int(model_rows[m]) for m in np.flatnonzero(np.any(differences[:, rivals] < -tie_atol, axis=1))
                  ],
                  scope="Only evaluated coherent tuples; no coverage claim")
    if endpoint_tolerance_squared is not None:
        tolerance = float(endpoint_tolerance_squared)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Invalid endpoint tolerance")
        feasible = np.flatnonzero(panel.min(axis=0) <= tolerance + tie_atol)
        result["endpoint_feasible_candidates"] = feasible.tolist()
        result["unique_feasible_in_evaluated_pool"] = (
            int(feasible[0]) if len(feasible) == 1 else None)
    return result


def quantized_step_stability(prequantization_8bit, absolute_change_bound_8bit):
    """Sufficient exact-output condition for round(clip(u, 0, 255)).

    Bounds must hold over the ENTIRE declared parameter set on a fixed input.
    Apply on every nominal visited state and use induction to certify a path;
    sampled perturbations or gradients do not establish the required bounds.
    Strict threshold distances safely handle round-to-even half-integer ties.
    """
    u = np.asarray(prequantization_8bit, dtype=np.float64)
    b = np.broadcast_to(np.asarray(absolute_change_bound_8bit, dtype=np.float64), u.shape)
    if not u.size or not np.isfinite(u).all() or np.isnan(b).any() or (b < 0).any():
        raise ValueError("Finite nonempty values and nonnegative bounds required")
    output = np.rint(np.clip(u, 0., 255.))
    lower = np.where(output == 0., -np.inf, output-.5)
    upper = np.where(output == 255., np.inf, output+.5)
    margin = np.maximum(0., np.minimum(u-lower, upper-u))
    safe = (b == 0.) | (b < margin)
    return dict(all_outputs_unchanged=bool(np.all(safe)),
                pixel_values_checked=int(u.size),
                uncertified_pixel_values=int(np.count_nonzero(~safe)),
                minimum_threshold_margin_8bit=float(margin.min()),
                minimum_bound_slack_8bit=float(np.min(margin-b)),
                scope="Conditional on valid uniform input-fixed bounds; not estimated from sampling")
