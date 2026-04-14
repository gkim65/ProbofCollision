"""
tca.py

Find Time of Closest Approach (TCA) between two spacecraft by propagating
forward from initial conditions and minimizing miss distance.

Uses a coarse grid search followed by scipy scalar minimization.

Performance note: the coarse grid uses a single pair of propagators stepped
forward sequentially (O(T) total work). The fine minimization creates fresh
propagators from the nearest coarse point so it only integrates a short arc.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from brahe import (
    Epoch,
    NumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfig,
)


def _make_propagator(epoch: Epoch, eci_state: np.ndarray,
                     two_body: bool = True) -> NumericalOrbitPropagator:
    prop_config = NumericalPropagationConfig.default()
    force_config = ForceModelConfig.two_body() if two_body else ForceModelConfig.default()
    return NumericalOrbitPropagator(epoch, eci_state, prop_config, force_config)


def _miss_distance_from(
    t_offset_s: float,
    anchor_epoch: Epoch,
    sc1_anchor: np.ndarray,
    sc2_anchor: np.ndarray,
    two_body: bool,
) -> float:
    """
    Return miss distance (m) at anchor_epoch + t_offset_s.

    Creates propagators from anchor_epoch, so only integrates the arc
    [0, t_offset_s] rather than from the original T=0.
    """
    target = anchor_epoch + t_offset_s
    prop1 = _make_propagator(anchor_epoch, sc1_anchor, two_body)
    prop2 = _make_propagator(anchor_epoch, sc2_anchor, two_body)
    prop1.propagate_to(target)
    prop2.propagate_to(target)
    r1 = np.array(prop1.current_state()[:3])
    r2 = np.array(prop2.current_state()[:3])
    return float(np.linalg.norm(r1 - r2))


def find_tca(
    epoch_start: Epoch,
    sc1_eci_t0: np.ndarray,
    sc2_eci_t0: np.ndarray,
    window_hours: float = 24.0,
    coarse_steps: int = 200,
    two_body: bool = True,
) -> tuple[Epoch, float]:
    """
    Find the Time of Closest Approach (TCA) within [epoch_start, epoch_start + window_hours].

    Strategy:
        1. Coarse grid: step a single pair of propagators forward through the
           window, recording miss distance at each node.  O(T) integrations.
        2. Fine minimization: Brent's method over ±1 coarse step around the
           best node, anchored at that node so the arc is short.

    Returns:
        epoch_tca     : Epoch of closest approach
        miss_distance : Miss distance at TCA (m)
    """
    window_s = window_hours * 3600.0
    offsets  = np.linspace(0.0, window_s, coarse_steps)

    # --- coarse grid: reuse propagators, step forward sequentially ---
    prop1 = _make_propagator(epoch_start, sc1_eci_t0, two_body)
    prop2 = _make_propagator(epoch_start, sc2_eci_t0, two_body)

    distances  = np.empty(coarse_steps)
    sc1_states = np.empty((coarse_steps, 6))
    sc2_states = np.empty((coarse_steps, 6))

    for i, t in enumerate(offsets):
        epoch = epoch_start + t
        prop1.propagate_to(epoch)
        prop2.propagate_to(epoch)
        s1 = np.array(prop1.current_state()[:6])
        s2 = np.array(prop2.current_state()[:6])
        sc1_states[i] = s1
        sc2_states[i] = s2
        distances[i] = float(np.linalg.norm(s1[:3] - s2[:3]))

    best_idx = int(np.argmin(distances))

    # --- fine minimization anchored at the best coarse node ---
    step         = window_s / (coarse_steps - 1)
    anchor_epoch = epoch_start + offsets[best_idx]
    anchor_s1    = sc1_states[best_idx]
    anchor_s2    = sc2_states[best_idx]

    # Always search ±1 coarse step around the anchor. Allowing a small amount of
    # propagation past the window edges lets the minimizer find boundary minima cleanly.
    lo = -step
    hi =  step

    result = minimize_scalar(
        _miss_distance_from,
        bounds=(lo, hi),
        method="bounded",
        args=(anchor_epoch, anchor_s1, anchor_s2, two_body),
        options={"xatol": 1.0},   # 1-second precision
    )

    tca_offset_s = float(result.x)
    epoch_tca    = anchor_epoch + tca_offset_s
    miss_dist    = float(result.fun)

    return epoch_tca, miss_dist


def get_states_at_tca(
    epoch_start: Epoch,
    sc1_eci_t0: np.ndarray,
    sc2_eci_t0: np.ndarray,
    epoch_tca: Epoch,
    two_body: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate both spacecraft to epoch_tca and return their ECI states.

    Returns:
        sc1_eci_tca : [6,] array (m, m/s)
        sc2_eci_tca : [6,] array (m, m/s)
    """
    prop1 = _make_propagator(epoch_start, sc1_eci_t0, two_body)
    prop2 = _make_propagator(epoch_start, sc2_eci_t0, two_body)
    prop1.propagate_to(epoch_tca)
    prop2.propagate_to(epoch_tca)
    return (
        np.array(prop1.current_state()[:6]),
        np.array(prop2.current_state()[:6]),
    )
