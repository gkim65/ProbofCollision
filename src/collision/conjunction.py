"""
conjunction.py

Generate synthetic conjunction scenarios for testing Pc methods.
Ported from genConjunctions.jl logic.

Approach:
  1. Define SC1 ECI state at TCA via orbital elements
  2. Place SC2 relative to SC1 at TCA in RTN frame
  3. Propagate both backwards TCA_HOURS to get T=0 initial conditions
"""

import numpy as np
import brahe
from brahe import (
    Epoch, AngleFormat, R_EARTH,
    state_koe_to_eci, state_rtn_to_eci, state_eci_to_rtn,
    NumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfig,
)

# ---------------------------------------------------------------------------
# Default scenario parameters
# ---------------------------------------------------------------------------

TCA_HOURS = 24.0          # planning horizon before TCA (hours)
R_ALT     = 550e3         # orbit altitude (m)
INCL      = 55.0          # inclination (deg)
R_MAG     = 500.0         # miss distance at TCA (m)
V_MAG     = 15.0          # relative speed at TCA (m/s)

SC1_OE_AT_TCA = np.array([
    R_EARTH + R_ALT,  # semi-major axis (m)
    0.001,            # eccentricity
    INCL,             # inclination (deg)
    20.0,             # RAAN (deg)
    0.0,              # argument of perigee (deg)
    0.0,              # mean anomaly at TCA (deg)
])

EPOCH_TCA = Epoch(2025, 6, 2, 0, 0, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_propagator(epoch: Epoch, eci_state: np.ndarray,
                     two_body: bool = True) -> NumericalOrbitPropagator:
    prop_config = NumericalPropagationConfig.default()
    force_config = ForceModelConfig.two_body() if two_body else ForceModelConfig.default()
    return NumericalOrbitPropagator(epoch, eci_state, prop_config, force_config)


def _sc2_rtn_at_tca(r_mag: float, v_mag: float,
                    conjunction_type: str, seed: int) -> np.ndarray:
    """
    Generate SC2 relative state (RTN) at TCA.

    conjunction_type:
        "head-on"   — SC2 approaches from behind along-track (T-)
        "overtaking"— SC2 approaches from ahead along-track (T+)
        "crossing"  — SC2 approaches from out-of-plane (N+)
        "random"    — random direction
    """
    rng = np.random.default_rng(seed)
    noise = r_mag * 0.01

    if conjunction_type == "head-on":
        r_rel = np.array([rng.standard_normal() * noise,
                          -r_mag,
                          rng.standard_normal() * noise])
    elif conjunction_type == "overtaking":
        r_rel = np.array([rng.standard_normal() * noise,
                          r_mag,
                          rng.standard_normal() * noise])
    elif conjunction_type == "crossing":
        r_rel = np.array([rng.standard_normal() * noise,
                          rng.standard_normal() * noise,
                          r_mag])
    else:  # random
        r_rel = r_mag * (2 * rng.random(3) - 1)
        r_rel = r_rel / np.linalg.norm(r_rel) * r_mag

    # Relative velocity perpendicular to r_rel
    tmp = rng.standard_normal(3)
    v_rel = tmp - (np.dot(tmp, r_rel) / np.dot(r_rel, r_rel)) * r_rel
    v_rel = v_rel / np.linalg.norm(v_rel) * v_mag

    return np.concatenate([r_rel, v_rel])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_conjunction(
    tca_hours: float = TCA_HOURS,
    r_mag: float = R_MAG,
    v_mag: float = V_MAG,
    conjunction_type: str = "crossing",
    seed: int = 42,
    two_body: bool = True,
    sc1_oe: np.ndarray = SC1_OE_AT_TCA,
    epoch_tca: Epoch = EPOCH_TCA,
) -> dict:
    """
    Generate initial conditions for a conjunction scenario.

    Returns a dict with keys:
        epoch_start   : Epoch at T=0 (tca_hours before TCA)
        epoch_tca     : Epoch at TCA
        sc1_eci_t0    : SC1 ECI state at T=0  [6,] m, m/s
        sc2_eci_t0    : SC2 ECI state at T=0  [6,] m, m/s
        sc1_eci_tca   : SC1 ECI state at TCA  [6,] m, m/s
        sc2_eci_tca   : SC2 ECI state at TCA  [6,] m, m/s
        miss_distance : scalar miss distance at TCA (m)
        rel_speed     : scalar relative speed at TCA (m/s)
    """
    sc1_eci_tca = np.array(state_koe_to_eci(sc1_oe, AngleFormat.DEGREES))

    sc2_rtn = _sc2_rtn_at_tca(r_mag, v_mag, conjunction_type, seed)
    sc2_eci_tca = np.array(state_rtn_to_eci(sc1_eci_tca, sc2_rtn))

    # Propagate backwards to T=0
    epoch_start = epoch_tca - tca_hours * 3600.0
    prop1 = _make_propagator(epoch_tca, sc1_eci_tca, two_body)
    prop2 = _make_propagator(epoch_tca, sc2_eci_tca, two_body)
    prop1.propagate_to(epoch_start)
    prop2.propagate_to(epoch_start)

    sc1_eci_t0 = np.array(prop1.current_state()[:6])
    sc2_eci_t0 = np.array(prop2.current_state()[:6])

    rtn_tca = np.array(state_eci_to_rtn(sc1_eci_tca, sc2_eci_tca))

    return dict(
        epoch_start=epoch_start,
        epoch_tca=epoch_tca,
        sc1_eci_t0=sc1_eci_t0,
        sc2_eci_t0=sc2_eci_t0,
        sc1_eci_tca=sc1_eci_tca,
        sc2_eci_tca=sc2_eci_tca,
        miss_distance=float(np.linalg.norm(rtn_tca[:3])),
        rel_speed=float(np.linalg.norm(rtn_tca[3:])),
    )


def sample_rtn_trajectory(
    scenario: dict,
    n_samples: int = 49,
    two_body: bool = True,
) -> np.ndarray:
    """
    Propagate both spacecraft from T=0 to TCA and sample RTN relative state.

    Returns array of shape (n_samples, 7):
        [t_hours, dR_km, dT_km, dN_km, dVR_kms, dVT_kms, dVN_kms]
    """
    epoch_start = scenario["epoch_start"]
    epoch_tca   = scenario["epoch_tca"]
    tca_hours   = (epoch_tca - epoch_start) / 3600.0

    prop1 = _make_propagator(epoch_start, scenario["sc1_eci_t0"], two_body)
    prop2 = _make_propagator(epoch_start, scenario["sc2_eci_t0"], two_body)

    rows = []
    for i in range(n_samples):
        t_hours = tca_hours * i / (n_samples - 1)
        t = epoch_start + t_hours * 3600.0
        prop1.propagate_to(t)
        prop2.propagate_to(t)
        s1  = np.array(prop1.current_state()[:6])
        s2  = np.array(prop2.current_state()[:6])
        rtn = np.array(state_eci_to_rtn(s1, s2))
        rows.append([t_hours] + list(rtn / 1e3))  # m -> km, m/s -> km/s

    return np.array(rows)
