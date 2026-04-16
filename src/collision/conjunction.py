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
        "head-on"    — SC2 offset along T-; velocity mostly anti-parallel to T
                       (opposite-direction orbits, high v_rel ~500+ m/s).
                       Results in large di and high closing speed.

        "overtaking" — SC2 offset along T+; velocity mostly along T-
                       (same orbital plane, SC2 slightly ahead, being caught).
                       Results in tiny da, low v_rel, pure along-track pass.

        "crossing"   — SC2 offset along N; velocity mostly along N
                       (two orbits in different planes whose ground tracks
                       cross — the most common real-world conjunction type).
                       The N-dominant v_rel produces large di and high v_rel.

        "radial"     — SC2 offset along R (radial); velocity mostly along T
                       (different altitudes, same orbital plane — like a
                       decaying object passing through a lower satellite's
                       altitude). Results in small di, moderate v_rel in T.

        "random"     — random offset direction, random perpendicular velocity
    """
    rng = np.random.default_rng(seed)
    noise = r_mag * 0.01

    if conjunction_type == "head-on":
        # Miss offset: SC2 is along-track behind SC1 (T-)
        # Small noise in R and N so miss vector isn't perfectly degenerate
        r_rel = np.array([rng.standard_normal() * noise,
                          -r_mag,
                          rng.standard_normal() * noise])
        # Velocity: approaching head-on means v_rel is strongly anti-T.
        # Allow small R/N components via noise so v_rel ⊥ r_rel is satisfied.
        v_nominal = np.array([rng.standard_normal() * noise,
                               0.0,
                               rng.standard_normal() * noise])
        # Project out the r_rel component to enforce perpendicularity exactly
        v_rel = v_nominal - (np.dot(v_nominal, r_rel) / np.dot(r_rel, r_rel)) * r_rel

    elif conjunction_type == "overtaking":
        # Miss offset: SC2 is along-track ahead of SC1 (T+)
        r_rel = np.array([rng.standard_normal() * noise,
                          r_mag,
                          rng.standard_normal() * noise])
        # Velocity: SC2 is being caught, so v_rel is along T- (SC1 catching up).
        # Pure -T then project out r_rel component (which is mostly T).
        v_nominal = np.array([0.0, -v_mag, 0.0])
        v_rel = v_nominal - (np.dot(v_nominal, r_rel) / np.dot(r_rel, r_rel)) * r_rel

    elif conjunction_type == "crossing":
        # True crossing conjunction: two satellites in different orbital planes
        # whose orbit planes intersect. The key signature is:
        #   - v_rel is dominated by the N (out-of-plane) component → large di
        #   - r_rel (miss vector) must be ⊥ to v_rel, so it lies in R-T plane
        #     (since v_rel ≈ N, anything ⊥ to N lives in R-T)
        #
        # So: place the miss offset in T (along-track), velocity in -N.
        # This is correct: at TCA the two spacecraft are separated along-track
        # while flying through each other's planes in the N direction.
        r_rel = np.array([rng.standard_normal() * noise,
                          r_mag,
                          rng.standard_normal() * noise])
        # v_rel purely along -N (approaching from +N side), then project to
        # enforce exact perpendicularity with r_rel.
        v_nominal = np.array([0.0, 0.0, -v_mag])
        v_rel = v_nominal - (np.dot(v_nominal, r_rel) / np.dot(r_rel, r_rel)) * r_rel

    elif conjunction_type == "radial":
        # Miss offset: SC2 is radially offset (R direction)
        # Represents objects in same orbital plane but different altitudes
        # whose orbits bring them to the same point.
        r_rel = np.array([r_mag,
                          rng.standard_normal() * noise,
                          rng.standard_normal() * noise])
        # Velocity: along-track (T) dominant — the relative motion is mostly
        # due to the altitude difference causing different orbital speeds.
        v_nominal = np.array([rng.standard_normal() * noise * 0.1,
                               v_mag,
                               rng.standard_normal() * noise * 0.1])
        v_rel = v_nominal - (np.dot(v_nominal, r_rel) / np.dot(r_rel, r_rel)) * r_rel

    else:  # random
        r_rel = r_mag * (2 * rng.random(3) - 1)
        r_rel = r_rel / np.linalg.norm(r_rel) * r_mag
        tmp = rng.standard_normal(3)
        v_rel = tmp - (np.dot(tmp, r_rel) / np.dot(r_rel, r_rel)) * r_rel

    # Normalise to exact v_mag (perpendicularity already enforced above)
    v_rel = v_rel / np.linalg.norm(v_rel) * v_mag

    return np.concatenate([r_rel, v_rel])


# ---------------------------------------------------------------------------
# Head-on: build SC2 directly in ECI from a retrograde orbit
# ---------------------------------------------------------------------------

def _sc2_eci_head_on(sc1_eci_tca: np.ndarray, r_mag: float, seed: int) -> np.ndarray:
    """
    Generate SC2 ECI state for a true head-on conjunction.

    Strategy: put SC2 at the same ECI position as SC1 at TCA, but with a
    retrograde velocity (reflected through the orbital plane normal). Then
    offset by r_mag along SC1's along-track (T) direction to get the miss
    distance.

    Result:
      - SC2 velocity ≈ anti-parallel to SC1 velocity
      - Relative speed ≈ 2 × orbital speed ≈ 14-15 km/s
      - Miss distance ≈ r_mag
    """
    rng = np.random.default_rng(seed)

    r1 = sc1_eci_tca[:3]   # position (m)
    v1 = sc1_eci_tca[3:]   # velocity (m/s)

    # --- Build SC1's RTN unit vectors at TCA ---
    r_hat = r1 / np.linalg.norm(r1)
    h = np.cross(r1, v1)           # angular momentum vector
    n_hat = h / np.linalg.norm(h)  # orbit normal (N)
    t_hat = np.cross(n_hat, r_hat) # along-track (T)

    # --- SC2 position: SC1 position + r_mag offset along T ---
    # Small noise in N so it's not perfectly degenerate
    noise_n = rng.standard_normal() * r_mag * 0.01
    r2 = r1 + r_mag * t_hat + noise_n * n_hat

    # --- SC2 velocity: retrograde — flip v1 through the orbit plane ---
    # Retrograde means reversing the T component while keeping R and N.
    # Decompose v1 into R, T, N components:
    v1_r = np.dot(v1, r_hat) * r_hat
    v1_t = np.dot(v1, t_hat) * t_hat
    v1_n = np.dot(v1, n_hat) * n_hat
    # Retrograde: negate T (along-track) and keep R and N
    # This gives a satellite going the opposite way around Earth
    v2 = v1_r - v1_t + v1_n

    return np.concatenate([r2, v2])


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

    if conjunction_type == "head-on":
        # True head-on: SC2 in retrograde orbit, built directly from OEs
        sc2_eci_tca = _sc2_eci_head_on(sc1_eci_tca, r_mag, seed)
    else:
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
