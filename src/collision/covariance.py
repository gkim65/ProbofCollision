"""
covariance.py

Generate realistic position-velocity covariances for LEO spacecraft in the
RTN frame and rotate them to ECI.

Typical LEO position uncertainty (1-sigma):
    Radial:    10 –  100 m
    Along-track: 100 – 1000 m   (largest due to along-track uncertainty)
    Cross-track:  10 –  100 m

Typical LEO velocity uncertainty (1-sigma):
    Radial:    0.01 – 0.1 m/s
    Along-track: 0.1  – 0.5 m/s
    Cross-track: 0.01 – 0.05 m/s
"""

import numpy as np
import brahe
from brahe import rotation_rtn_to_eci


def generate_covariances(
    sc1_eci_tca: np.ndarray,
    sc2_eci_tca: np.ndarray,
    pos_std_rtn: tuple[float, float, float] = (100.0, 500.0, 50.0),
    vel_std_rtn: tuple[float, float, float] = (0.1, 0.5, 0.05),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 6×6 ECI covariance matrices for two spacecraft from diagonal RTN
    uncertainties.

    The covariance is first constructed as diagonal in the RTN frame, then
    rotated to ECI using brahe.rotation_rtn_to_eci.  Both spacecraft receive
    the same uncertainty profile (different orientations because their ECI
    states differ).

    Args:
        sc1_eci_tca: [6,] ECI state of SC1 at TCA (m, m/s)
        sc2_eci_tca: [6,] ECI state of SC2 at TCA (m, m/s)
        pos_std_rtn: 1-sigma position uncertainties (R, T, N) in metres
        vel_std_rtn: 1-sigma velocity uncertainties (R, T, N) in m/s

    Returns:
        cov1: [6,6] position-velocity covariance of SC1 in ECI (m², m²/s², m·m/s)
        cov2: [6,6] position-velocity covariance of SC2 in ECI (m², m²/s², m·m/s)
    """
    cov1 = _build_cov_eci(sc1_eci_tca, pos_std_rtn, vel_std_rtn)
    cov2 = _build_cov_eci(sc2_eci_tca, pos_std_rtn, vel_std_rtn)
    return cov1, cov2


def _build_cov_eci(
    eci_state: np.ndarray,
    pos_std_rtn: tuple[float, float, float],
    vel_std_rtn: tuple[float, float, float],
) -> np.ndarray:
    """
    Build a 6×6 ECI covariance from diagonal RTN uncertainties for one spacecraft.

    Strategy:
        1. Build 6×6 diagonal covariance in RTN (position block, velocity block).
        2. Assemble 6×6 rotation matrix T_rtn_to_eci from the 3×3 block.
        3. Rotate: C_eci = T @ C_rtn @ T.T
    """
    R_mat = np.array(rotation_rtn_to_eci(eci_state))   # (3, 3) RTN → ECI

    # Full 6×6 rotation (position and velocity blocks share the same rotation)
    T = np.zeros((6, 6))
    T[:3, :3] = R_mat
    T[3:, 3:] = R_mat

    # Diagonal covariance in RTN
    pos_var = np.array(pos_std_rtn) ** 2   # (R, T, N) variances in m²
    vel_var = np.array(vel_std_rtn) ** 2   # (R, T, N) variances in m²/s²
    C_rtn = np.diag(np.concatenate([pos_var, vel_var]))  # (6, 6)

    return T @ C_rtn @ T.T
