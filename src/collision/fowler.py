"""
fowler.py

Analytic 2D-projection probability of collision (Fowler 1993).

Reference:
    Fowler, W.T. (1993). "A Probability of Collision Analysis for Conjunction
    Assessments." AAS 93-177.

Algorithm summary
-----------------
At TCA, both spacecraft covariances are projected onto the *encounter plane* —
the plane perpendicular to the relative velocity vector.  The combined 2D
Gaussian is then integrated over a disk of radius equal to the hard-body radius
(HBR), which represents the combined physical cross-section of the two objects.

Steps
-----
1. Relative position r_rel = r1 − r2, relative velocity v_rel = v1 − v2
2. Encounter-plane basis vectors:
       z_hat = v_rel / |v_rel|                    (along relative velocity)
       x_hat = (r_rel − (r_rel·z_hat) z_hat) / |…|  (in-plane, toward miss)
       y_hat = z_hat × x_hat                       (completes right-hand frame)
3. Combined position covariance: C_pos = C1[:3,:3] + C2[:3,:3]
4. Project to 2D encounter plane:
       B     = [[x_hat], [y_hat]]   shape (2, 3)
       C_2d  = B @ C_pos @ B.T      shape (2, 2)
       miss_2d = B @ r_rel          shape (2,)
5. Integrate 2D Gaussian over hard-body disk using scipy.integrate.dblquad.
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal


def fowler_pc(
    sc1_eci: np.ndarray,
    sc2_eci: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    hard_body_radius: float,
) -> float:
    """
    Compute probability of collision using the Fowler (1993) analytic method.

    Args:
        sc1_eci:          [6,] ECI state of SC1 at TCA (m, m/s)
        sc2_eci:          [6,] ECI state of SC2 at TCA (m, m/s)
        cov1:             [6,6] position-velocity covariance of SC1 in ECI
        cov2:             [6,6] position-velocity covariance of SC2 in ECI
        hard_body_radius: combined hard-body radius of both objects (m)

    Returns:
        Pc: probability of collision in [0, 1]

    Raises:
        ValueError: if relative speed at TCA is zero (degenerate encounter plane)
    """
    r1 = sc1_eci[:3]
    v1 = sc1_eci[3:]
    r2 = sc2_eci[:3]
    v2 = sc2_eci[3:]

    r_rel = r1 - r2
    v_rel = v1 - v2

    v_rel_mag = float(np.linalg.norm(v_rel))
    if v_rel_mag == 0.0:
        raise ValueError("Relative speed at TCA is zero — encounter plane is undefined.")

    # ------------------------------------------------------------------
    # 1. Build encounter-plane orthonormal basis
    # ------------------------------------------------------------------
    z_hat = v_rel / v_rel_mag

    # Component of r_rel perpendicular to z_hat (the in-plane miss vector)
    r_perp = r_rel - np.dot(r_rel, z_hat) * z_hat
    r_perp_mag = float(np.linalg.norm(r_perp))

    if r_perp_mag < 1e-10:
        # r_rel is parallel to v_rel (head-on, zero impact parameter).
        # Pick an arbitrary x_hat perpendicular to z_hat.
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arbitrary, z_hat)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        r_perp = arbitrary - np.dot(arbitrary, z_hat) * z_hat
        r_perp_mag = float(np.linalg.norm(r_perp))

    x_hat = r_perp / r_perp_mag
    y_hat = np.cross(z_hat, x_hat)

    # ------------------------------------------------------------------
    # 2. Combined position covariance and 2D projection
    # ------------------------------------------------------------------
    C_pos = cov1[:3, :3] + cov2[:3, :3]        # (3, 3)
    B = np.array([x_hat, y_hat])                # (2, 3)
    C_2d = B @ C_pos @ B.T                      # (2, 2)
    miss_2d = B @ r_rel                         # (2,)

    # ------------------------------------------------------------------
    # 3. Integrate 2D Gaussian over hard-body disk
    # ------------------------------------------------------------------
    return _integrate_gaussian_over_disk(miss_2d, C_2d, hard_body_radius)


def _integrate_gaussian_over_disk(
    mean: np.ndarray,
    cov: np.ndarray,
    radius: float,
) -> float:
    """
    Numerically integrate a 2D Gaussian over a disk centred at the origin.

    ∫∫_{x²+y²≤R²} N([x,y] | mean, cov) dx dy

    Uses scipy.integrate.dblquad with the disk parameterised as
        y in [-R, R],  x in [-sqrt(R²-y²), sqrt(R²-y²)].
    """
    rv = multivariate_normal(mean=mean, cov=cov)
    R = radius

    def integrand(x: float, y: float) -> float:
        return float(rv.pdf(np.array([x, y])))

    def x_lo(y: float) -> float:
        return -np.sqrt(max(R * R - y * y, 0.0))

    def x_hi(y: float) -> float:
        return  np.sqrt(max(R * R - y * y, 0.0))

    result, _err = dblquad(
        integrand,
        -R, R,          # outer integral: y
        x_lo, x_hi,     # inner integral: x(y)
        epsabs=1e-10,
        epsrel=1e-8,
    )
    # Clamp to [0, 1] to guard against tiny numerical overshoots
    return float(np.clip(result, 0.0, 1.0))
