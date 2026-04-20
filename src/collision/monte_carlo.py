"""
monte_carlo.py

Monte Carlo probability of collision baseline.

Reference:
    Alfriend, K.T. et al. (2010). "Spacecraft Conjunction Analysis and the
    Probability of Collision." Chapter 5.

Algorithm summary
-----------------
The encounter is modelled in the 2D encounter plane (perpendicular to the
relative velocity vector at TCA), consistent with Fowler (1993) and Chan
(1997).  The combined 3×3 position covariance is projected to a 2×2 matrix
in this plane; the 2D miss vector is sampled and compared against the
hard-body circle of radius HBR.  This ensures that all three methods
(Fowler, Chan, Monte Carlo) are solving the same integral and their results
are directly comparable.

Steps
-----
1. Compute r_rel = r1 − r2 and C_pos = C1[:3,:3] + C2[:3,:3] at TCA.
2. Build the encounter-plane orthonormal basis (x_hat, y_hat) ⊥ v_rel/|v_rel|
   using the same construction as fowler.py and chan1997.py.
3. Project to 2D:
       B       = [[x_hat], [y_hat]]   shape (2, 3)
       C_2d    = B @ C_pos @ B.T      shape (2, 2)
       miss_2d = B @ r_rel            shape (2,)
4. Draw N samples from N(miss_2d, C_2d) using a seeded RNG.
5. Count samples where ||r_sample||₂ < HBR.
6. Pc_mc ≈ count / N.
7. 95% confidence interval:
       half_width = 2 * sqrt(pc * (1 - pc) / N)
       CI = (pc − half_width, pc + half_width), clamped to [0, 1].

Numerical notes
---------------
- N = 1_000_000 is sufficient for Pc ~ 1e-4 (relative std ≈ 3%).
- For Pc < 1e-6 the count is typically 0 and the CI is uninformative;
  use importance_sampling.py for those regimes.
- The seed ensures reproducibility across calls; pass seed=None for
  non-deterministic sampling.
- If the relative speed at TCA is zero, a ValueError is raised (no
  encounter plane can be defined).
"""

import numpy as np

__all__ = ["monte_carlo_pc"]


def monte_carlo_pc(
    sc1_eci: np.ndarray,
    sc2_eci: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    hard_body_radius: float,
    n_samples: int = 1_000_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Estimate probability of collision via Monte Carlo sampling in the encounter plane.

    Projects the combined position covariance onto the encounter plane
    (perpendicular to v_rel), samples the 2D relative-position distribution,
    and counts the fraction of samples within the hard-body circle.  The
    encounter-plane construction is identical to fowler_pc and chan_pc, so
    results are directly comparable.

    Args:
        sc1_eci:          [6,] ECI state of SC1 at TCA (m, m/s)
        sc2_eci:          [6,] ECI state of SC2 at TCA (m, m/s)
        cov1:             [6,6] position-velocity covariance of SC1 in ECI
        cov2:             [6,6] position-velocity covariance of SC2 in ECI
        hard_body_radius: combined hard-body radius of both objects (m)
        n_samples:        number of Monte Carlo samples (default: 1_000_000)
        seed:             RNG seed for reproducibility (default: 42)

    Returns:
        (pc_estimate, ci_low, ci_high): point estimate and 95% CI bounds,
        all in [0, 1].

    Raises:
        ValueError: if n_samples < 1 or hard_body_radius <= 0.
        ValueError: if relative speed at TCA is zero (degenerate encounter plane).
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if hard_body_radius <= 0.0:
        raise ValueError(f"hard_body_radius must be > 0, got {hard_body_radius}")

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
    # Build encounter-plane orthonormal basis (identical to fowler.py)
    # ------------------------------------------------------------------
    z_hat = v_rel / v_rel_mag

    r_perp = r_rel - np.dot(r_rel, z_hat) * z_hat
    r_perp_mag = float(np.linalg.norm(r_perp))

    if r_perp_mag < 1e-10:
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arbitrary, z_hat)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        r_perp = arbitrary - np.dot(arbitrary, z_hat) * z_hat
        r_perp_mag = float(np.linalg.norm(r_perp))

    x_hat = r_perp / r_perp_mag
    y_hat = np.cross(z_hat, x_hat)

    # ------------------------------------------------------------------
    # Project to 2D encounter plane
    # ------------------------------------------------------------------
    C_pos = cov1[:3, :3] + cov2[:3, :3]   # (3, 3)
    B = np.array([x_hat, y_hat])           # (2, 3)
    C_2d = B @ C_pos @ B.T                 # (2, 2)
    miss_2d = B @ r_rel                    # (2,)

    # ------------------------------------------------------------------
    # Sample and count
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(miss_2d, C_2d, size=n_samples)  # (N, 2)

    distances_sq = np.sum(samples**2, axis=1)
    count = int(np.sum(distances_sq < hard_body_radius**2))

    pc = count / n_samples

    # 95% normal-approximation CI: pc ± 2*sqrt(pc*(1-pc)/N)
    half_width = 2.0 * float(np.sqrt(pc * (1.0 - pc) / n_samples))
    ci_low = float(np.clip(pc - half_width, 0.0, 1.0))
    ci_high = float(np.clip(pc + half_width, 0.0, 1.0))

    return float(pc), ci_low, ci_high
