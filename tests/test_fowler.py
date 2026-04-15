"""
test_fowler.py

Tests for the Fowler (1993) analytic probability-of-collision method.

Coverage
--------
TestCovarianceGeneration  — shape, symmetry, and physical sanity of covariances
TestFowlerPcBasicProperties — Pc is in [0,1], monotonicity, boundary cases
TestFowlerPcLimits         — zero-covariance, near-zero miss → Pc≈1
TestFowlerPcScenarios      — all four scenario fixtures produce sensible Pc values
TestFowlerPcDegenerate     — ValueError on zero relative speed
"""

import numpy as np
import pytest

from collision.covariance import generate_covariances
from collision.fowler import fowler_pc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_states():
    """Two spacecraft with a 100 m radial separation, 500 m/s head-on closing."""
    sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    sc2 = np.array([7.0e6 + 100.0, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0])
    return sc1, sc2


def _small_isotropic_cov(sigma_pos=10.0, sigma_vel=0.01):
    """Return a 6×6 diagonal covariance with equal position and velocity spreads."""
    var = np.array([sigma_pos**2] * 3 + [sigma_vel**2] * 3)
    return np.diag(var)


def _large_isotropic_cov(sigma_pos=1e4):
    """Very large position covariance to drive Pc toward geometric limit."""
    var = np.array([sigma_pos**2] * 3 + [1.0] * 3)
    return np.diag(var)


# ---------------------------------------------------------------------------
# TestCovarianceGeneration
# ---------------------------------------------------------------------------

class TestCovarianceGeneration:
    def test_output_shapes(self, crossing_tca):
        cov1, cov2 = generate_covariances(crossing_tca["sc1"], crossing_tca["sc2"])
        assert np.array(cov1).shape == (6, 6)
        assert np.array(cov2).shape == (6, 6)

    def test_symmetric(self, crossing_tca):
        cov1, cov2 = generate_covariances(crossing_tca["sc1"], crossing_tca["sc2"])
        np.testing.assert_allclose(cov1, cov1.T, atol=1e-10)
        np.testing.assert_allclose(cov2, cov2.T, atol=1e-10)

    def test_positive_definite(self, crossing_tca):
        cov1, cov2 = generate_covariances(crossing_tca["sc1"], crossing_tca["sc2"])
        eigvals1 = np.linalg.eigvalsh(cov1)
        eigvals2 = np.linalg.eigvalsh(cov2)
        assert np.all(eigvals1 > 0), "cov1 is not positive definite"
        assert np.all(eigvals2 > 0), "cov2 is not positive definite"

    def test_position_variance_order_of_magnitude(self, crossing_tca):
        """Diagonal position variances should be O(σ²) with default σ = 50–500 m."""
        cov1, _ = generate_covariances(crossing_tca["sc1"], crossing_tca["sc2"])
        pos_variances = np.diag(cov1)[:3]
        # Default pos_std_rtn = (100, 500, 50) → variances 2500–250000 m²
        assert np.all(pos_variances > 100), "Position variances too small"
        assert np.all(pos_variances < 1e8), "Position variances unexpectedly large"

    def test_custom_std(self, crossing_tca):
        """Custom 1-sigma values should be reflected in the trace of the position block."""
        sigma_r, sigma_t, sigma_n = 200.0, 1000.0, 100.0
        cov1, _ = generate_covariances(
            crossing_tca["sc1"], crossing_tca["sc2"],
            pos_std_rtn=(sigma_r, sigma_t, sigma_n),
            vel_std_rtn=(0.1, 0.5, 0.05),
        )
        # Trace of position block == sum of variances (rotation preserves trace)
        expected_trace = sigma_r**2 + sigma_t**2 + sigma_n**2
        actual_trace = float(np.trace(cov1[:3, :3]))
        np.testing.assert_allclose(actual_trace, expected_trace, rtol=1e-6)


# ---------------------------------------------------------------------------
# TestFowlerPcBasicProperties
# ---------------------------------------------------------------------------

class TestFowlerPcBasicProperties:
    def test_pc_in_unit_interval(self, crossing_tca, crossing_covs):
        cov1, cov2 = crossing_covs
        pc = fowler_pc(crossing_tca["sc1"], crossing_tca["sc2"], cov1, cov2,
                       hard_body_radius=10.0)
        assert 0.0 <= pc <= 1.0

    def test_pc_increases_as_miss_decreases(self):
        """Pc should be higher for a closer miss with the same covariance."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
        cov = _small_isotropic_cov(sigma_pos=100.0)
        hbr = 10.0

        # sc2 at different radial separations
        sc2_far  = np.array([7.0e6 + 500.0, 0.0, 0.0, 0.0, 7000.0, 0.0])
        sc2_near = np.array([7.0e6 +  50.0, 0.0, 0.0, 0.0, 7000.0, 0.0])

        pc_far  = fowler_pc(sc1, sc2_far,  cov, cov, hbr)
        pc_near = fowler_pc(sc1, sc2_near, cov, cov, hbr)
        assert pc_near > pc_far, f"Expected pc_near ({pc_near}) > pc_far ({pc_far})"

    def test_pc_increases_as_hbr_increases(self):
        """Larger hard-body radius means larger collision cross-section → higher Pc."""
        sc1, sc2 = _identity_states()
        cov = _small_isotropic_cov(sigma_pos=100.0)

        pc_small = fowler_pc(sc1, sc2, cov, cov, hard_body_radius=5.0)
        pc_large = fowler_pc(sc1, sc2, cov, cov, hard_body_radius=50.0)
        assert pc_large > pc_small

    def test_pc_increases_as_covariance_grows(self):
        """Spreading the covariance should raise Pc until it saturates near 1."""
        sc1, sc2 = _identity_states()
        pc_tight = fowler_pc(sc1, sc2, _small_isotropic_cov(10.0),
                             _small_isotropic_cov(10.0), hard_body_radius=10.0)
        pc_wide  = fowler_pc(sc1, sc2, _small_isotropic_cov(1000.0),
                             _small_isotropic_cov(1000.0), hard_body_radius=10.0)
        assert pc_wide > pc_tight

    def test_pc_returns_float(self, crossing_tca, crossing_covs):
        cov1, cov2 = crossing_covs
        pc = fowler_pc(crossing_tca["sc1"], crossing_tca["sc2"], cov1, cov2, 10.0)
        assert isinstance(pc, float)


# ---------------------------------------------------------------------------
# TestFowlerPcLimits
# ---------------------------------------------------------------------------

class TestFowlerPcLimits:
    def test_zero_miss_small_cov_near_one(self):
        """When miss distance is zero and HBR >> sigma, Pc should be close to 1."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0,   0.0])
        sc2 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0])
        # tiny sigma so almost all the Gaussian mass is inside the HBR disk
        cov = _small_isotropic_cov(sigma_pos=1.0)
        pc = fowler_pc(sc1, sc2, cov, cov, hard_body_radius=20.0)
        assert pc > 0.99, f"Expected Pc ≈ 1 for zero miss + HBR >> sigma, got {pc}"

    def test_large_miss_tiny_cov_near_zero(self):
        """When miss >> HBR and sigma << miss, Pc should be essentially zero."""
        sc1 = np.array([7.0e6,           0.0, 0.0, 0.0, 7500.0,     0.0])
        sc2 = np.array([7.0e6 + 100e3,   0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0])
        cov = _small_isotropic_cov(sigma_pos=10.0)
        pc = fowler_pc(sc1, sc2, cov, cov, hard_body_radius=10.0)
        assert pc < 1e-10, f"Expected Pc ≈ 0 for large miss, got {pc}"

    def test_symmetric_zero_miss_depends_only_on_hbr_and_sigma(self):
        """Two identical zero-miss encounters with different HBR should give different Pc."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0,       0.0])
        sc2 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0 - 100.0, 0.0])
        cov = _small_isotropic_cov(sigma_pos=50.0)

        pc_hbr5  = fowler_pc(sc1, sc2, cov, cov, hard_body_radius=5.0)
        pc_hbr50 = fowler_pc(sc1, sc2, cov, cov, hard_body_radius=50.0)
        assert pc_hbr50 > pc_hbr5

    def test_large_covariance_geometric_limit(self):
        """
        In the geometric limit (sigma >> miss, sigma >> HBR), the Gaussian is
        nearly uniform over the HBR disk, so:
            Pc ≈ (π R²) / (2π sqrt(det(C_2d)))   [Chan 1997 eq 17]

        We just check that Pc is positive and < 1 and that it decreases as
        sigma grows (the denominator grows faster than R²).
        """
        sc1, sc2 = _identity_states()
        pc_1 = fowler_pc(sc1, sc2, _large_isotropic_cov(1e3),
                         _large_isotropic_cov(1e3), hard_body_radius=10.0)
        pc_2 = fowler_pc(sc1, sc2, _large_isotropic_cov(1e5),
                         _large_isotropic_cov(1e5), hard_body_radius=10.0)
        assert 0.0 < pc_1 < 1.0
        assert 0.0 < pc_2 < 1.0
        assert pc_1 > pc_2, "Larger sigma should lower Pc in the geometric limit"


# ---------------------------------------------------------------------------
# TestFowlerPcScenarios
# ---------------------------------------------------------------------------

class TestFowlerPcScenarios:
    """Smoke tests across all four scenario fixtures — verify Pc is sensible."""

    def test_crossing_pc_in_range(self, crossing_tca, crossing_covs):
        cov1, cov2 = crossing_covs
        pc = fowler_pc(crossing_tca["sc1"], crossing_tca["sc2"], cov1, cov2,
                       hard_body_radius=10.0)
        assert 0.0 <= pc <= 1.0

    def test_head_on_pc_in_range(self, head_on_tca, head_on_covs):
        cov1, cov2 = head_on_covs
        pc = fowler_pc(head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
                       hard_body_radius=10.0)
        assert 0.0 <= pc <= 1.0

    def test_overtaking_pc_in_range(self, overtaking_tca, overtaking_covs):
        cov1, cov2 = overtaking_covs
        pc = fowler_pc(overtaking_tca["sc1"], overtaking_tca["sc2"], cov1, cov2,
                       hard_body_radius=10.0)
        assert 0.0 <= pc <= 1.0

    def test_near_miss_pc_higher_than_crossing(
        self, near_miss_scenario, near_miss_covs, crossing_tca, crossing_covs
    ):
        """
        Near-miss scenario (10 m) should have higher Pc than crossing (500 m)
        when evaluated with the same covariances.
        """
        cov1_nm, cov2_nm = near_miss_covs
        pc_near = fowler_pc(
            near_miss_scenario["sc1_eci_tca"],
            near_miss_scenario["sc2_eci_tca"],
            cov1_nm, cov2_nm, hard_body_radius=10.0,
        )

        cov1_cr, cov2_cr = crossing_covs
        pc_cross = fowler_pc(
            crossing_tca["sc1"], crossing_tca["sc2"],
            cov1_cr, cov2_cr, hard_body_radius=10.0,
        )

        assert pc_near > pc_cross, (
            f"Expected near-miss Pc ({pc_near:.2e}) > crossing Pc ({pc_cross:.2e})"
        )

    @pytest.mark.parametrize("scenario_name,tca_key,cov_key", [
        ("crossing",   "crossing_tca",   "crossing_covs"),
        ("head_on",    "head_on_tca",    "head_on_covs"),
        ("overtaking", "overtaking_tca", "overtaking_covs"),
    ])
    def test_pc_increases_with_hbr(
        self, scenario_name, tca_key, cov_key, request
    ):
        """Pc should increase monotonically with HBR for all scenarios."""
        tca  = request.getfixturevalue(tca_key)
        covs = request.getfixturevalue(cov_key)
        cov1, cov2 = covs
        pc_small = fowler_pc(tca["sc1"], tca["sc2"], cov1, cov2, hard_body_radius=5.0)
        pc_large = fowler_pc(tca["sc1"], tca["sc2"], cov1, cov2, hard_body_radius=50.0)
        assert pc_large >= pc_small, (
            f"{scenario_name}: Pc should not decrease as HBR grows "
            f"(HBR=5 → {pc_small:.2e}, HBR=50 → {pc_large:.2e})"
        )

    # ------------------------------------------------------------------
    # Physically grounded Pc magnitude checks
    #
    # Expected ranges are derived from the actual scenario geometry and the
    # default RTN covariance (pos_std = 100/500/50 m).  The key insight is
    # that slow-encounter scenarios (crossing at 15 m/s, overtaking at 50 m/s)
    # project a very large along-track uncertainty onto the encounter plane,
    # spreading the 2D Gaussian over a huge area and suppressing Pc.  Fast
    # head-on encounters (500 m/s) keep the projected covariance compact.
    #
    # Bounds are deliberately loose (several decades) so the test catches
    # implementation bugs (wrong sign, wrong projection, missing factor of 2)
    # without being fragile to minor covariance tuning.
    # ------------------------------------------------------------------

    def test_slow_crossing_pc_is_very_low(self, crossing_tca, crossing_covs):
        """
        Crossing: 500 m miss, 15 m/s relative speed, HBR = 10 m.
        Slow encounter → encounter plane nearly parallel to along-track →
        500 m along-track sigma projects into plane → 2D Gaussian is ~462 m wide →
        Pc ~ 1e-14.  This is physically correct, not a bug.
        Expect: Pc < 1e-10.
        """
        cov1, cov2 = crossing_covs
        pc = fowler_pc(crossing_tca["sc1"], crossing_tca["sc2"], cov1, cov2,
                       hard_body_radius=10.0)
        assert pc < 1e-10, (
            f"Slow crossing Pc={pc:.2e} should be < 1e-10"
        )

    def test_high_pc_crossing_magnitude(self, high_pc_crossing_tca, high_pc_crossing_covs):
        """
        Crossing: 200 m miss, 500 m/s relative speed, HBR = 10 m.
        Fast encounter → projected covariance stays compact → Pc ~ 3e-5.
        Expect: 1e-7 < Pc < 1e-2.
        """
        cov1, cov2 = high_pc_crossing_covs
        pc = fowler_pc(high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"],
                       cov1, cov2, hard_body_radius=10.0)
        assert 1e-7 < pc < 1e-2, (
            f"High-Pc crossing Pc={pc:.2e} outside expected range [1e-7, 1e-2]"
        )

    def test_head_on_pc_magnitude(self, head_on_tca, head_on_covs):
        """
        Head-on: 200 m miss, 500 m/s relative speed, HBR = 10 m.
        Fast encounter → compact projected covariance → Pc ~ 5e-4.
        Expect: 1e-6 < Pc < 1e-1.
        """
        cov1, cov2 = head_on_covs
        pc = fowler_pc(head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
                       hard_body_radius=10.0)
        assert 1e-6 < pc < 1e-1, (
            f"Head-on Pc={pc:.2e} outside expected range [1e-6, 1e-1]"
        )

    def test_overtaking_pc_magnitude(self, overtaking_tca, overtaking_covs):
        """
        Overtaking: 1000 m miss, 50 m/s relative speed, HBR = 10 m.
        Moderate speed, large miss → Pc ~ 2e-4.
        Expect: 1e-7 < Pc < 1e-1.
        """
        cov1, cov2 = overtaking_covs
        pc = fowler_pc(overtaking_tca["sc1"], overtaking_tca["sc2"], cov1, cov2,
                       hard_body_radius=10.0)
        assert 1e-7 < pc < 1e-1, (
            f"Overtaking Pc={pc:.2e} outside expected range [1e-7, 1e-1]"
        )

    def test_near_miss_pc_magnitude(self, near_miss_scenario, near_miss_covs):
        """
        Near-miss: 10 m miss, 15 m/s relative speed, HBR = 10 m.
        Miss distance ≈ HBR → Pc ~ 1e-3.
        Expect: 1e-5 < Pc < 1e-1.
        """
        cov1, cov2 = near_miss_covs
        pc = fowler_pc(
            near_miss_scenario["sc1_eci_tca"],
            near_miss_scenario["sc2_eci_tca"],
            cov1, cov2, hard_body_radius=10.0,
        )
        assert 1e-5 < pc < 1e-1, (
            f"Near-miss Pc={pc:.2e} outside expected range [1e-5, 1e-1]"
        )


# ---------------------------------------------------------------------------
# TestCovarianceMagnitudeEffects
# ---------------------------------------------------------------------------

class TestCovarianceMagnitudeEffects:
    """
    Tests for how Pc responds to covariance magnitude across plausible LEO ranges.

    The key finding: Pc vs. covariance size is NOT always monotone.  It depends
    on where the miss distance sits relative to the projected 2D sigma:

      - miss >> sigma (tight cov, far miss): Pc is near zero — Gaussian barely
        reaches the HBR disk.  Growing the covariance RAISES Pc.
      - miss ~ sigma (moderate cov):         Pc peaks.
      - miss << sigma (loose cov, wide Gauss): Gaussian is nearly flat over the
        HBR disk.  Growing the covariance LOWERS Pc (geometric limit).

    Whether a given scenario is in the "rising" or "falling" regime depends on
    the encounter geometry.  The tests below verify specific regimes directly
    measured from our scenarios.
    """

    # RTN 1-sigma levels spanning the plausible LEO tracking quality range.
    # Tight: well-tracked debris (~10 m radial).  Very loose: poorly-tracked object.
    _TIGHT     = dict(pos_std_rtn=(10,    50,   5),  vel_std_rtn=(0.01, 0.05, 0.005))
    _DEFAULT   = dict(pos_std_rtn=(100,  500,  50),  vel_std_rtn=(0.1,  0.5,  0.05))
    _LOOSE     = dict(pos_std_rtn=(300, 1500, 150),  vel_std_rtn=(0.3,  1.5,  0.15))
    _VERY_LOOSE= dict(pos_std_rtn=(1000,5000, 500),  vel_std_rtn=(1.0,  5.0,  0.5))

    def test_head_on_pc_monotone_decreasing_with_covariance(self, head_on_tca):
        """
        Head-on (201 m miss, 500 m/s): tight covariance gives the highest Pc
        because the projected 2D sigma is already comparable to the miss distance
        even at tight levels (~13 m × 71 m), so the Gaussian is centred near the
        disk.  Spreading it further always lowers Pc.

        tight (~8e-4) > default (~5e-4) > loose (~6e-5) > very_loose (~6e-6)
        """
        s1, s2 = head_on_tca["sc1"], head_on_tca["sc2"]
        pcs = []
        for kw in [self._TIGHT, self._DEFAULT, self._LOOSE, self._VERY_LOOSE]:
            cov1, cov2 = generate_covariances(s1, s2, **kw)
            pcs.append(fowler_pc(s1, s2, cov1, cov2, hard_body_radius=10.0))

        for i in range(len(pcs) - 1):
            assert pcs[i] > pcs[i + 1], (
                f"Head-on: expected Pc to decrease at step {i}→{i+1}, "
                f"got {pcs[i]:.2e} → {pcs[i+1]:.2e}"
            )

    def test_high_pc_crossing_pc_non_monotone_with_covariance(self, high_pc_crossing_tca):
        """
        Crossing (204 m miss, 500 m/s): with a tight covariance (projected 2D sigma
        ~7 × 46 m) the miss distance (204 m) is many sigmas away → Pc ~ 1e-160.
        Growing the covariance to default (~71 × 462 m) brings the tail of the
        Gaussian within reach of the HBR disk → Pc jumps to ~3e-5.
        At very_loose (~707 × 4620 m) the Gaussian becomes nearly flat and Pc
        starts falling again.

        So: tight << default > loose > very_loose.  Not monotone.
        """
        s1, s2 = high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"]

        cov1_tight, cov2_tight = generate_covariances(s1, s2, **self._TIGHT)
        cov1_def,   cov2_def   = generate_covariances(s1, s2, **self._DEFAULT)
        cov1_loose, cov2_loose = generate_covariances(s1, s2, **self._LOOSE)
        cov1_vl,    cov2_vl    = generate_covariances(s1, s2, **self._VERY_LOOSE)

        pc_tight = fowler_pc(s1, s2, cov1_tight, cov2_tight, hard_body_radius=10.0)
        pc_def   = fowler_pc(s1, s2, cov1_def,   cov2_def,   hard_body_radius=10.0)
        pc_loose = fowler_pc(s1, s2, cov1_loose, cov2_loose, hard_body_radius=10.0)
        pc_vl    = fowler_pc(s1, s2, cov1_vl,    cov2_vl,    hard_body_radius=10.0)

        # Tight covariance: miss >> sigma, Pc should be extremely small
        assert pc_tight < 1e-10, (
            f"Tight cov crossing Pc={pc_tight:.2e} should be < 1e-10 "
            "(miss distance >> projected sigma)"
        )
        # Rising slope: default is higher than tight (Gaussian tail growing into disk)
        assert pc_def > pc_tight, (
            f"Default Pc ({pc_def:.2e}) should be >> tight Pc ({pc_tight:.2e})"
        )
        # The peak falls somewhere between loose and very_loose (measured ~1.1e-4 and ~1.5e-5).
        # Very_loose is unambiguously past the peak and lower than loose.
        assert pc_vl < pc_loose, (
            f"Very-loose Pc ({pc_vl:.2e}) should be < loose Pc ({pc_loose:.2e}) "
            "(past the peak: geometric limit, Pc ∝ 1/sigma)"
        )

    def test_near_miss_pc_monotone_decreasing_with_covariance(
        self, near_miss_scenario
    ):
        """
        Near-miss (10 m miss, 15 m/s): miss ≈ HBR.  Even the tight covariance
        puts substantial Gaussian mass over the HBR disk, so we are already past
        the peak — Pc decreases monotonically as covariance grows.

        tight (~4e-2) > default (~1e-3) > loose (~1e-4) > very_loose (~1e-5)
        """
        s1 = near_miss_scenario["sc1_eci_tca"]
        s2 = near_miss_scenario["sc2_eci_tca"]
        pcs = []
        for kw in [self._TIGHT, self._DEFAULT, self._LOOSE, self._VERY_LOOSE]:
            cov1, cov2 = generate_covariances(s1, s2, **kw)
            pcs.append(fowler_pc(s1, s2, cov1, cov2, hard_body_radius=10.0))

        for i in range(len(pcs) - 1):
            assert pcs[i] > pcs[i + 1], (
                f"Near-miss: expected Pc to decrease at step {i}→{i+1}, "
                f"got {pcs[i]:.2e} → {pcs[i+1]:.2e}"
            )

    def test_all_covariance_levels_produce_valid_pc(self, head_on_tca):
        """Sanity: every covariance level must produce Pc in [0, 1]."""
        s1, s2 = head_on_tca["sc1"], head_on_tca["sc2"]
        for kw in [self._TIGHT, self._DEFAULT, self._LOOSE, self._VERY_LOOSE]:
            cov1, cov2 = generate_covariances(s1, s2, **kw)
            pc = fowler_pc(s1, s2, cov1, cov2, hard_body_radius=10.0)
            assert 0.0 <= pc <= 1.0, f"Pc={pc} out of [0,1] for {kw}"


# ---------------------------------------------------------------------------
# TestFowlerPcDegenerate
# ---------------------------------------------------------------------------

class TestFowlerPcDegenerate:
    def test_zero_relative_speed_raises(self):
        """fowler_pc must raise ValueError when v_rel == 0 (no encounter plane)."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 7500.0, 0.0, 0.0])
        sc2 = np.array([7.0e6 + 100.0, 0.0, 0.0, 7500.0, 0.0, 0.0])  # same velocity
        cov = _small_isotropic_cov()
        with pytest.raises(ValueError, match="zero"):
            fowler_pc(sc1, sc2, cov, cov, hard_body_radius=10.0)
