"""
test_monte_carlo_3d.py

Tests for the 3D trajectory-integrated Monte Carlo Pc method.

The 3D method perturbs full 6D states at TCA and finds the minimum miss
distance over a time window, rather than projecting to the encounter plane.
For fast encounters it converges toward the 2D result; for slow encounters
it can differ because velocity uncertainty genuinely shifts the true TCA.

Coverage
--------
TestMC3DBasic           — return type, range, reproducibility, CI properties
TestMC3DVs2D            — agreement with 2D MC within factor-of-5 where counts allow
TestMC3DProperties      — monotonicity, larger N narrows CI
TestMC3DDegenerate      — ValueError on bad inputs
"""

import numpy as np
import pytest

from collision.monte_carlo_3d import monte_carlo_3d_pc
from collision.monte_carlo import monte_carlo_pc
from collision.chan1997 import chan_pc
from collision.covariance import generate_covariances


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _isotropic_cov(sigma_pos=100.0):
    var = np.array([sigma_pos**2] * 3 + [0.1**2] * 3)
    return np.diag(var)


# ---------------------------------------------------------------------------
# TestMC3DBasic
# ---------------------------------------------------------------------------

class TestMC3DBasic:
    """Return-type, range, and reproducibility invariants."""

    def test_returns_tuple_of_three(self, head_on_mc3d):
        assert len(head_on_mc3d) == 3

    def test_all_elements_are_float(self, head_on_mc3d):
        pc, ci_low, ci_high = head_on_mc3d
        assert isinstance(pc, float)
        assert isinstance(ci_low, float)
        assert isinstance(ci_high, float)

    def test_pc_in_unit_interval(self, head_on_mc3d):
        pc, _, _ = head_on_mc3d
        assert 0.0 <= pc <= 1.0

    def test_ci_ordered(self, head_on_mc3d):
        pc, ci_low, ci_high = head_on_mc3d
        assert ci_low <= pc <= ci_high

    def test_ci_bounds_in_unit_interval(self, head_on_mc3d):
        _, ci_low, ci_high = head_on_mc3d
        assert 0.0 <= ci_low <= 1.0
        assert 0.0 <= ci_high <= 1.0

    def test_fixed_seed_reproducible(self, head_on_tca, head_on_covs):
        cov1, cov2 = head_on_covs
        r1 = monte_carlo_3d_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            head_on_tca["epoch"], 10.0, n_samples=500, seed=0,
        )
        r2 = monte_carlo_3d_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            head_on_tca["epoch"], 10.0, n_samples=500, seed=0,
        )
        assert r1 == r2

    def test_different_seeds_differ(self, head_on_tca, head_on_covs):
        """Different seeds should produce different sample counts.

        Use N=5000 so each seed expects ~25 hits for Pc~5e-3; the
        probability that two independent seeds give identical counts
        is < 5%, making this test robust.
        """
        cov1, cov2 = head_on_covs
        pc_a, _, _ = monte_carlo_3d_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            head_on_tca["epoch"], 10.0, n_samples=5_000, seed=1,
        )
        pc_b, _, _ = monte_carlo_3d_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            head_on_tca["epoch"], 10.0, n_samples=5_000, seed=2,
        )
        assert pc_a != pc_b

    def test_zero_miss_high_pc(self, head_on_tca, head_on_covs):
        """Near-zero miss + HBR >> sigma → Pc should be near 1."""
        cov1, cov2 = head_on_covs
        # Manually place SC2 at SC1's position
        sc1 = head_on_tca["sc1"].copy()
        sc2 = sc1.copy()
        sc2[3:] -= np.array([0.0, 0.0, 500.0])  # give relative velocity
        tight = _isotropic_cov(sigma_pos=1.0)
        pc, _, _ = monte_carlo_3d_pc(
            sc1, sc2, tight, tight, head_on_tca["epoch"],
            hard_body_radius=20.0, n_samples=500, seed=42,
        )
        assert pc > 0.9, f"Expected Pc ≈ 1 for zero miss / tiny cov, got {pc:.4f}"

    def test_large_miss_near_zero_pc(self, head_on_tca):
        """100 km miss with tight covariance → Pc = 0 at any N."""
        s1 = head_on_tca["sc1"].copy()
        s2 = s1.copy()
        s2[:3] += np.array([100e3, 0.0, 0.0])
        s2[3:] -= np.array([500.0, 0.0, 0.0])
        tight = _isotropic_cov(sigma_pos=10.0)
        pc, _, _ = monte_carlo_3d_pc(
            s1, s2, tight, tight, head_on_tca["epoch"],
            hard_body_radius=10.0, n_samples=1_000, seed=42,
        )
        assert pc == 0.0, f"Expected Pc=0 for 100 km miss, got {pc:.2e}"

    def test_larger_n_narrows_ci(self, head_on_tca, head_on_covs):
        """CI half-width should shrink as N grows."""
        cov1, cov2 = head_on_covs
        _, lo_small, hi_small = monte_carlo_3d_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            head_on_tca["epoch"], 10.0, n_samples=500, seed=42,
        )
        _, lo_large, hi_large = monte_carlo_3d_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            head_on_tca["epoch"], 10.0, n_samples=5_000, seed=42,
        )
        width_small = hi_small - lo_small
        width_large = hi_large - lo_large
        assert width_large <= width_small + 1e-4, (
            f"Larger N should not widen CI: N=500 width={width_small:.2e}, "
            f"N=5k width={width_large:.2e}"
        )


# ---------------------------------------------------------------------------
# TestMC3DVs2D
# ---------------------------------------------------------------------------

class TestMC3DVs2D:
    """
    The 3D method and 2D method both estimate Pc for the same conjunction.
    For fast encounters (v_rel > ~1 km/s), they should agree within a
    factor of 5 when enough samples are used.

    For slow encounters (overtaking at ~50 m/s), results can legitimately
    differ because velocity uncertainty genuinely shifts the true TCA away
    from the nominal — a real physical effect not captured by the 2D method.
    These scenarios are skipped from tight comparison tests.
    """

    _FACTOR = 5.0    # multiplicative tolerance (wider than 2D-vs-Chan because
                     # the 3D method has higher variance at equal N)
    _MIN_PC = 1e-4   # skip if 2D Pc is below this (insufficient 3D counts at N=10k)

    def _check(self, sc1, sc2, cov1, cov2, epoch, hbr=10.0):
        pc_2d, _, _ = monte_carlo_pc(sc1, sc2, cov1, cov2, hbr, seed=42)
        if pc_2d < self._MIN_PC:
            pytest.skip(f"2D Pc={pc_2d:.1e} too small for 3D comparison at N=10k")
        pc_3d, _, _ = monte_carlo_3d_pc(sc1, sc2, cov1, cov2, epoch, hbr,
                                         n_samples=10_000, seed=42)
        assert pc_3d <= self._FACTOR * pc_2d, (
            f"3D ({pc_3d:.3e}) > {self._FACTOR}× 2D ({pc_2d:.3e})"
        )
        assert pc_3d >= pc_2d / self._FACTOR, (
            f"3D ({pc_3d:.3e}) < 2D ({pc_2d:.3e}) / {self._FACTOR}"
        )

    def test_head_on_scenario(self, head_on_tca, head_on_covs):
        cov1, cov2 = head_on_covs
        self._check(head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
                    head_on_tca["epoch"])

    def test_high_pc_crossing_scenario(self, high_pc_crossing_tca, high_pc_crossing_covs):
        cov1, cov2 = high_pc_crossing_covs
        self._check(high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"],
                    cov1, cov2, high_pc_crossing_tca["epoch"])

    def test_head_on_from_fixtures(self, head_on_mc3d, head_on_mc):
        """Cross-check session fixtures against each other."""
        pc_2d, _, _ = head_on_mc
        pc_3d, _, _ = head_on_mc3d
        if pc_2d < self._MIN_PC:
            pytest.skip(f"2D Pc={pc_2d:.1e} too small for comparison")
        assert pc_3d <= self._FACTOR * pc_2d
        assert pc_3d >= pc_2d / self._FACTOR

    def test_near_miss_from_fixtures(self, near_miss_mc3d, near_miss_mc):
        """Near-miss (slow v_rel=15 m/s): 3D should agree with 2D within factor-of-5."""
        pc_2d, _, _ = near_miss_mc
        pc_3d, _, _ = near_miss_mc3d
        if pc_2d < self._MIN_PC:
            pytest.skip(f"2D Pc={pc_2d:.1e} too small for comparison")
        assert pc_3d <= self._FACTOR * pc_2d
        assert pc_3d >= pc_2d / self._FACTOR

    def test_crossing_from_fixtures(self, crossing_mc3d, crossing_mc):
        """Crossing (v_rel=15 m/s): skip if 2D Pc too low for 3D to hit at N=10k."""
        pc_2d, _, _ = crossing_mc
        pc_3d, _, _ = crossing_mc3d
        if pc_2d < self._MIN_PC:
            pytest.skip(f"2D Pc={pc_2d:.1e} too small for 3D comparison at N=10k")
        assert pc_3d <= self._FACTOR * pc_2d
        assert pc_3d >= pc_2d / self._FACTOR

    def test_overtaking_from_fixtures(self, overtaking_mc3d, overtaking_tca, overtaking_covs):
        """Overtaking (v_rel=50 m/s): skip if 2D Pc too low; documents 3D result."""
        cov1, cov2 = overtaking_covs
        pc_2d, _, _ = monte_carlo_pc(
            overtaking_tca["sc1"], overtaking_tca["sc2"], cov1, cov2,
            hard_body_radius=10.0, n_samples=100_000, seed=42,
        )
        pc_3d, _, _ = overtaking_mc3d
        if pc_2d < self._MIN_PC:
            pytest.skip(f"Overtaking 2D Pc={pc_2d:.1e} too small for 3D comparison at N=10k")
        assert pc_3d <= self._FACTOR * pc_2d
        assert pc_3d >= pc_2d / self._FACTOR


# ---------------------------------------------------------------------------
# TestMC3DProperties
# ---------------------------------------------------------------------------

class TestMC3DProperties:
    """Physical-sanity and monotonicity invariants."""

    def test_pc_increases_as_hbr_increases(self, head_on_tca, head_on_covs):
        """Larger HBR → higher or equal Pc."""
        cov1, cov2 = head_on_covs
        pc_small, _, _ = monte_carlo_3d_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            head_on_tca["epoch"], hard_body_radius=5.0,
            n_samples=5_000, seed=42,
        )
        pc_large, _, _ = monte_carlo_3d_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            head_on_tca["epoch"], hard_body_radius=20.0,
            n_samples=5_000, seed=42,
        )
        assert pc_large >= pc_small, (
            f"HBR=5→{pc_small:.2e}, HBR=20→{pc_large:.2e}: should not decrease"
        )

    def test_tighter_cov_changes_pc(self, head_on_tca):
        """MC-3D responds to covariance changes (not frozen at a constant value)."""
        s1, s2 = head_on_tca["sc1"], head_on_tca["sc2"]
        epoch = head_on_tca["epoch"]
        cov_tight, _ = generate_covariances(s1, s2, pos_std_rtn=(10, 50, 5),
                                              vel_std_rtn=(0.01, 0.05, 0.005))
        cov_loose, _ = generate_covariances(s1, s2, pos_std_rtn=(500, 2500, 250),
                                              vel_std_rtn=(0.5, 2.5, 0.25))
        pc_tight, _, _ = monte_carlo_3d_pc(s1, s2, cov_tight, cov_tight, epoch,
                                            10.0, n_samples=5_000, seed=42)
        pc_loose, _, _ = monte_carlo_3d_pc(s1, s2, cov_loose, cov_loose, epoch,
                                            10.0, n_samples=5_000, seed=42)
        assert pc_tight != pc_loose, "Pc should differ for tight vs loose covariance"


# ---------------------------------------------------------------------------
# TestMC3DDegenerate
# ---------------------------------------------------------------------------

class TestMC3DDegenerate:
    """Input-validation edge cases."""

    def test_invalid_n_samples_raises(self, head_on_tca, head_on_covs):
        cov1, cov2 = head_on_covs
        with pytest.raises(ValueError, match="n_samples"):
            monte_carlo_3d_pc(
                head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
                head_on_tca["epoch"], hard_body_radius=10.0, n_samples=0,
            )

    def test_invalid_hbr_raises(self, head_on_tca, head_on_covs):
        cov1, cov2 = head_on_covs
        with pytest.raises(ValueError, match="hard_body_radius"):
            monte_carlo_3d_pc(
                head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
                head_on_tca["epoch"], hard_body_radius=-1.0,
            )
