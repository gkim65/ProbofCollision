"""
test_monte_carlo.py

Tests for the Monte Carlo probability-of-collision baseline.

Coverage
--------
TestMonteCarloBasic         — return type, range, reproducibility, CI width
TestMonteCarloPcVsChan      — loose agreement with Chan (2–3× tolerance)
TestMonteCarloProperties    — monotonicity, larger N narrows CI
TestMonteCarloDegenerate    — ValueError on bad inputs
"""

import numpy as np
import pytest

from collision.monte_carlo import monte_carlo_pc
from collision.chan1997 import chan_pc
from collision.covariance import generate_covariances


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _identity_states():
    """Two spacecraft with 100 m radial separation, 500 m/s closing speed."""
    sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    sc2 = np.array([7.0e6 + 100.0, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0])
    return sc1, sc2


def _isotropic_cov(sigma_pos=100.0):
    var = np.array([sigma_pos**2] * 3 + [0.1**2] * 3)
    return np.diag(var)


# ---------------------------------------------------------------------------
# TestMonteCarloBasic
# ---------------------------------------------------------------------------

class TestMonteCarloBasic:
    """Return-type, range, and reproducibility invariants."""

    def test_returns_tuple_of_three(self, head_on_mc):
        assert len(head_on_mc) == 3

    def test_all_elements_are_float(self, head_on_mc):
        pc, ci_low, ci_high = head_on_mc
        assert isinstance(pc, float)
        assert isinstance(ci_low, float)
        assert isinstance(ci_high, float)

    def test_pc_in_unit_interval(self, head_on_mc):
        pc, ci_low, ci_high = head_on_mc
        assert 0.0 <= pc <= 1.0

    def test_ci_ordered(self, head_on_mc):
        pc, ci_low, ci_high = head_on_mc
        assert ci_low <= pc <= ci_high

    def test_ci_bounds_in_unit_interval(self, head_on_mc):
        _, ci_low, ci_high = head_on_mc
        assert 0.0 <= ci_low <= 1.0
        assert 0.0 <= ci_high <= 1.0

    def test_fixed_seed_reproducible(self, head_on_tca, head_on_covs):
        """Same seed must produce identical results."""
        cov1, cov2 = head_on_covs
        r1 = monte_carlo_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            hard_body_radius=10.0, n_samples=100_000, seed=0,
        )
        r2 = monte_carlo_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            hard_body_radius=10.0, n_samples=100_000, seed=0,
        )
        assert r1 == r2

    def test_different_seeds_differ(self, head_on_tca, head_on_covs):
        """Different seeds should (almost surely) produce different estimates."""
        cov1, cov2 = head_on_covs
        pc_a, _, _ = monte_carlo_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            hard_body_radius=10.0, n_samples=100_000, seed=1,
        )
        pc_b, _, _ = monte_carlo_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            hard_body_radius=10.0, n_samples=100_000, seed=2,
        )
        # Very unlikely to be exactly equal at 100k samples
        assert pc_a != pc_b

    def test_larger_n_narrows_ci(self, head_on_tca, head_on_covs):
        """CI half-width should shrink as N grows."""
        cov1, cov2 = head_on_covs
        _, lo_small, hi_small = monte_carlo_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            hard_body_radius=10.0, n_samples=10_000, seed=42,
        )
        _, lo_large, hi_large = monte_carlo_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            hard_body_radius=10.0, n_samples=1_000_000, seed=42,
        )
        width_small = hi_small - lo_small
        width_large = hi_large - lo_large
        assert width_large < width_small, (
            f"Larger N should narrow CI: N=10k width={width_small:.2e}, "
            f"N=1M width={width_large:.2e}"
        )

    def test_zero_miss_high_pc(self):
        """Zero-miss + large HBR relative to sigma → Pc close to 1."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
        sc2 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0])
        cov = _isotropic_cov(sigma_pos=1.0)
        pc, _, _ = monte_carlo_pc(sc1, sc2, cov, cov, hard_body_radius=20.0,
                                   n_samples=1_000_000, seed=42)
        assert pc > 0.99, f"Expected Pc ≈ 1 for zero miss / tiny cov, got {pc:.4f}"

    def test_large_miss_near_zero_pc(self):
        """100 km miss with tight covariance → Pc = 0 in any reasonable N."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
        sc2 = np.array([7.0e6 + 100e3, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0])
        cov = _isotropic_cov(sigma_pos=10.0)
        pc, _, _ = monte_carlo_pc(sc1, sc2, cov, cov, hard_body_radius=10.0,
                                   n_samples=1_000_000, seed=42)
        assert pc == 0.0, f"Expected Pc=0 for 100 km miss, got {pc:.2e}"


# ---------------------------------------------------------------------------
# TestMonteCarloPcVsChan
# ---------------------------------------------------------------------------

class TestMonteCarloPcVsChan:
    """
    Monte Carlo and Chan (1997) should agree to within a factor of 3 for
    scenarios where Pc is large enough to get a reliable sample count.

    At N=1e6, the relative standard deviation is ~1/sqrt(N*pc), so for
    Pc ~ 1e-3 it is ~3%.  We use a loose 3× tolerance to accommodate
    sampling noise without needing importance sampling.
    """

    _FACTOR = 3.0   # multiplicative tolerance
    _MIN_PC = 1e-5  # skip comparison if analytic Pc is too small for MC

    def _check(self, sc1, sc2, cov1, cov2, hbr=10.0, n=1_000_000):
        pc_chan = chan_pc(sc1, sc2, cov1, cov2, hbr)
        pc_mc, _, _ = monte_carlo_pc(sc1, sc2, cov1, cov2, hbr,
                                      n_samples=n, seed=42)
        if pc_chan < self._MIN_PC:
            pytest.skip(f"Chan Pc={pc_chan:.1e} too small for MC at N={n}")
        # Check that MC is within factor-of-3 of Chan in both directions
        assert pc_mc <= self._FACTOR * pc_chan, (
            f"MC ({pc_mc:.3e}) > {self._FACTOR}× Chan ({pc_chan:.3e})"
        )
        assert pc_mc >= pc_chan / self._FACTOR, (
            f"MC ({pc_mc:.3e}) < Chan ({pc_chan:.3e}) / {self._FACTOR}"
        )

    def test_head_on_scenario(self, head_on_tca, head_on_covs):
        cov1, cov2 = head_on_covs
        self._check(head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2)

    def test_near_miss_scenario(self, near_miss_scenario, near_miss_covs):
        cov1, cov2 = near_miss_covs
        self._check(
            near_miss_scenario["sc1_eci_tca"],
            near_miss_scenario["sc2_eci_tca"],
            cov1, cov2,
        )

    def test_high_pc_crossing_scenario(self, high_pc_crossing_tca, high_pc_crossing_covs):
        cov1, cov2 = high_pc_crossing_covs
        self._check(high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"],
                    cov1, cov2)

    def test_head_on_mc_vs_chan_from_fixture(self, head_on_mc, head_on_tca, head_on_covs):
        """Cross-check the session fixture against the Chan value."""
        cov1, cov2 = head_on_covs
        pc_chan = chan_pc(head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2, 10.0)
        pc_mc, _, _ = head_on_mc
        assert pc_mc <= self._FACTOR * pc_chan
        assert pc_mc >= pc_chan / self._FACTOR


# ---------------------------------------------------------------------------
# TestMonteCarloProperties
# ---------------------------------------------------------------------------

class TestMonteCarloProperties:
    """Physical-sanity and monotonicity invariants."""

    def test_pc_increases_as_miss_decreases(self):
        """Moving SC2 closer should increase Pc."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
        cov = _isotropic_cov(sigma_pos=100.0)

        sc2_far  = np.array([7.0e6 + 500.0, 0.0, 0.0, 0.0, 7000.0, 0.0])
        sc2_near = np.array([7.0e6 +  50.0, 0.0, 0.0, 0.0, 7000.0, 0.0])

        pc_far, _, _  = monte_carlo_pc(sc1, sc2_far,  cov, cov, 10.0, seed=42)
        pc_near, _, _ = monte_carlo_pc(sc1, sc2_near, cov, cov, 10.0, seed=42)
        assert pc_near > pc_far, (
            f"Closer miss should give higher Pc: near={pc_near:.2e}, far={pc_far:.2e}"
        )

    def test_pc_increases_as_hbr_increases(self, head_on_tca, head_on_covs):
        """Larger HBR → larger cross-section → higher Pc."""
        cov1, cov2 = head_on_covs
        pc_small, _, _ = monte_carlo_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            hard_body_radius=5.0, n_samples=1_000_000, seed=42,
        )
        pc_large, _, _ = monte_carlo_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
            hard_body_radius=20.0, n_samples=1_000_000, seed=42,
        )
        assert pc_large >= pc_small, (
            f"HBR=5 → {pc_small:.2e}, HBR=20 → {pc_large:.2e}: should not decrease"
        )

    def test_swap_sc1_sc2_symmetric(self, head_on_tca, head_on_covs):
        """Swapping SC1/SC2 must give identical Pc (relative position changes sign,
        but the covariance and HBR sphere are symmetric)."""
        cov1, cov2 = head_on_covs
        s1, s2 = head_on_tca["sc1"], head_on_tca["sc2"]
        pc_12, _, _ = monte_carlo_pc(s1, s2, cov1, cov2, 10.0,
                                      n_samples=1_000_000, seed=42)
        pc_21, _, _ = monte_carlo_pc(s2, s1, cov2, cov1, 10.0,
                                      n_samples=1_000_000, seed=42)
        # Exact equality because same seed samples the same RNG sequence;
        # mean differs only in sign, but |r_sample| < HBR is sign-invariant
        # only if the distribution is symmetric.  Use a tolerance of 1 count.
        tol = 1.0 / 1_000_000
        assert abs(pc_12 - pc_21) <= tol, (
            f"Swap asymmetry: {pc_12:.4e} vs {pc_21:.4e}"
        )

    def test_tighter_cov_can_change_pc(self, head_on_tca):
        """Verify MC responds to covariance changes (not frozen)."""
        s1, s2 = head_on_tca["sc1"], head_on_tca["sc2"]
        cov_tight, _ = generate_covariances(s1, s2, pos_std_rtn=(10, 50, 5),
                                              vel_std_rtn=(0.01, 0.05, 0.005))
        cov_loose, _ = generate_covariances(s1, s2, pos_std_rtn=(500, 2500, 250),
                                              vel_std_rtn=(0.5, 2.5, 0.25))
        pc_tight, _, _ = monte_carlo_pc(s1, s2, cov_tight, cov_tight, 10.0,
                                         n_samples=1_000_000, seed=42)
        pc_loose, _, _ = monte_carlo_pc(s1, s2, cov_loose, cov_loose, 10.0,
                                         n_samples=1_000_000, seed=42)
        # They should differ; we don't prescribe which is larger (geometry-dependent)
        assert pc_tight != pc_loose, "Pc should differ for tight vs loose covariance"


# ---------------------------------------------------------------------------
# TestMonteCarloDegenerate
# ---------------------------------------------------------------------------

class TestMonteCarloDegenerate:
    """Input-validation edge cases."""

    def test_invalid_n_samples_raises(self):
        sc1, sc2 = _identity_states()
        cov = _isotropic_cov()
        with pytest.raises(ValueError, match="n_samples"):
            monte_carlo_pc(sc1, sc2, cov, cov, hard_body_radius=10.0, n_samples=0)

    def test_invalid_hbr_raises(self):
        sc1, sc2 = _identity_states()
        cov = _isotropic_cov()
        with pytest.raises(ValueError, match="hard_body_radius"):
            monte_carlo_pc(sc1, sc2, cov, cov, hard_body_radius=-1.0)

    def test_n_equals_one(self):
        """n_samples=1 is valid — returns 0.0 or 1.0."""
        sc1, sc2 = _identity_states()
        cov = _isotropic_cov()
        pc, ci_low, ci_high = monte_carlo_pc(sc1, sc2, cov, cov,
                                              hard_body_radius=10.0,
                                              n_samples=1, seed=0)
        assert pc in (0.0, 1.0)
        assert 0.0 <= ci_low <= ci_high <= 1.0
