"""
test_chan1997.py

Tests for the Chan (1997) series-expansion probability-of-collision method.

Coverage
--------
TestChanPcVsFowler      — agrees with Fowler to within 1% for all scenarios
TestChanPcProperties    — Pc in [0,1], monotonicity, symmetry
TestChanPcLimits        — near-zero Pc, near-one Pc, large-miss limit
TestChanPcDegenerate    — ValueError on zero relative speed
"""

import numpy as np
import pytest

from collision.chan1997 import chan_pc
from collision.fowler import fowler_pc
from collision.covariance import generate_covariances


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _identity_states():
    """Two spacecraft with a 100 m radial separation, 500 m/s head-on closing."""
    sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    sc2 = np.array([7.0e6 + 100.0, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0])
    return sc1, sc2


def _small_isotropic_cov(sigma_pos=10.0, sigma_vel=0.01):
    var = np.array([sigma_pos**2] * 3 + [sigma_vel**2] * 3)
    return np.diag(var)


def _large_isotropic_cov(sigma_pos=1e4):
    var = np.array([sigma_pos**2] * 3 + [1.0] * 3)
    return np.diag(var)


# ---------------------------------------------------------------------------
# TestChanPcVsFowler
# ---------------------------------------------------------------------------

class TestChanPcVsFowler:
    """Chan (1997) must agree with Fowler (1993) to within 1% for all scenarios."""

    _HBR = 10.0
    _TOL = 0.01   # 1 % relative tolerance

    def _check(self, sc1, sc2, cov1, cov2):
        pc_f = fowler_pc(sc1, sc2, cov1, cov2, self._HBR)
        pc_c = chan_pc(sc1, sc2, cov1, cov2, self._HBR)
        if pc_f < 1e-20:
            # Both should be essentially zero
            assert pc_c < 1e-15, (
                f"Fowler={pc_f:.2e} ≈ 0 but Chan={pc_c:.2e}"
            )
        else:
            rel_err = abs(pc_c - pc_f) / pc_f
            assert rel_err <= self._TOL, (
                f"Chan ({pc_c:.4e}) vs Fowler ({pc_f:.4e}): "
                f"relative error {rel_err:.2%} > {self._TOL:.0%}"
            )

    def test_crossing_scenario(self, crossing_tca, crossing_covs):
        cov1, cov2 = crossing_covs
        self._check(crossing_tca["sc1"], crossing_tca["sc2"], cov1, cov2)

    def test_head_on_scenario(self, head_on_tca, head_on_covs):
        cov1, cov2 = head_on_covs
        self._check(head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2)

    def test_overtaking_scenario(self, overtaking_tca, overtaking_covs):
        cov1, cov2 = overtaking_covs
        self._check(overtaking_tca["sc1"], overtaking_tca["sc2"], cov1, cov2)

    def test_high_pc_crossing_scenario(self, high_pc_crossing_tca, high_pc_crossing_covs):
        cov1, cov2 = high_pc_crossing_covs
        self._check(
            high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"], cov1, cov2
        )

    def test_near_miss_scenario(self, near_miss_scenario, near_miss_covs):
        cov1, cov2 = near_miss_covs
        self._check(
            near_miss_scenario["sc1_eci_tca"],
            near_miss_scenario["sc2_eci_tca"],
            cov1, cov2,
        )

    @pytest.mark.parametrize("sigma_pos", [50.0, 100.0, 500.0])
    def test_isotropic_cov_various_sigma(self, crossing_tca, sigma_pos):
        """Chan and Fowler should agree for isotropic covariance at various σ."""
        cov = _small_isotropic_cov(sigma_pos=sigma_pos)
        self._check(crossing_tca["sc1"], crossing_tca["sc2"], cov, cov)

    @pytest.mark.parametrize("hbr", [5.0, 10.0, 20.0])
    def test_various_hbr(self, head_on_tca, head_on_covs, hbr):
        """
        Chan and Fowler should agree across typical LEO hard-body radii (5–20 m).

        Very large HBR (> σ) is outside the operationally relevant regime and
        can increase the approximation error above 1%; those values are not
        tested here.
        """
        cov1, cov2 = head_on_covs
        pc_f = fowler_pc(head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2, hbr)
        pc_c = chan_pc(head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2, hbr)
        rel_err = abs(pc_c - pc_f) / pc_f
        assert rel_err <= self._TOL, (
            f"HBR={hbr}: Chan {pc_c:.4e} vs Fowler {pc_f:.4e}, err={rel_err:.2%}"
        )


# ---------------------------------------------------------------------------
# TestChanPcProperties
# ---------------------------------------------------------------------------

class TestChanPcProperties:
    """Basic invariants that any correct Pc implementation must satisfy."""

    def test_pc_in_unit_interval(self, crossing_tca, crossing_covs):
        cov1, cov2 = crossing_covs
        pc = chan_pc(crossing_tca["sc1"], crossing_tca["sc2"], cov1, cov2,
                     hard_body_radius=10.0)
        assert 0.0 <= pc <= 1.0

    def test_pc_returns_float(self, crossing_tca, crossing_covs):
        cov1, cov2 = crossing_covs
        pc = chan_pc(crossing_tca["sc1"], crossing_tca["sc2"], cov1, cov2, 10.0)
        assert isinstance(pc, float)

    def test_pc_increases_as_miss_decreases(self):
        """Pc should rise as SC2 moves closer to SC1."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
        cov = _small_isotropic_cov(sigma_pos=100.0)
        hbr = 10.0

        sc2_far  = np.array([7.0e6 + 500.0, 0.0, 0.0, 0.0, 7000.0, 0.0])
        sc2_near = np.array([7.0e6 +  50.0, 0.0, 0.0, 0.0, 7000.0, 0.0])

        pc_far  = chan_pc(sc1, sc2_far,  cov, cov, hbr)
        pc_near = chan_pc(sc1, sc2_near, cov, cov, hbr)
        assert pc_near > pc_far

    def test_pc_increases_as_hbr_increases(self):
        """Larger HBR → larger cross-section → higher Pc."""
        sc1, sc2 = _identity_states()
        cov = _small_isotropic_cov(sigma_pos=100.0)
        pc_small = chan_pc(sc1, sc2, cov, cov, hard_body_radius=5.0)
        pc_large = chan_pc(sc1, sc2, cov, cov, hard_body_radius=50.0)
        assert pc_large > pc_small

    def test_symmetric_swap_sc1_sc2(self, head_on_tca, head_on_covs):
        """Swapping SC1 and SC2 must give the same Pc."""
        cov1, cov2 = head_on_covs
        s1, s2 = head_on_tca["sc1"], head_on_tca["sc2"]
        pc_12 = chan_pc(s1, s2, cov1, cov2, hard_body_radius=10.0)
        pc_21 = chan_pc(s2, s1, cov2, cov1, hard_body_radius=10.0)
        assert abs(pc_12 - pc_21) < 1e-12, (
            f"Swap asymmetry: {pc_12:.4e} vs {pc_21:.4e}"
        )

    @pytest.mark.parametrize("scenario_name,tca_key,cov_key", [
        ("crossing",   "crossing_tca",           "crossing_covs"),
        ("head_on",    "head_on_tca",             "head_on_covs"),
        ("overtaking", "overtaking_tca",          "overtaking_covs"),
        ("high_pc",    "high_pc_crossing_tca",    "high_pc_crossing_covs"),
    ])
    def test_pc_increases_with_hbr(self, scenario_name, tca_key, cov_key, request):
        """Pc must be monotone-increasing in HBR for every scenario."""
        tca  = request.getfixturevalue(tca_key)
        covs = request.getfixturevalue(cov_key)
        cov1, cov2 = covs
        pc_small = chan_pc(tca["sc1"], tca["sc2"], cov1, cov2, hard_body_radius=5.0)
        pc_large = chan_pc(tca["sc1"], tca["sc2"], cov1, cov2, hard_body_radius=50.0)
        assert pc_large >= pc_small, (
            f"{scenario_name}: Pc should not decrease as HBR grows "
            f"(HBR=5 → {pc_small:.2e}, HBR=50 → {pc_large:.2e})"
        )


# ---------------------------------------------------------------------------
# TestChanPcLimits
# ---------------------------------------------------------------------------

class TestChanPcLimits:
    """Extreme-value and stability tests."""

    def test_near_one_zero_miss_large_hbr(self):
        """Zero miss + HBR >> σ → Pc ≈ 1."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0,       0.0])
        sc2 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0])
        cov = _small_isotropic_cov(sigma_pos=1.0)
        pc = chan_pc(sc1, sc2, cov, cov, hard_body_radius=20.0)
        assert pc > 0.99, f"Expected Pc ≈ 1, got {pc:.4f}"

    def test_near_zero_large_miss_tiny_cov(self):
        """Large miss >> HBR with tight σ → Pc essentially 0."""
        sc1 = np.array([7.0e6,         0.0, 0.0, 0.0, 7500.0,     0.0])
        sc2 = np.array([7.0e6 + 100e3, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0])
        cov = _small_isotropic_cov(sigma_pos=10.0)
        pc = chan_pc(sc1, sc2, cov, cov, hard_body_radius=10.0)
        assert pc < 1e-10, f"Expected Pc ≈ 0 for large miss, got {pc:.2e}"

    def test_crossing_pc_is_low(self, crossing_tca, crossing_covs):
        """
        Crossing (500 m miss, 500 m/s): Pc ~ 3.9e-4 with default covariances.
        Chan must produce a non-trivial, positive result (not underflow to 0).
        """
        cov1, cov2 = crossing_covs
        pc = chan_pc(crossing_tca["sc1"], crossing_tca["sc2"], cov1, cov2,
                     hard_body_radius=10.0)
        assert 0.0 < pc < 1e-1, f"Crossing Pc={pc:.2e} should be in (0, 0.1)"

    def test_near_miss_pc_magnitude(self, near_miss_scenario, near_miss_covs):
        """Near-miss (10 m): Pc ~ 1e-3, should be in [1e-5, 1e-1]."""
        cov1, cov2 = near_miss_covs
        pc = chan_pc(
            near_miss_scenario["sc1_eci_tca"],
            near_miss_scenario["sc2_eci_tca"],
            cov1, cov2, hard_body_radius=10.0,
        )
        assert 1e-5 < pc < 1e-1, f"Near-miss Pc={pc:.2e} outside [1e-5, 1e-1]"

    def test_head_on_pc_magnitude(self, head_on_tca, head_on_covs):
        """Head-on: Pc ~ 5e-3, should be in [1e-6, 1e-1]."""
        cov1, cov2 = head_on_covs
        pc = chan_pc(head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2,
                     hard_body_radius=10.0)
        assert 1e-6 < pc < 1e-1, f"Head-on Pc={pc:.2e} outside [1e-6, 1e-1]"

    def test_isotropic_limit(self):
        """For isotropic cov the Chan formula reduces to ncx2.cdf exactly."""
        from scipy.stats import ncx2
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
        sc2 = np.array([7.0e6 + 200.0, 0.0, 0.0, 0.0, 7500.0 - 500.0, 0.0])
        sigma = 100.0
        cov = np.diag([sigma**2] * 3 + [0.01**2] * 3)
        pc_chan = chan_pc(sc1, sc2, cov, cov, hard_body_radius=10.0)
        # Should be non-trivial and match Fowler
        pc_fowl = fowler_pc(sc1, sc2, cov, cov, hard_body_radius=10.0)
        rel_err = abs(pc_chan - pc_fowl) / max(pc_fowl, 1e-30)
        assert rel_err < 0.01, f"Isotropic: Chan={pc_chan:.4e}, Fowler={pc_fowl:.4e}"

    def test_pc_valid_at_all_covariance_levels(self, head_on_tca):
        """Pc in [0,1] at every covariance magnitude from tight to very loose."""
        tight      = dict(pos_std_rtn=(10,   50,   5),  vel_std_rtn=(0.01, 0.05, 0.005))
        default    = dict(pos_std_rtn=(100,  500,  50), vel_std_rtn=(0.1,  0.5,  0.05))
        loose      = dict(pos_std_rtn=(300,  1500, 150),vel_std_rtn=(0.3,  1.5,  0.15))
        very_loose = dict(pos_std_rtn=(1000, 5000, 500),vel_std_rtn=(1.0,  5.0,  0.5))
        s1, s2 = head_on_tca["sc1"], head_on_tca["sc2"]
        for kw in [tight, default, loose, very_loose]:
            cov1, cov2 = generate_covariances(s1, s2, **kw)
            pc = chan_pc(s1, s2, cov1, cov2, hard_body_radius=10.0)
            assert 0.0 <= pc <= 1.0, f"Pc={pc} out of [0,1] for {kw}"


# ---------------------------------------------------------------------------
# TestChanPcDegenerate
# ---------------------------------------------------------------------------

class TestChanPcDegenerate:
    def test_zero_relative_speed_raises(self):
        """chan_pc must raise ValueError when v_rel == 0 (no encounter plane)."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 7500.0, 0.0, 0.0])
        sc2 = np.array([7.0e6 + 100.0, 0.0, 0.0, 7500.0, 0.0, 0.0])
        cov = _small_isotropic_cov()
        with pytest.raises(ValueError, match="zero"):
            chan_pc(sc1, sc2, cov, cov, hard_body_radius=10.0)
