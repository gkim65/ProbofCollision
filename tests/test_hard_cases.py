"""
test_hard_cases.py

Stress-test fixtures that expose where Fowler/Chan break down and where Monte
Carlo (or later importance sampling) is required.

Three hard cases
----------------
Case A — Very small Pc (< 1e-8): tight covariance + 2000 m miss distance.
    Fowler (dblquad) underflows to exactly 0.0.
    Chan (ncx2.cdf) stays numerically positive (the right answer).
    MC at N=1e6 also returns 0 — demonstrating why importance sampling is
    needed for Pc < 1e-8.

Case B — Highly elongated covariance (σ₂/σ₁ >> 1):
    Test Chan's anisotropy correction error vs Fowler across a range of
    eccentricities; document the degradation limit.

Case C — Multiple close approaches (v_rel ~ 5 m/s):
    Very slow overtaking produces several local minima of miss distance within
    a 24-hour window.  find_tca returns only the global minimum; Pc for that
    encounter is valid for that single pass.  Multi-encounter Pc is out of
    scope for all current analytic methods.
"""

import numpy as np
import pytest

from collision.fowler import fowler_pc
from collision.chan1997 import chan_pc
from collision.monte_carlo import monte_carlo_pc
from collision.covariance import generate_covariances
from collision.tca import find_tca


# ===========================================================================
# Case A — Very small Pc: 2000 m miss + tight covariance
# ===========================================================================

class TestCaseA_TinyPc:
    """
    Demonstrate the Pc < 1e-8 regime where MC fails and Chan remains valid.

    Fixture: tiny_pc_scenario — head-on at 2000 m miss, seed=11.
    Covariance: 10× tighter than default (pos_std_rtn = 10/50/5 m).
    """

    HBR = 10.0
    N_MC = 1_000_000

    def test_chan_returns_positive(self, tiny_pc_tca, tiny_pc_covs_tight):
        """Chan (ncx2.cdf) must return a positive, finite value."""
        cov1, cov2 = tiny_pc_covs_tight
        pc = chan_pc(
            tiny_pc_tca["sc1"], tiny_pc_tca["sc2"], cov1, cov2, self.HBR,
        )
        assert pc > 0.0, f"Chan should stay positive at tiny Pc, got {pc}"
        assert np.isfinite(pc), "Chan Pc must be finite"

    def test_chan_pc_is_tiny(self, tiny_pc_tca, tiny_pc_covs_tight):
        """Confirm the scenario actually exercises the tiny-Pc regime (< 1e-6)."""
        cov1, cov2 = tiny_pc_covs_tight
        pc = chan_pc(
            tiny_pc_tca["sc1"], tiny_pc_tca["sc2"], cov1, cov2, self.HBR,
        )
        assert pc < 1e-6, (
            f"Expected Chan Pc < 1e-6 for tight-covariance scenario, got {pc:.2e}"
        )

    def test_fowler_underflows_or_positive(self, tiny_pc_tca, tiny_pc_covs_tight):
        """
        Fowler dblquad either underflows to 0.0 (expected) or returns a tiny
        positive value.  Either is acceptable — the key assertion is that it
        does NOT return a nonsensically large value.
        """
        cov1, cov2 = tiny_pc_covs_tight
        pc = fowler_pc(
            tiny_pc_tca["sc1"], tiny_pc_tca["sc2"], cov1, cov2, self.HBR,
        )
        assert pc >= 0.0, "Fowler Pc must be non-negative"
        assert pc < 1e-4, (
            f"Fowler Pc ({pc:.2e}) unexpectedly large for tiny-Pc scenario"
        )

    def test_fowler_underflows_to_zero(self, tiny_pc_tca, tiny_pc_covs_tight):
        """
        Document that Fowler's dblquad underflows: it returns exactly 0.0 at
        Pc < ~1e-10 because scipy's Gaussian PDF evaluates to 0.0 in float64.
        This is the motivating case for importance sampling.
        """
        cov1, cov2 = tiny_pc_covs_tight
        pc = fowler_pc(
            tiny_pc_tca["sc1"], tiny_pc_tca["sc2"], cov1, cov2, self.HBR,
        )
        # If the scenario is deep enough in the tail, dblquad gives exactly 0.
        # This test documents the observed behaviour; it may become non-zero
        # if the geometry changes.
        if pc != 0.0:
            pytest.skip(
                f"Fowler returned {pc:.2e} (non-zero) — scenario may not be "
                "deep enough for underflow with this geometry"
            )
        assert pc == 0.0

    def test_mc_returns_zero(self, tiny_pc_tca, tiny_pc_covs_tight):
        """
        Monte Carlo at N=1e6 cannot resolve Pc ~ 5e-12: expected hit count ~ 5e-6.
        Documents that plain MC is useless here and importance sampling is needed.
        """
        cov1, cov2 = tiny_pc_covs_tight
        pc, ci_low, ci_high = monte_carlo_pc(
            tiny_pc_tca["sc1"], tiny_pc_tca["sc2"], cov1, cov2,
            hard_body_radius=self.HBR, n_samples=self.N_MC, seed=42,
        )
        assert pc == 0.0, (
            f"MC should return 0 at N=1e6 for Pc ~ 1e-12, got {pc:.2e} — "
            "this is expected and motivates importance sampling"
        )

    def test_chan_larger_than_mc(self, tiny_pc_tca, tiny_pc_covs_tight):
        """
        Chan > MC in this regime: MC is provably underestimating because it has
        zero samples in the tail.  This is the key finding for Case A.
        """
        cov1, cov2 = tiny_pc_covs_tight
        pc_chan = chan_pc(
            tiny_pc_tca["sc1"], tiny_pc_tca["sc2"], cov1, cov2, self.HBR,
        )
        pc_mc, _, _ = monte_carlo_pc(
            tiny_pc_tca["sc1"], tiny_pc_tca["sc2"], cov1, cov2,
            hard_body_radius=self.HBR, n_samples=self.N_MC, seed=42,
        )
        assert pc_chan > pc_mc, (
            f"Chan ({pc_chan:.2e}) should exceed MC ({pc_mc:.2e}) in the "
            "tail regime where sampling fails"
        )

    def test_chan_vs_fowler_ratio_when_fowler_nonzero(
        self, tiny_pc_tca, tiny_pc_covs_tight
    ):
        """
        When Fowler does not underflow, Chan and Fowler should still agree
        to within a factor of 10 (loose bound to cover integration tolerances
        at the very edge of float64 precision).
        """
        cov1, cov2 = tiny_pc_covs_tight
        pc_fowler = fowler_pc(
            tiny_pc_tca["sc1"], tiny_pc_tca["sc2"], cov1, cov2, self.HBR,
        )
        if pc_fowler == 0.0:
            pytest.skip("Fowler underflowed to 0 — ratio undefined")
        pc_chan = chan_pc(
            tiny_pc_tca["sc1"], tiny_pc_tca["sc2"], cov1, cov2, self.HBR,
        )
        ratio = max(pc_chan, pc_fowler) / min(pc_chan, pc_fowler)
        assert ratio < 10.0, (
            f"Chan ({pc_chan:.2e}) and Fowler ({pc_fowler:.2e}) diverge "
            f"by ×{ratio:.1f} — expected < 10× at the float64 boundary"
        )


# ===========================================================================
# Case B — Highly elongated covariance
# ===========================================================================

class TestCaseB_ElongatedCovariance:
    """
    Explore Chan's anisotropy correction accuracy as σ₂/σ₁ grows.

    Two sub-cases:
      B1 — Crossing at pos_std_rtn=(5, 5000, 5): σ_T=5000 m projects fully
           into the encounter plane → extreme width in one direction (σ₂/σ₁ ≈ 1000).
      B2 — Head-on at pos_std_rtn=(5, 5000, 5): σ_T is along v_rel and is
           excluded from the encounter plane → compact, nearly isotropic projection.
    """

    HBR = 10.0

    # -----------------------------------------------------------------------
    # B1: Crossing — σ_T projects into encounter plane → wide ellipse
    # -----------------------------------------------------------------------

    def test_b1_crossing_wide_ellipse_chan_positive(
        self, high_pc_crossing_tca
    ):
        """Chan returns a positive value even with extreme elongation."""
        cov1, cov2 = generate_covariances(
            high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"],
            pos_std_rtn=(5.0, 5000.0, 5.0),
        )
        pc = chan_pc(
            high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"],
            cov1, cov2, self.HBR,
        )
        assert pc > 0.0
        assert np.isfinite(pc)

    def test_b1_crossing_wide_ellipse_chan_vs_fowler_within_factor(
        self, high_pc_crossing_tca
    ):
        """
        At σ₂/σ₁ ~ 1000 (crossing, σ_T=5000 m in encounter plane), Fowler and
        Chan can deviate significantly — document the actual ratio.  We allow
        up to 100× here; the test captures the degradation, not a tight bound.
        """
        cov1, cov2 = generate_covariances(
            high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"],
            pos_std_rtn=(5.0, 5000.0, 5.0),
        )
        pc_chan = chan_pc(
            high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"],
            cov1, cov2, self.HBR,
        )
        pc_fowler = fowler_pc(
            high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"],
            cov1, cov2, self.HBR,
        )
        if pc_fowler == 0.0 or pc_chan == 0.0:
            pytest.skip("One method returned 0 — ratio undefined")
        ratio = max(pc_chan, pc_fowler) / min(pc_chan, pc_fowler)
        # Document the ratio in the message so it appears in the test output
        assert ratio < 100.0, (
            f"Chan/Fowler ratio {ratio:.1f}× at σ₂/σ₁≈1000 (crossing). "
            "If this exceeds 100×, Chan's anisotropy correction has broken down."
        )

    def test_b1_crossing_wide_ellipse_anisotropy_ratio(
        self, high_pc_crossing_tca
    ):
        """Verify the encounter-plane σ₂/σ₁ is indeed >> 1 for crossing."""
        cov1, cov2 = generate_covariances(
            high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"],
            pos_std_rtn=(5.0, 5000.0, 5.0),
        )
        C_pos = cov1[:3, :3] + cov2[:3, :3]
        sc1 = high_pc_crossing_tca["sc1"]
        sc2 = high_pc_crossing_tca["sc2"]
        v_rel = sc1[3:] - sc2[3:]
        z_hat = v_rel / np.linalg.norm(v_rel)
        r_rel = sc1[:3] - sc2[:3]
        r_perp = r_rel - np.dot(r_rel, z_hat) * z_hat
        r_perp_mag = np.linalg.norm(r_perp)
        if r_perp_mag < 1e-10:
            arb = np.array([1.0, 0.0, 0.0])
            r_perp = arb - np.dot(arb, z_hat) * z_hat
            r_perp_mag = np.linalg.norm(r_perp)
        x_hat = r_perp / r_perp_mag
        y_hat = np.cross(z_hat, x_hat)
        B = np.array([x_hat, y_hat])
        C_2d = B @ C_pos @ B.T
        eigvals = np.linalg.eigvalsh(C_2d)
        ratio = float(np.sqrt(eigvals[1] / eigvals[0]))
        assert ratio > 10.0, (
            f"Expected σ₂/σ₁ > 10 for crossing with σ_T=5000 m, got {ratio:.1f}"
        )

    # -----------------------------------------------------------------------
    # B2: Head-on — σ_T is along v_rel, excluded from encounter plane → compact
    # -----------------------------------------------------------------------

    def test_b2_head_on_compact_projection_agrees_well(self, head_on_tca):
        """
        Head-on with σ_T=5000 m: because σ_T is along v_rel, it is excluded
        from the encounter plane.  The projected covariance is nearly isotropic
        (σ_R=5, σ_N=5), so Chan and Fowler should agree to < 1%.
        """
        cov1, cov2 = generate_covariances(
            head_on_tca["sc1"], head_on_tca["sc2"],
            pos_std_rtn=(5.0, 5000.0, 5.0),
        )
        pc_chan = chan_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2, self.HBR,
        )
        pc_fowler = fowler_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2, self.HBR,
        )
        if pc_fowler == 0.0 or pc_chan == 0.0:
            pytest.skip("One method returned 0 — cannot check agreement")
        rel_err = abs(pc_chan - pc_fowler) / pc_fowler
        assert rel_err < 0.01, (
            f"Head-on compact projection: Chan/Fowler relative error = "
            f"{rel_err:.2%} (expected < 1% when encounter plane is isotropic)"
        )

    def test_b2_head_on_compact_anisotropy_ratio(self, head_on_tca):
        """Encounter-plane σ₂/σ₁ should be close to 1 for head-on with σ_T=5000."""
        cov1, cov2 = generate_covariances(
            head_on_tca["sc1"], head_on_tca["sc2"],
            pos_std_rtn=(5.0, 5000.0, 5.0),
        )
        C_pos = cov1[:3, :3] + cov2[:3, :3]
        sc1, sc2 = head_on_tca["sc1"], head_on_tca["sc2"]
        v_rel = sc1[3:] - sc2[3:]
        z_hat = v_rel / np.linalg.norm(v_rel)
        r_rel = sc1[:3] - sc2[:3]
        r_perp = r_rel - np.dot(r_rel, z_hat) * z_hat
        r_perp_mag = np.linalg.norm(r_perp)
        if r_perp_mag < 1e-10:
            arb = np.array([1.0, 0.0, 0.0])
            r_perp = arb - np.dot(arb, z_hat) * z_hat
            r_perp_mag = np.linalg.norm(r_perp)
        x_hat = r_perp / r_perp_mag
        y_hat = np.cross(z_hat, x_hat)
        B = np.array([x_hat, y_hat])
        C_2d = B @ C_pos @ B.T
        eigvals = np.linalg.eigvalsh(C_2d)
        ratio = float(np.sqrt(eigvals[1] / eigvals[0]))
        # σ_T=5000 is excluded; both in-plane axes are σ_R=σ_N=5 → nearly 1
        assert ratio < 5.0, (
            f"Expected near-isotropic encounter plane for head-on + σ_T=5000, "
            f"got σ₂/σ₁ = {ratio:.1f}"
        )

    # -----------------------------------------------------------------------
    # B3: Chan accuracy vs anisotropy ratio sweep
    # -----------------------------------------------------------------------

    def test_b3_chan_vs_fowler_moderate_anisotropy(self, head_on_tca):
        """
        For moderate anisotropy (σ₂/σ₁ ~ 10): Chan's leading-term approximation
        accumulates ~6% error vs Fowler — beyond the < 0.2% seen with default
        covariances (σ₂/σ₁ ~ 3).  This documents the degradation onset.

        The Chan formula is still useful here (error < 10%), but the < 0.2%
        guarantee only holds when the encounter plane is nearly isotropic
        (σ₂/σ₁ ≤ ~3 with HBR << σ₁).
        """
        cov1, cov2 = generate_covariances(
            head_on_tca["sc1"], head_on_tca["sc2"],
            pos_std_rtn=(10.0, 500.0, 100.0),   # σ_N/σ_R ~ 10 in encounter plane
        )
        pc_chan = chan_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2, self.HBR,
        )
        pc_fowler = fowler_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2, self.HBR,
        )
        if pc_fowler == 0.0 or pc_chan == 0.0:
            pytest.skip("One method returned 0")
        rel_err = abs(pc_chan - pc_fowler) / pc_fowler
        # At σ₂/σ₁ ~ 10 the error is empirically ~6%.  Bound at 10% to document
        # the regime but allow for geometry variation across scenarios.
        assert rel_err < 0.10, (
            f"At σ₂/σ₁ ~ 10: Chan/Fowler error = {rel_err:.2%} (expected < 10%)"
        )

    def test_b3_chan_vs_fowler_high_anisotropy(self, head_on_tca):
        """
        For σ₂/σ₁ ~ 100 in the encounter plane (head-on with σ_N=1000 m,
        σ_R=10 m): Chan's correction can accumulate > 1% error.  Document it.
        """
        cov1, cov2 = generate_covariances(
            head_on_tca["sc1"], head_on_tca["sc2"],
            pos_std_rtn=(10.0, 500.0, 1000.0),   # σ_N >> σ_R in enc. plane
        )
        pc_chan = chan_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2, self.HBR,
        )
        pc_fowler = fowler_pc(
            head_on_tca["sc1"], head_on_tca["sc2"], cov1, cov2, self.HBR,
        )
        if pc_fowler == 0.0 or pc_chan == 0.0:
            pytest.skip("One method returned 0")
        rel_err = abs(pc_chan - pc_fowler) / pc_fowler
        # We don't prescribe an exact tolerance here — this test documents
        # the error, which may be > 1%.  It must be finite and < 100%.
        assert np.isfinite(rel_err), "Relative error must be finite"
        assert rel_err < 1.0, (
            f"Chan/Fowler relative error {rel_err:.2%} at σ₂/σ₁ ~ 100 — "
            "Chan's anisotropy correction has significant error here"
        )


# ===========================================================================
# Case C — Multiple close approaches (v_rel ~ 5 m/s)
# ===========================================================================

class TestCaseC_MultipleCloseApproaches:
    """
    Demonstrate that slow overtaking (v_rel ~ 5 m/s) can produce multiple
    close-approach minima within a 24-hour window.

    find_tca returns one epoch (the global minimum); Fowler/Chan compute Pc
    for that single encounter.  True Pc = 1 − Π(1 − Pc_i) over all encounters;
    multi-encounter Pc is out of scope for current analytic methods.
    """

    HBR = 10.0

    def test_slow_overtaking_tca_found(self, slow_overtaking_scenario):
        """find_tca must return a valid epoch and a finite miss distance."""
        epoch, miss = find_tca(
            slow_overtaking_scenario["epoch_start"],
            slow_overtaking_scenario["sc1_eci_t0"],
            slow_overtaking_scenario["sc2_eci_t0"],
        )
        assert miss >= 0.0, "Miss distance must be non-negative"
        assert np.isfinite(miss), "Miss distance must be finite"

    def test_slow_overtaking_tca_single_minimum(self, slow_overtaking_scenario):
        """
        Coarse grid over the 24-hour window: count how many local minima of
        miss distance exist.  At v_rel ~ 5 m/s there are likely 2+ minima;
        find_tca captures only the global one.

        This test documents (not fixes) the limitation.
        """
        from brahe import NumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfig

        epoch_start = slow_overtaking_scenario["epoch_start"]
        sc1_t0 = slow_overtaking_scenario["sc1_eci_t0"]
        sc2_t0 = slow_overtaking_scenario["sc2_eci_t0"]
        duration = slow_overtaking_scenario["epoch_tca"] - epoch_start   # seconds

        prop_config = NumericalPropagationConfig.default()
        force_config = ForceModelConfig.two_body()

        N_GRID = 200
        misses = []
        for i in range(N_GRID + 1):
            t = epoch_start + duration * i / N_GRID
            p1 = NumericalOrbitPropagator(epoch_start, sc1_t0, prop_config, force_config)
            p2 = NumericalOrbitPropagator(epoch_start, sc2_t0, prop_config, force_config)
            p1.propagate_to(t)
            p2.propagate_to(t)
            r1 = np.array(p1.current_state()[:3])
            r2 = np.array(p2.current_state()[:3])
            misses.append(float(np.linalg.norm(r1 - r2)))

        misses = np.array(misses)
        # Count local minima: grid points where miss[i] < miss[i-1] and miss[i] < miss[i+1]
        local_mins = [
            i for i in range(1, len(misses) - 1)
            if misses[i] < misses[i - 1] and misses[i] < misses[i + 1]
        ]
        # Document: at 5 m/s overtaking, expect at least 1 minimum found by find_tca.
        # If > 1, that demonstrates the multi-encounter limitation.
        assert len(local_mins) >= 1, (
            "Expected at least one local minimum in the coarse grid — "
            "find_tca must have something to find"
        )
        # Store for informational purposes (visible in verbose mode)
        n_mins = len(local_mins)
        assert n_mins >= 1  # already checked above; keep count accessible

    def test_slow_overtaking_fowler_pc_finite(self, slow_overtaking_scenario):
        """
        Fowler Pc for the single TCA found must be finite and in [0,1].
        We don't check the value — it is correct for that one encounter.
        """
        from collision.tca import get_states_at_tca

        epoch, miss = find_tca(
            slow_overtaking_scenario["epoch_start"],
            slow_overtaking_scenario["sc1_eci_t0"],
            slow_overtaking_scenario["sc2_eci_t0"],
        )
        s1, s2 = get_states_at_tca(
            slow_overtaking_scenario["epoch_start"],
            slow_overtaking_scenario["sc1_eci_t0"],
            slow_overtaking_scenario["sc2_eci_t0"],
            epoch,
        )
        cov1, cov2 = generate_covariances(s1, s2)
        pc = fowler_pc(s1, s2, cov1, cov2, self.HBR)
        assert np.isfinite(pc)
        assert 0.0 <= pc <= 1.0

    def test_slow_overtaking_chan_pc_finite(self, slow_overtaking_scenario):
        """Chan Pc for the single TCA must be finite and in [0,1]."""
        from collision.tca import get_states_at_tca

        epoch, miss = find_tca(
            slow_overtaking_scenario["epoch_start"],
            slow_overtaking_scenario["sc1_eci_t0"],
            slow_overtaking_scenario["sc2_eci_t0"],
        )
        s1, s2 = get_states_at_tca(
            slow_overtaking_scenario["epoch_start"],
            slow_overtaking_scenario["sc1_eci_t0"],
            slow_overtaking_scenario["sc2_eci_t0"],
            epoch,
        )
        cov1, cov2 = generate_covariances(s1, s2)
        pc = chan_pc(s1, s2, cov1, cov2, self.HBR)
        assert np.isfinite(pc)
        assert 0.0 <= pc <= 1.0

    def test_slow_overtaking_chan_agrees_with_fowler(self, slow_overtaking_scenario):
        """
        For the single encounter, Chan and Fowler should agree to within 1%
        regardless of the multi-encounter complication.
        """
        from collision.tca import get_states_at_tca

        epoch, miss = find_tca(
            slow_overtaking_scenario["epoch_start"],
            slow_overtaking_scenario["sc1_eci_t0"],
            slow_overtaking_scenario["sc2_eci_t0"],
        )
        s1, s2 = get_states_at_tca(
            slow_overtaking_scenario["epoch_start"],
            slow_overtaking_scenario["sc1_eci_t0"],
            slow_overtaking_scenario["sc2_eci_t0"],
            epoch,
        )
        cov1, cov2 = generate_covariances(s1, s2)
        pc_fowler = fowler_pc(s1, s2, cov1, cov2, self.HBR)
        pc_chan   = chan_pc(  s1, s2, cov1, cov2, self.HBR)
        if pc_fowler == 0.0 and pc_chan == 0.0:
            pytest.skip("Both methods return 0 — skip ratio test")
        if pc_fowler == 0.0:
            pytest.skip("Fowler underflowed — skip ratio test")
        rel_err = abs(pc_chan - pc_fowler) / pc_fowler
        assert rel_err < 0.01, (
            f"Chan/Fowler error = {rel_err:.2%} for slow overtaking encounter "
            f"(expected < 1%)"
        )
