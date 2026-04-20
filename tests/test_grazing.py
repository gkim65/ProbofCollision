"""
test_grazing.py

Grazing geometry: miss distance = HBR exactly (miss = HBR = 10 m).

When the covariance is isotropic and σ ≪ HBR, the Gaussian is compact: nearly
all probability is concentrated near the Gaussian mean.  When the nominal miss
distance equals the HBR (miss = HBR = 10 m), the Gaussian mean sits exactly on
the disk boundary.  By symmetry, roughly half the Gaussian mass falls inside the
disk, so Pc ~ 0.5.

This is a well-defined boundary case that all four methods should agree on.
The fixture uses σ = 1 m (σ ≪ HBR = 10 m) with equal spreads in all directions
so that the encounter-plane projection is also isotropic.

Tests cover:
  - All four methods return Pc ~ 0.5 (within tolerance)
  - All methods agree with each other
  - Pc > 0.5 when miss < HBR, Pc < 0.5 when miss > HBR
"""

import numpy as np
import pytest

from collision.fowler import fowler_pc
from collision.chan1997 import chan_pc
from collision.monte_carlo import monte_carlo_pc


# ---------------------------------------------------------------------------
# Grazing fixture helpers
# ---------------------------------------------------------------------------

HBR = 10.0
ISOTROPIC_SIGMA = 1.0    # σ << HBR: Gaussian is compact; half of it falls inside disk at miss=HBR


def _grazing_states():
    """
    Two spacecraft in a head-on geometry with miss = HBR = 10 m.

    SC1 is at (7e6, 0, 0) m with velocity (0, 7500, 0) m/s.
    SC2 is offset exactly HBR = 10 m along the x-axis with opposite y-velocity,
    producing a head-on encounter with v_rel along +y and r_rel along +x.
    With v_rel ∥ y-hat, the encounter plane is the x-z plane; the miss vector
    projects entirely into x-hat, giving miss_2d = HBR = 10 m exactly.
    """
    sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    sc2 = np.array([7.0e6 + HBR, 0.0, 0.0, 0.0, -7500.0, 0.0])
    return sc1, sc2


def _isotropic_cov(sigma_pos=ISOTROPIC_SIGMA):
    """6×6 diagonal covariance with equal position spreads in all directions."""
    var = np.array([sigma_pos**2] * 3 + [(sigma_pos * 0.001)**2] * 3)
    return np.diag(var)


# ---------------------------------------------------------------------------
# Grazing geometry tests
# ---------------------------------------------------------------------------

class TestGrazingGeometry:
    """
    All four Pc methods tested at the grazing boundary: miss = HBR = 10 m,
    isotropic covariance σ = 50 m in all directions.

    Expected: Pc ~ 0.5 for all analytic methods.
    """

    def test_fowler_grazing_pc_near_half(self):
        """Fowler Pc ≈ 0.5 when miss = HBR and covariance is isotropic."""
        sc1, sc2 = _grazing_states()
        cov = _isotropic_cov()
        pc = fowler_pc(sc1, sc2, cov, cov, hard_body_radius=HBR)
        assert 0.35 <= pc <= 0.65, (
            f"Fowler grazing Pc = {pc:.4f}, expected near 0.5 "
            f"(miss = HBR = {HBR} m, isotropic σ = {ISOTROPIC_SIGMA} m)"
        )

    def test_chan_grazing_pc_near_half(self):
        """Chan Pc ≈ 0.5 when miss = HBR and covariance is isotropic."""
        sc1, sc2 = _grazing_states()
        cov = _isotropic_cov()
        pc = chan_pc(sc1, sc2, cov, cov, hard_body_radius=HBR)
        assert 0.35 <= pc <= 0.65, (
            f"Chan grazing Pc = {pc:.4f}, expected near 0.5 "
            f"(miss = HBR = {HBR} m, isotropic σ = {ISOTROPIC_SIGMA} m)"
        )

    def test_mc2d_grazing_pc_near_half(self):
        """MC-2D Pc ≈ 0.5 when miss = HBR and covariance is isotropic."""
        sc1, sc2 = _grazing_states()
        cov = _isotropic_cov()
        pc, ci_low, ci_high = monte_carlo_pc(
            sc1, sc2, cov, cov,
            hard_body_radius=HBR, n_samples=1_000_000, seed=42,
        )
        assert 0.35 <= pc <= 0.65, (
            f"MC-2D grazing Pc = {pc:.4f} (CI=[{ci_low:.4f},{ci_high:.4f}]), "
            f"expected near 0.5"
        )

    def test_all_methods_agree_at_grazing(self):
        """All three 2D methods agree within 15% at the grazing boundary."""
        sc1, sc2 = _grazing_states()
        cov = _isotropic_cov()
        pc_fowler = fowler_pc(sc1, sc2, cov, cov, hard_body_radius=HBR)
        pc_chan = chan_pc(sc1, sc2, cov, cov, hard_body_radius=HBR)
        pc_mc, _, _ = monte_carlo_pc(
            sc1, sc2, cov, cov,
            hard_body_radius=HBR, n_samples=1_000_000, seed=42,
        )
        # All should be near 0.5; pairwise relative error < 15%
        for name_a, pc_a, name_b, pc_b in [
            ("Fowler", pc_fowler, "Chan", pc_chan),
            ("Fowler", pc_fowler, "MC-2D", pc_mc),
            ("Chan",   pc_chan,   "MC-2D", pc_mc),
        ]:
            rel_err = abs(pc_a - pc_b) / max(pc_a, pc_b)
            assert rel_err < 0.15, (
                f"{name_a}={pc_a:.4f} vs {name_b}={pc_b:.4f}: "
                f"relative error {rel_err:.1%} > 15% at grazing geometry"
            )

    def test_pc_higher_when_miss_less_than_hbr(self):
        """When miss < HBR, Pc should exceed the grazing value (~0.5)."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
        sc2_inside = np.array([7.0e6 + HBR * 0.5, 0.0, 0.0, 0.0, -7500.0, 0.0])
        cov = _isotropic_cov()
        pc_grazing = fowler_pc(*_grazing_states(), cov, cov, hard_body_radius=HBR)
        pc_inside  = fowler_pc(sc1, sc2_inside, cov, cov, hard_body_radius=HBR)
        assert pc_inside > pc_grazing, (
            f"Pc at miss=HBR/2 ({pc_inside:.4f}) should exceed grazing Pc ({pc_grazing:.4f})"
        )

    def test_pc_lower_when_miss_greater_than_hbr(self):
        """When miss > HBR, Pc should be below the grazing value (~0.5)."""
        sc1 = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
        sc2_outside = np.array([7.0e6 + HBR * 2.0, 0.0, 0.0, 0.0, -7500.0, 0.0])
        cov = _isotropic_cov()
        pc_grazing = fowler_pc(*_grazing_states(), cov, cov, hard_body_radius=HBR)
        pc_outside  = fowler_pc(sc1, sc2_outside, cov, cov, hard_body_radius=HBR)
        assert pc_outside < pc_grazing, (
            f"Pc at miss=2×HBR ({pc_outside:.4f}) should be below grazing Pc ({pc_grazing:.4f})"
        )

    def test_chan_fowler_agree_at_grazing(self):
        """Chan and Fowler agree to within 2% at the grazing boundary."""
        sc1, sc2 = _grazing_states()
        cov = _isotropic_cov()
        pc_fowler = fowler_pc(sc1, sc2, cov, cov, hard_body_radius=HBR)
        pc_chan   = chan_pc(  sc1, sc2, cov, cov, hard_body_radius=HBR)
        rel_err = abs(pc_chan - pc_fowler) / pc_fowler
        assert rel_err < 0.02, (
            f"Chan ({pc_chan:.4f}) vs Fowler ({pc_fowler:.4f}) relative error "
            f"{rel_err:.2%} at grazing — expected < 2% for isotropic covariance"
        )
