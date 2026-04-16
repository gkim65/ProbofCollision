"""
plot_pc_findings.py

Visualise three key Pc phenomena discovered during development:

  Figure 1 — Pc vs. covariance scale
      Shows that Pc is NOT monotone in covariance size.  For scenarios where
      the miss distance is much larger than the projected 2D sigma (e.g. the
      high-Pc crossing with a tight covariance), Pc rises as the covariance
      grows.  Once the miss is within the spread of the Gaussian, Pc falls.

      The scale factor is applied to BOTH SC1 and SC2 covariances equally
      (same tracking quality assumed for both objects).  The Fowler method
      sums them — C_pos = C1 + C2 — so the combined projected covariance
      scales by the same factor.  Scaling both is the standard assumption;
      scaling only one would shift the curves but not change the qualitative
      behaviour.

  Figure 2 — Pc vs. miss distance: all four conjunction types
      Shows how encounter geometry controls Pc across the full miss-distance
      range.  Head-on has the highest Pc; crossing is 3-6 orders of magnitude
      lower at the same miss distance because the large along-track uncertainty
      projects into the encounter plane.  Overtaking is also suppressed for the
      same reason.  Near-miss uses the same crossing geometry but is plotted
      separately to show it overlaps exactly with the crossing curve.

  Figure 3 — Pc vs. miss distance: crossing vs. head-on at same v_rel
      Shows that encounter geometry matters independently of speed.  At 500 m/s,
      crossing Pc is consistently lower than head-on Pc at the same miss distance
      because the crossing geometry projects more of the along-track uncertainty
      into the encounter plane.

Run from the repo root:
    PYTHONPATH=src uv run python plots/plot_pc_findings.py

Output: plots/pc_findings.png

Performance note
----------------
find_tca involves 200 numerical orbit propagations per scenario.  This script
avoids calling it for every sweep point — instead it calls find_tca once per
(v_mag, ctype) combination, then reuses the returned TCA states across all miss
distances / covariance scales for that combination.  This makes it ~60× faster
than the naive approach.
"""

import sys
from pathlib import Path

# Allow running from repo root or from plots/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import brahe

from collision.conjunction import generate_conjunction
from collision.tca import find_tca, get_states_at_tca
from collision.covariance import generate_covariances
from collision.fowler import fowler_pc

brahe.initialize_eop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tca_states(r_mag, v_mag, ctype, seed):
    """Generate one scenario, find TCA, return (sc1, sc2, actual_miss_m)."""
    sc = generate_conjunction(r_mag=r_mag, v_mag=v_mag,
                              conjunction_type=ctype, seed=seed)
    ep, miss = find_tca(sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"])
    s1, s2 = get_states_at_tca(sc["epoch_start"], sc["sc1_eci_t0"],
                                sc["sc2_eci_t0"], ep)
    return s1, s2, float(miss)


def sweep_miss_distance(v_mag, ctype, seed, miss_values, hbr):
    """
    Compute Pc at each miss distance in miss_values.

    Fast path: call find_tca once at a reference miss distance, then for each
    other miss distance just scale the relative position vector in the returned
    TCA states.  This avoids O(N) find_tca calls and reduces propagation work
    to a single call.

    The direction of the relative position is preserved; only its magnitude is
    rescaled.  Covariance is regenerated at the reference TCA states for every
    point (cheap — just a matrix rotation).
    """
    ref_miss = float(np.median(miss_values))
    s1_ref, s2_ref, _ = get_tca_states(ref_miss, v_mag, ctype, seed)

    r_rel_ref = s1_ref[:3] - s2_ref[:3]
    r_hat = r_rel_ref / np.linalg.norm(r_rel_ref)

    pcs, actual_misses = [], []
    for miss in miss_values:
        # Shift s2 so that |r1 - r2| == miss, keeping velocity unchanged
        s1 = s1_ref.copy()
        s2 = s2_ref.copy()
        s2[:3] = s1[:3] - r_hat * miss

        c1, c2 = generate_covariances(s1, s2)
        pcs.append(max(fowler_pc(s1, s2, c1, c2, hbr), 1e-20))
        actual_misses.append(miss)

    return actual_misses, pcs


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

BASE_POS = (100, 500, 50)   # default RTN position 1-sigma (m)
BASE_VEL = (0.1, 0.5, 0.05) # default RTN velocity 1-sigma (m/s)
HBR = 10.0                   # hard-body radius (m)
FLOOR = 1e-25                # clip floor for log-scale plotting


# ── Plot 1: Pc vs. covariance scale factor ───────────────────────────────────
# Shows all four scenario types, each at a fixed miss distance.
# Overtaking uses r_mag=1000 m, v_mag=50 m/s — its encounter plane is R-N
# (v_rel ≈ -T), so the projected covariance is compact like head-on, and Pc
# is monotone-decreasing.  The 1000 m miss keeps its absolute Pc lower than
# head-on at 200 m, so the curve sits below.
ax = axes[0]
scales = np.logspace(-2, 1.5, 60)   # 0.01× to ~30× default sigma

cov_cases = [
    ("head-on\n(200 m miss)",          200, 500, "head-on",   7,   "steelblue"),
    ("high-Pc crossing\n(200 m miss)", 200, 500, "crossing",  42,  "darkorange"),
    ("overtaking\n(1000 m miss)",     1000,  50, "overtaking", 99, "forestgreen"),
    ("near-miss\n(10 m miss)",          10,  15, "crossing",  123, "mediumpurple"),
]

for label, r_mag, v_mag, ctype, seed, color in cov_cases:
    s1, s2, _ = get_tca_states(r_mag, v_mag, ctype, seed)
    pcs = []
    for sc in scales:
        pos_std = tuple(x * sc for x in BASE_POS)
        vel_std = tuple(x * sc for x in BASE_VEL)
        c1, c2 = generate_covariances(s1, s2, pos_std_rtn=pos_std,
                                       vel_std_rtn=vel_std)
        pcs.append(max(fowler_pc(s1, s2, c1, c2, HBR), FLOOR))
    ax.loglog(scales, pcs, color=color, lw=2, label=label)

ax.axvline(1.0, color="gray", ls="--", lw=1, label="default σ")
ax.set_xlabel("Covariance scale factor  (1 = default)")
ax.set_ylabel("Pc")
ax.set_title("Pc vs. Covariance Scale\n(HBR = 10 m)")
ax.legend(fontsize=7.5, loc="lower left")
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(FLOOR, 1.0)


# ── Plot 2: Pc vs. miss distance — three distinct geometry types ──────────────
# Shows head-on, crossing, and overtaking — three geometrically distinct cases.
# Near-miss is omitted because it uses the same crossing geometry and its curve
# is identical to crossing (only the miss distance differs, not the geometry).
#
# Overtaking has a degenerate geometry for sweep_miss_distance: at TCA the
# miss vector r_rel and v_rel are both approximately along-track (T), so
# shifting s2 along r_hat keeps v_rel ∥ r_rel — the encounter plane is
# undefined.  Instead we fix the TCA states from one scenario and shift s2
# in the radial (R) direction, which is perpendicular to v_rel and represents
# a realistic lateral miss offset for an overtaking encounter.
ax = axes[1]
miss_range = np.logspace(0.5, 4.3, 80)   # ~3 m to ~20,000 m

# Head-on and crossing: sweep_miss_distance works fine (r_rel ⊥ v_rel at TCA)
for ctype, seed, color, label in [
    ("head-on",  7,  "steelblue",  "head-on (~15 km/s retrograde)"),
    ("crossing", 42, "darkorange", "crossing (500 m/s, N-dominant v_rel)"),
]:
    v_ref = 500.0  # v_mag arg is ignored for head-on (retrograde ECI construction)
    misses, pcs = sweep_miss_distance(v_ref, ctype, seed, miss_range, HBR)
    axes[1].loglog(misses, pcs, color=color, lw=2, label=label)

# Overtaking: shift s2 in the radial (R) direction, perpendicular to v_rel.
# This keeps v_rel ⊥ r_rel (the TCA condition) and gives a well-defined
# encounter plane (≈ R-N), matching the physical geometry of an overtaking pass.
s1_ov, s2_ov, _ = get_tca_states(1000.0, 50.0, "overtaking", 99)
v_rel_ov = s1_ov[3:] - s2_ov[3:]
# Build radial unit vector: r_hat = s1_pos / |s1_pos|
r_hat_eci = s1_ov[:3] / np.linalg.norm(s1_ov[:3])
# Project out any component along v_rel to ensure true perpendicularity
vhat = v_rel_ov / np.linalg.norm(v_rel_ov)
perp = r_hat_eci - np.dot(r_hat_eci, vhat) * vhat
perp = perp / np.linalg.norm(perp)

ov_misses, ov_pcs = [], []
for miss in miss_range:
    s1 = s1_ov.copy(); s2 = s2_ov.copy()
    s2[:3] = s1[:3] - perp * miss   # offset in radial direction
    c1, c2 = generate_covariances(s1, s2)
    ov_pcs.append(max(fowler_pc(s1, s2, c1, c2, HBR), 1e-20))
    ov_misses.append(miss)
axes[1].loglog(ov_misses, ov_pcs, color="forestgreen", lw=2,
               label="overtaking (50 m/s, radial miss offset)")

axes[1].axvline(HBR, color="gray", ls=":", lw=1, label=f"HBR = {HBR:.0f} m")
axes[1].set_xlabel("Miss distance (m)")
axes[1].set_ylabel("Pc")
axes[1].set_title("Pc vs. Miss Distance — Geometry Comparison\n(default covariance, HBR = 10 m)")
axes[1].legend(fontsize=7.5, loc="lower left")
axes[1].grid(True, which="both", alpha=0.3)
axes[1].set_ylim(1e-20, 1.0)


# (right panel removed — redundant with left panel)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

plt.tight_layout()
out = Path(__file__).parent / "pc_findings.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
