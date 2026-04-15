"""
plot_pc_findings.py

Visualise three key Pc phenomena discovered during development:

  Figure 1 — Pc vs. covariance scale
      Shows that Pc is NOT monotone in covariance size.  For scenarios where
      the miss distance is much larger than the projected 2D sigma (e.g. the
      high-Pc crossing with a tight covariance), Pc rises as the covariance
      grows.  Once the miss is within the spread of the Gaussian, Pc falls.

  Figure 2 — Pc vs. miss distance, varying relative speed (head-on)
      Shows that faster relative speed gives higher Pc at the same miss distance.
      Faster v_rel keeps the projected covariance compact in the encounter plane.

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

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

BASE_POS = (100, 500, 50)   # default RTN position 1-sigma (m)
BASE_VEL = (0.1, 0.5, 0.05) # default RTN velocity 1-sigma (m/s)
HBR = 10.0                   # hard-body radius (m)
FLOOR = 1e-25                # clip floor for log-scale plotting


# ── Plot 1: Pc vs. covariance scale factor ───────────────────────────────────
ax = axes[0]
scales = np.logspace(-2, 1.5, 60)   # 0.01× to ~30× default sigma

cov_cases = [
    ("head-on\n(201 m, 500 m/s)",           200, 500, "head-on",  7,   "steelblue"),
    ("high-Pc crossing\n(204 m, 500 m/s)",  200, 500, "crossing", 42,  "darkorange"),
    ("near-miss\n(10 m, 15 m/s)",            10,  15, "crossing", 123, "forestgreen"),
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


# ── Plot 2: Pc vs. miss distance — effect of relative speed (head-on) ────────
# One find_tca call per speed level (3 total), then reuse TCA states.
ax = axes[1]
miss_range = np.logspace(0.7, 3.2, 60)   # 5 m to ~1600 m

speed_cases = [
    (50,  "royalblue",  "50 m/s"),
    (200, "darkorange", "200 m/s"),
    (500, "forestgreen","500 m/s"),
]

for v_mag, color, label in speed_cases:
    misses, pcs = sweep_miss_distance(v_mag, "head-on", 7, miss_range, HBR)
    ax.loglog(misses, pcs, color=color, lw=2, label=f"v_rel ≈ {label}")

ax.axvline(HBR, color="gray", ls=":", lw=1, label=f"HBR = {HBR:.0f} m")
ax.set_xlabel("Miss distance (m)")
ax.set_ylabel("Pc")
ax.set_title("Pc vs. Miss Distance\n(head-on, default cov, HBR = 10 m)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(1e-20, 1.0)


# ── Plot 3: crossing vs. head-on at same v_rel (500 m/s) ─────────────────────
# One find_tca call per geometry type (2 total), then reuse TCA states.
ax = axes[2]
miss_range2 = np.logspace(0.7, 3.2, 60)

geom_cases = [
    ("head-on",  "steelblue",  "head-on"),
    ("crossing", "darkorange", "crossing"),
]

for ctype, color, label in geom_cases:
    misses, pcs = sweep_miss_distance(500, ctype, 7, miss_range2, HBR)
    ax.loglog(misses, pcs, color=color, lw=2, label=label)

ax.axvline(HBR, color="gray", ls=":", lw=1, label=f"HBR = {HBR:.0f} m")
ax.set_xlabel("Miss distance (m)")
ax.set_ylabel("Pc")
ax.set_title("Pc vs. Miss: Crossing vs. Head-on\n(v_rel ≈ 500 m/s, default cov, HBR = 10 m)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(1e-20, 1.0)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

plt.tight_layout()
out = Path(__file__).parent / "pc_findings.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
