"""
plot_3d_orbits.py

3D visualization of each conjunction scenario in RTN relative coordinates.
SC1 is always at the origin. SC2 trajectory is shown relative to SC1.

Layout per scenario:
  - Full 24h path of SC2 relative to SC1 (faint, colored by time)
  - Highlighted segment in a window around TCA (bright)
  - SC1 at origin (blue dot), SC2 at TCA (red dot)
  - Dashed green line = miss vector at TCA

Run from the repo root:
    PYTHONPATH=src uv run python plots/plot_3d_orbits.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import brahe

from collision.conjunction import generate_conjunction, sample_rtn_trajectory
from collision.tca import find_tca

brahe.initialize_eop()

OUT_DIR = Path(__file__).parent

SCENARIOS = [
    dict(name="crossing",   label="Crossing",
         r_mag=500.0,  v_mag=7000.0, conjunction_type="crossing",   seed=42,
         window_min=5,  n_full=300, view=(25, -55)),
    dict(name="head_on",    label="Head-On",
         r_mag=200.0,  v_mag=500.0, conjunction_type="head-on",    seed=7,
         window_min=3,  n_full=300, view=(20, -40)),
    dict(name="overtaking", label="Overtaking",
         r_mag=1000.0, v_mag=15.0,  conjunction_type="overtaking", seed=99,
         window_min=90, n_full=500, view=(20, -50)),
    dict(name="near_miss",  label="Near Miss",
         r_mag=10.0,   v_mag=500.0, conjunction_type="crossing",   seed=123,
         window_min=5,  n_full=300, view=(25, -55)),
]

C1 = "#2196F3"   # SC1 blue
C2 = "#E53935"   # SC2 red
TCA_GREEN = "#43A047"


def plot_scenario(ax, sc, cfg, tca_hours, miss_m):
    # ── Sample full trajectory ──────────────────────────────────────────────
    traj = sample_rtn_trajectory(sc, n_samples=cfg["n_full"])
    t_h  = traj[:, 0]
    dR   = traj[:, 1]   # km
    dT   = traj[:, 2]
    dN   = traj[:, 3]

    # ── Color full path by time (light grey → red) ─────────────────────────
    # Draw as a series of segments colored by time fraction
    norm = Normalize(vmin=0, vmax=traj[-1, 0])
    cmap = plt.cm.Reds
    n = len(t_h)
    for i in range(n - 1):
        frac = t_h[i] / traj[-1, 0]
        color = cmap(0.2 + 0.8 * frac)   # avoid too-light reds at start
        ax.plot(dT[i:i+2], dR[i:i+2], dN[i:i+2],
                color=color, alpha=0.5, lw=1.0)

    # ── Highlight window around TCA ─────────────────────────────────────────
    win_h = cfg["window_min"] / 60.0
    mask  = np.abs(t_h - tca_hours) <= win_h
    if mask.sum() > 1:
        ax.plot(dT[mask], dR[mask], dN[mask],
                color=C2, lw=2.5, alpha=1.0,
                label=f"SC2 near TCA (±{cfg['window_min']} min)")

    # ── TCA point on SC2 ────────────────────────────────────────────────────
    tca_idx = np.argmin(np.abs(t_h - tca_hours))
    tx, ty, tz = dT[tca_idx], dR[tca_idx], dN[tca_idx]
    ax.scatter([tx], [ty], [tz], color=C2, s=80, zorder=10,
               edgecolors="white", linewidths=1.5, label=f"SC2 @ TCA")

    # ── SC1 at origin ───────────────────────────────────────────────────────
    ax.scatter([0], [0], [0], color=C1, s=80, zorder=10,
               edgecolors="white", linewidths=1.5, label="SC1 (always at origin)")

    # ── Miss vector ─────────────────────────────────────────────────────────
    ax.plot([0, tx], [0, ty], [0, tz],
            color=TCA_GREEN, lw=2.0, ls="--", zorder=8,
            label=f"Miss vector ({miss_m:.1f} m)")

    # ── Axes ────────────────────────────────────────────────────────────────
    ax.set_xlabel("T (along-track, km)", fontsize=8, labelpad=5)
    ax.set_ylabel("R (radial, km)", fontsize=8, labelpad=5)
    ax.set_zlabel("N (cross-track, km)", fontsize=8, labelpad=5)
    ax.set_title(f"{cfg['label']}\nmiss = {miss_m:.1f} m  |  v_rel = {cfg['v_mag']} m/s",
                 fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    ax.view_init(*cfg["view"])
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.7)

    # Axis limits from full trajectory (keep aspect roughly equal)
    rmax = max(np.max(np.abs(dR)), np.max(np.abs(dT)), np.max(np.abs(dN))) * 1.1
    rmax = max(rmax, abs(tx) * 1.3, abs(ty) * 1.3, abs(tz) * 1.3, 0.01)
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_zlim(-rmax, rmax)

    # Add origin cross-hairs for reference
    ax.plot([-rmax, rmax], [0, 0], [0, 0], color="grey", lw=0.4, alpha=0.4)
    ax.plot([0, 0], [-rmax, rmax], [0, 0], color="grey", lw=0.4, alpha=0.4)
    ax.plot([0, 0], [0, 0], [-rmax, rmax], color="grey", lw=0.4, alpha=0.4)


# ── 2×2 combined figure ────────────────────────────────────────────────────

print("Generating 3D conjunction plots...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle(
    "Conjunction Scenarios — SC2 Relative Motion in RTN Frame\n"
    "SC1 fixed at origin  |  path colored light→dark red as time progresses toward TCA",
    fontsize=13, fontweight="bold", y=0.99
)

for idx, cfg in enumerate(SCENARIOS):
    print(f"  {cfg['name']} ...", end=" ", flush=True)

    sc = generate_conjunction(
        r_mag=cfg["r_mag"], v_mag=cfg["v_mag"],
        conjunction_type=cfg["conjunction_type"], seed=cfg["seed"],
    )
    epoch_tca, miss_m = find_tca(sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"])
    tca_hours = (epoch_tca - sc["epoch_start"]) / 3600.0

    ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
    plot_scenario(ax, sc, cfg, tca_hours, miss_m)
    print("done")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT_DIR / "conjunctions_3d_orbits.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out.name}")

# ── Individual figures ─────────────────────────────────────────────────────

for cfg in SCENARIOS:
    print(f"  individual: {cfg['name']} ...", end=" ", flush=True)

    sc = generate_conjunction(
        r_mag=cfg["r_mag"], v_mag=cfg["v_mag"],
        conjunction_type=cfg["conjunction_type"], seed=cfg["seed"],
    )
    epoch_tca, miss_m = find_tca(sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"])
    tca_hours = (epoch_tca - sc["epoch_start"]) / 3600.0

    fig = plt.figure(figsize=(9, 8))
    fig.suptitle(
        f"{cfg['label']} Conjunction — SC2 Relative Motion in RTN Frame\n"
        "SC1 at origin  |  path colored light→dark as time → TCA",
        fontsize=11, fontweight="bold"
    )
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_scenario(ax, sc, cfg, tca_hours, miss_m)

    plt.tight_layout()
    out = OUT_DIR / f"conjunctions_3d_{cfg['name']}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out.name}")

print("Done.")
