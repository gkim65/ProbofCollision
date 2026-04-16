"""
plot_3d_eci.py

Earth-centered (ECI) 3D view of conjunction scenarios.

Because both spacecraft share nearly the same orbit, a pure ECI plot at full
scale makes the conjunction invisible (500 m gap on a 6928 km orbit).
So each scenario gets TWO panels side-by-side:

  Left  — Full ECI view: Earth sphere + both orbit arcs over 24 h (context)
  Right — Zoomed ECI view: ±window around TCA, both spacecraft clearly separated

Run from the repo root:
    PYTHONPATH=src uv run python plots/plot_3d_eci.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import brahe
from brahe import R_EARTH

from collision.conjunction import generate_conjunction, _make_propagator
from collision.tca import find_tca

brahe.initialize_eop()

OUT_DIR = Path(__file__).parent

# All scenarios use the same camera angle for consistency.
# Lower elev = camera looks more horizontally at the orbit plane.
VIEW_FULL = (12, 20)
VIEW_ZOOM = (30, 20)

SCENARIOS = [
    dict(name="crossing",   label="Crossing",
         r_mag=500.0,  v_mag=7000.0, conjunction_type="crossing",   seed=42,
         zoom_sec=2),    # ±2 sec — ~14 km arc, 534 m miss is ~4% of view
    dict(name="head_on",    label="Head-On",
         r_mag=200.0,  v_mag=500.0, conjunction_type="head-on",    seed=7,
         zoom_sec=2),    # ±2 sec — ~30 km arc, 195 m miss visible
    dict(name="overtaking", label="Overtaking",
         r_mag=1000.0, v_mag=15.0,  conjunction_type="overtaking", seed=99,
         zoom_sec=180),  # ±3 min — slow drift, tight enough to see separation
    dict(name="near_miss",  label="Near Miss",
         r_mag=10.0,   v_mag=500.0, conjunction_type="crossing",   seed=123,
         zoom_sec=2),    # ±2 sec — ~14 km arc, 12 m miss clearly visible
]

C1 = "#1565C0"    # dark blue — SC1
C2 = "#B71C1C"    # dark red  — SC2
TCA_GREEN = "#2E7D32"
EARTH_BLUE = "#1A237E"


def sample_eci(sc, n_samples=300):
    """
    Sample ECI positions + velocities over the full 0→TCA arc.
    Returns (t_h, xyz1_km, xyz2_km, vel1_kms, vel2_kms).
    vel arrays are (N,3) — velocity at every sample point.
    """
    epoch_start = sc["epoch_start"]
    epoch_tca   = sc["epoch_tca"]
    tca_hours   = (epoch_tca - epoch_start) / 3600.0

    prop1 = _make_propagator(epoch_start, sc["sc1_eci_t0"])
    prop2 = _make_propagator(epoch_start, sc["sc2_eci_t0"])

    times, xyz1, xyz2, vel1, vel2 = [], [], [], [], []
    for i in range(n_samples):
        t_h = tca_hours * i / (n_samples - 1)
        t   = epoch_start + t_h * 3600.0
        prop1.propagate_to(t)
        prop2.propagate_to(t)
        s1 = np.array(prop1.current_state()[:6])
        s2 = np.array(prop2.current_state()[:6])
        times.append(t_h)
        xyz1.append(s1[:3] / 1e3)
        xyz2.append(s2[:3] / 1e3)
        vel1.append(s1[3:] / 1e3)
        vel2.append(s2[3:] / 1e3)

    return (np.array(times), np.array(xyz1), np.array(xyz2),
            np.array(vel1), np.array(vel2))


def sample_eci_zoom(sc, epoch_tca, zoom_sec, n_samples=200):
    """
    Sample ECI positions + velocities in a tight window (±zoom_sec) around TCA.
    Returns (t_h, xyz1_km, xyz2_km, vel1_kms, vel2_kms, tca_idx).
    """
    epoch_start = sc["epoch_start"]
    tca_hours   = (epoch_tca - epoch_start) / 3600.0
    win_h       = zoom_sec / 3600.0

    t_start = epoch_start + (tca_hours - win_h) * 3600.0
    t_end   = epoch_start + (tca_hours + win_h) * 3600.0

    prop1 = _make_propagator(epoch_start, sc["sc1_eci_t0"])
    prop2 = _make_propagator(epoch_start, sc["sc2_eci_t0"])

    times, xyz1, xyz2, vel1, vel2 = [], [], [], [], []
    for i in range(n_samples):
        t = t_start + (t_end - t_start) * i / (n_samples - 1)
        t_h = (t - epoch_start) / 3600.0
        prop1.propagate_to(t)
        prop2.propagate_to(t)
        s1 = np.array(prop1.current_state()[:6])
        s2 = np.array(prop2.current_state()[:6])
        times.append(t_h)
        xyz1.append(s1[:3] / 1e3)
        xyz2.append(s2[:3] / 1e3)
        vel1.append(s1[3:] / 1e3)
        vel2.append(s2[3:] / 1e3)

    times = np.array(times)
    xyz1  = np.array(xyz1)
    xyz2  = np.array(xyz2)
    vel1  = np.array(vel1)
    vel2  = np.array(vel2)
    tca_idx = np.argmin(np.abs(times - tca_hours))

    return times, xyz1, xyz2, vel1, vel2, tca_idx


def earth_sphere(ax, r_km, color=EARTH_BLUE, alpha=0.15, n=40):
    """Draw a translucent Earth sphere."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = r_km * np.outer(np.cos(u), np.sin(v))
    y = r_km * np.outer(np.sin(u), np.sin(v))
    z = r_km * np.outer(np.ones(n), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, zorder=0)
    # Equator ring
    eq_theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r_km * np.cos(eq_theta), r_km * np.sin(eq_theta),
            np.zeros_like(eq_theta), color=color, lw=0.6, alpha=0.5)


def velocity_arrow(ax, xyz, tip_idx, v_hat_at_tip, color, arrow_km):
    """
    Draw a velocity-direction arrow ON the orbit arc.

    The arrowhead tip is placed at xyz[tip_idx] (an actual sampled orbit point).
    The arrow shaft runs backward along the local velocity direction from that point,
    so the entire arrow lies on the orbital tangent at that position.

    xyz          : (N,3) array of orbit positions in km
    tip_idx      : index in xyz where the arrowhead tip sits
    v_hat_at_tip : unit velocity vector at tip_idx (from propagator, exact)
    arrow_km     : shaft length in km
    """
    tip  = xyz[tip_idx]
    tail = tip - v_hat_at_tip * arrow_km
    dx, dy, dz = tip - tail
    ax.quiver(tail[0], tail[1], tail[2], dx, dy, dz,
              color=color, lw=2.5,
              arrow_length_ratio=0.65,
              zorder=12)


def plot_eci_full(ax, times, xyz1, xyz2, vel1, vel2, tca_idx, cfg):
    """Full-scale ECI view with Earth."""
    earth_sphere(ax, R_EARTH / 1e3)

    # Orbit arcs — SC1 solid, SC2 dashed
    ax.plot(xyz1[:, 0], xyz1[:, 1], xyz1[:, 2],
            color=C1, lw=2.0, alpha=0.85, ls="-", label="SC1 orbit")
    ax.plot(xyz2[:, 0], xyz2[:, 1], xyz2[:, 2],
            color=C2, lw=2.0, alpha=0.85, ls="--", label="SC2 orbit")

    # TCA dots
    ax.scatter(*xyz1[tca_idx], color=C1, s=80, zorder=10,
               edgecolors="white", linewidths=1.5, label="SC1 @ TCA")
    ax.scatter(*xyz2[tca_idx], color=C2, s=80, zorder=10,
               edgecolors="white", linewidths=1.5, label="SC2 @ TCA")

    # Velocity arrows: place tip at a sample point BEFORE TCA on the actual orbit
    # SC1: further back (tip at tca_idx - 8), SC2: closer (tip at tca_idx - 3)
    # Arrow shaft length = 5 samples, so it always lies along the orbit arc.
    earth_r_km = R_EARTH / 1e3
    arrow_len_km = earth_r_km * 0.25

    for xyz, vel, color, tip_offset in [
        (xyz1, vel1, C1, 1),   # SC1 slightly further back
        (xyz2, vel2, C2, 0),   # SC2 just behind dot
    ]:
        tip_idx = max(tca_idx - tip_offset, 0)
        v_hat = vel[tip_idx] / (np.linalg.norm(vel[tip_idx]) + 1e-12)
        velocity_arrow(ax, xyz, tip_idx, v_hat, color, arrow_len_km)

    ax.set_xlabel("X (km)", fontsize=7)
    ax.set_ylabel("Y (km)", fontsize=7)
    ax.set_zlabel("Z (km)", fontsize=7)
    ax.set_title(f"{cfg['label']}\nFull orbit (Earth-centered)\nSC1 solid · SC2 dashed  |  arrows = orbital direction @ TCA",
                 fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.view_init(*VIEW_FULL)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.7)

    r = (R_EARTH / 1e3) * 1.35
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)


def plot_eci_zoom(ax, times_z, xyz1_z, xyz2_z, vel1_z, vel2_z, tca_idx_z, miss_m, cfg):
    """
    Zoomed ECI view around TCA using the fine-grained zoom sample.
    Axis limits sized to show the arc curvature and the miss vector clearly.
    """
    p1  = xyz1_z[tca_idx_z]
    p2  = xyz2_z[tca_idx_z]
    mid = (p1 + p2) / 2

    # Axis half-width: largest of arc spread from mid, or 10× miss distance
    miss_km   = miss_m / 1e3
    arc1_max  = np.max(np.linalg.norm(xyz1_z - mid, axis=1))
    arc2_max  = np.max(np.linalg.norm(xyz2_z - mid, axis=1))
    half = max(arc1_max * 1.25, arc2_max * 1.25, miss_km * 20, 0.001)

    # Orbit arcs — SC1 solid, SC2 dashed
    zoom_label = (f"±{cfg['zoom_sec']} sec" if cfg['zoom_sec'] < 60
                  else f"±{cfg['zoom_sec']//60} min")
    ax.plot(xyz1_z[:, 0], xyz1_z[:, 1], xyz1_z[:, 2],
            color=C1, lw=2.5, ls="-",  label=f"SC1 orbit ({zoom_label})")
    ax.plot(xyz2_z[:, 0], xyz2_z[:, 1], xyz2_z[:, 2],
            color=C2, lw=2.5, ls="--", label=f"SC2 orbit ({zoom_label})")

    # TCA dots
    ax.scatter(*p1, color=C1, s=90, zorder=10, edgecolors="white", linewidths=1.5)
    ax.scatter(*p2, color=C2, s=90, zorder=10, edgecolors="white", linewidths=1.5)

    # Velocity arrows: tip placed at an actual orbit sample before TCA
    # SC1 further back, SC2 closer — same pattern as full view
    arrow_len_km = half * 0.22
    for xyz_z, vel_z, color, tip_offset in [
        (xyz1_z, vel1_z, C1, 30),   # SC1 further back (~30 samples before TCA)
        (xyz2_z, vel2_z, C2, 10),   # SC2 closer (~10 samples before TCA)
    ]:
        tip_idx = max(tca_idx_z - tip_offset, 0)
        v_hat = vel_z[tip_idx] / (np.linalg.norm(vel_z[tip_idx]) + 1e-12)
        velocity_arrow(ax, xyz_z, tip_idx, v_hat, color, arrow_len_km)

    # Miss vector (dotted green line between the two TCA dots)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=TCA_GREEN, lw=2.0, ls=":", zorder=8,
            label=f"Miss ({miss_m:.1f} m)")

    ax.set_xlabel("X (km)", fontsize=7)
    ax.set_ylabel("Y (km)", fontsize=7)
    ax.set_zlabel("Z (km)", fontsize=7)
    ax.set_title(f"Zoomed {zoom_label} around TCA\nSC1 solid · SC2 dashed",
                 fontsize=8)
    ax.tick_params(labelsize=6)
    ax.view_init(*VIEW_ZOOM)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.7)

    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)


# ── One figure per scenario (2 panels: full + zoom) ───────────────────────

print("Generating ECI plots...")

for cfg in SCENARIOS:
    print(f"  {cfg['name']} ...", end=" ", flush=True)

    sc = generate_conjunction(
        r_mag=cfg["r_mag"], v_mag=cfg["v_mag"],
        conjunction_type=cfg["conjunction_type"], seed=cfg["seed"],
    )
    epoch_tca, miss_m = find_tca(sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"])
    tca_hours = (epoch_tca - sc["epoch_start"]) / 3600.0

    # Full-arc sample (coarse) for the full ECI panel
    times, xyz1, xyz2, vel1, vel2 = sample_eci(sc, n_samples=300)
    tca_idx = np.argmin(np.abs(times - tca_hours))

    # Fine-grained sample around TCA for the zoom panel
    times_z, xyz1_z, xyz2_z, vel1_z, vel2_z, tca_idx_z = sample_eci_zoom(
        sc, epoch_tca, cfg["zoom_sec"], n_samples=200
    )

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"{cfg['label']} Conjunction — ECI Frame\n"
        f"miss = {miss_m:.1f} m  |  v_rel = {cfg['v_mag']} m/s  |  r_offset = {cfg['r_mag']} m",
        fontsize=11, fontweight="bold"
    )

    ax_full = fig.add_subplot(1, 2, 1, projection="3d")
    plot_eci_full(ax_full, times, xyz1, xyz2, vel1, vel2, tca_idx, cfg)

    ax_zoom = fig.add_subplot(1, 2, 2, projection="3d")
    plot_eci_zoom(ax_zoom, times_z, xyz1_z, xyz2_z, vel1_z, vel2_z, tca_idx_z, miss_m, cfg)

    plt.tight_layout()
    out = OUT_DIR / f"conjunctions_eci_{cfg['name']}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out.name}")

# ── 2×2 combined figure ────────────────────────────────────────────────────

print("  combined figure ...", end=" ", flush=True)
fig = plt.figure(figsize=(22, 18))
fig.suptitle(
    "Conjunction Scenarios — ECI Frame\n"
    "Left: full orbit context with Earth  |  Right: zoomed view around TCA",
    fontsize=13, fontweight="bold", y=0.99
)

for idx, cfg in enumerate(SCENARIOS):
    sc = generate_conjunction(
        r_mag=cfg["r_mag"], v_mag=cfg["v_mag"],
        conjunction_type=cfg["conjunction_type"], seed=cfg["seed"],
    )
    epoch_tca, miss_m = find_tca(sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"])
    tca_hours = (epoch_tca - sc["epoch_start"]) / 3600.0

    times, xyz1, xyz2, vel1, vel2 = sample_eci(sc, n_samples=300)
    tca_idx = np.argmin(np.abs(times - tca_hours))

    times_z, xyz1_z, xyz2_z, vel1_z, vel2_z, tca_idx_z = sample_eci_zoom(
        sc, epoch_tca, cfg["zoom_sec"], n_samples=200
    )

    ax_full = fig.add_subplot(4, 2, idx * 2 + 1, projection="3d")
    ax_zoom = fig.add_subplot(4, 2, idx * 2 + 2, projection="3d")

    plot_eci_full(ax_full, times, xyz1, xyz2, vel1, vel2, tca_idx, cfg)
    plot_eci_zoom(ax_zoom, times_z, xyz1_z, xyz2_z, vel1_z, vel2_z, tca_idx_z, miss_m, cfg)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT_DIR / "conjunctions_eci_all.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved → {out.name}")

print("Done.")
