"""
plot_scenario_comparison.py

Two focused comparison figures that highlight what actually distinguishes each
conjunction type from the others:

Figure 1 — Overtaking vs Head-On vs Crossing
  Shows miss distance over time on the same axes.
  Key insight: overtaking oscillates many times (18 laps), head-on oscillates
  a few times (different planes, complex approach), crossing is a single clean
  monotonic approach.

Figure 2 — Crossing vs Near Miss (same geometry, different miss distance)
  Side-by-side encounter plane at TCA.
  Key insight: identical geometry, only how close the X gets to the HBR disk
  differs. Near miss nearly hits; crossing safely misses.

Figure 3 — Overtaking: 3D relative motion showing the lapping spiral
  Bird's-eye view of RTN space so the "lapping" is obvious.

Figure 4 — Head-On vs Crossing: velocity direction comparison
  Arrow diagrams showing how the relative velocity vector is oriented
  differently (N-dominant for both, but opposite T-offset direction).

Run from repo root:
    PYTHONPATH=src uv run python plots/plot_scenario_comparison.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle
import brahe
from brahe import state_eci_to_rtn

from collision.conjunction import generate_conjunction, sample_rtn_trajectory
from collision.tca import find_tca, get_states_at_tca
from collision.covariance import generate_covariances

brahe.initialize_eop()

OUT_DIR = Path(__file__).parent
HBR = 10.0  # m

SCENARIOS = {
    "crossing":   dict(r_mag=500.0,  v_mag=7000.0, conjunction_type="crossing",   seed=42),
    "head_on":    dict(r_mag=200.0,  v_mag=500.0, conjunction_type="head-on",    seed=7),
    "overtaking": dict(r_mag=1000.0, v_mag=15.0,  conjunction_type="overtaking", seed=99),
    "near_miss":  dict(r_mag=10.0,   v_mag=500.0, conjunction_type="crossing",   seed=123),
}

def load_scenario(name):
    cfg = SCENARIOS[name]
    sc = generate_conjunction(**cfg)
    epoch_tca, miss_m = find_tca(sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"])
    tca_h = (epoch_tca - sc["epoch_start"]) / 3600.0
    traj = sample_rtn_trajectory(sc, n_samples=500)
    s1, s2 = get_states_at_tca(sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"], epoch_tca)
    cov1, cov2 = generate_covariances(s1, s2)
    return dict(sc=sc, traj=traj, tca_h=tca_h, miss_m=miss_m, s1=s1, s2=s2, cov1=cov1, cov2=cov2)

def encounter_plane(s1, s2, cov1, cov2):
    r_rel = s1[:3] - s2[:3]
    v_rel = s1[3:] - s2[3:]
    z_hat = v_rel / np.linalg.norm(v_rel)
    r_perp = r_rel - np.dot(r_rel, z_hat) * z_hat
    r_perp_mag = np.linalg.norm(r_perp)
    if r_perp_mag < 1e-10:
        arb = np.array([1., 0., 0.])
        r_perp = arb - np.dot(arb, z_hat) * z_hat
        r_perp_mag = np.linalg.norm(r_perp)
    x_hat = r_perp / r_perp_mag
    y_hat = np.cross(z_hat, x_hat)
    B = np.array([x_hat, y_hat])
    C_2d = B @ (cov1[:3,:3] + cov2[:3,:3]) @ B.T
    miss_2d = B @ r_rel
    return miss_2d, C_2d

def cov_ellipse(mean, cov, n_sigma, **kw):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    w, h = 2*n_sigma*np.sqrt(vals)
    return Ellipse(xy=mean, width=w, height=h, angle=angle, **kw)

print("Loading scenarios...")
data = {name: load_scenario(name) for name in SCENARIOS}
print("Done loading.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Miss distance vs time — all four on same axes, normalised
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("What Distinguishes Each Conjunction Type", fontsize=13, fontweight="bold")

colors = {"crossing": "#1565C0", "head_on": "#B71C1C", "overtaking": "#2E7D32", "near_miss": "#F57F17"}
labels = {"crossing": "Crossing (500 m miss, 500 m/s)",
          "head_on":  "Head-On (200 m miss, 500 m/s)",
          "overtaking": "Overtaking (1000 m miss, 15 m/s)",
          "near_miss": "Near Miss (10 m miss, 500 m/s)"}

ax = axes[0]
for name, d in data.items():
    t = d["traj"][:,0]
    miss_km = np.sqrt(d["traj"][:,1]**2 + d["traj"][:,2]**2 + d["traj"][:,3]**2)
    miss_m_plot = miss_km * 1e3
    ax.plot(t, miss_m_plot / 1e3, color=colors[name], lw=2, label=labels[name])

ax.set_yscale("log")
ax.set_xlabel("Time (hours, 0 = simulation start, 24 = TCA)", fontsize=10)
ax.set_ylabel("Miss distance (km, log scale)", fontsize=10)
ax.set_title("Miss Distance vs Time\n(all four scenarios — time runs left→right toward TCA)",
             fontsize=10, fontweight="bold")
ax.axvline(24.0, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(23.6, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-4,
        "TCA", fontsize=8, color="gray", ha="right", va="bottom")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which="both")

# Annotation boxes explaining the pattern
ax.annotate("18 oscillations\n= 18 laps around SC1",
            xy=(12, data["overtaking"]["miss_m"]/1e3 * 3),
            fontsize=7.5, color=colors["overtaking"],
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors["overtaking"], alpha=0.8))
ax.annotate("Single clean\napproach to TCA",
            xy=(16, 2000),
            fontsize=7.5, color=colors["crossing"],
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors["crossing"], alpha=0.8))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 panel 2: Overtaking zoomed — miss distance showing the lapping
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]
d = data["overtaking"]
t = d["traj"][:,0]
miss_m = np.sqrt(d["traj"][:,1]**2 + d["traj"][:,2]**2 + d["traj"][:,3]**2) * 1e3
ax.plot(t, miss_m, color=colors["overtaking"], lw=1.8, label="Overtaking")
ax.axhline(HBR, color="red", ls="--", lw=1.2, label=f"HBR = {HBR:.0f} m")
ax.axvline(d["tca_h"], color="gray", ls="--", lw=1, label=f"TCA ({d['miss_m']:.0f} m)")
ax.set_xlabel("Time (hours)", fontsize=10)
ax.set_ylabel("Miss distance (m)", fontsize=10)
ax.set_title("Overtaking — Lapping Pattern\n(SC2 repeatedly passes SC1)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# annotate a few peaks and troughs
peaks_idx = []
for i in range(1, len(miss_m)-1):
    if miss_m[i] < miss_m[i-1] and miss_m[i] < miss_m[i+1]:
        peaks_idx.append(i)
for idx in peaks_idx[::3][:5]:
    ax.annotate(f"{miss_m[idx]:.0f} m",
                xy=(t[idx], miss_m[idx]),
                xytext=(t[idx], miss_m[idx] + 1500),
                fontsize=6.5, ha="center", color=colors["overtaking"],
                arrowprops=dict(arrowstyle="->", color=colors["overtaking"], lw=0.8))

plt.tight_layout()
out = OUT_DIR / "comparison_miss_distance.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Crossing vs Near Miss — encounter plane side by side
# Shows they are the SAME geometry, different miss distance
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("Crossing vs Near Miss — Same Geometry, Different Miss Distance\n"
             "Both are inclined-plane conjunctions at 500 m/s; only separation differs",
             fontsize=11, fontweight="bold")

ell_colors = ["#1f77b4", "#ff7f0e", "#d62728"]
ell_alphas = [0.35, 0.20, 0.10]

for ax, name, title in zip(axes,
                            ["crossing", "near_miss"],
                            ["Crossing  (miss = 500 m)", "Near Miss  (miss = 12 m)"]):
    d = data[name]
    miss_2d, C_2d = encounter_plane(d["s1"], d["s2"], d["cov1"], d["cov2"])

    # sigma ellipses
    for n_sig, color, alpha in zip([1,2,3], ell_colors, ell_alphas):
        ax.add_patch(cov_ellipse(miss_2d, C_2d, n_sig,
                                  fill=True, facecolor=color, alpha=alpha,
                                  edgecolor=color, linewidth=1.2))
        ax.add_patch(cov_ellipse(miss_2d, C_2d, n_sig,
                                  fill=False, edgecolor=color, linewidth=1.2,
                                  label=f"{n_sig}σ"))

    # HBR disk
    ax.add_patch(Circle((0,0), HBR, color="red", fill=True,
                          alpha=0.2, linewidth=0))
    ax.add_patch(Circle((0,0), HBR, color="red", fill=False,
                          linewidth=2.0, linestyle="--", label=f"HBR = {HBR:.0f} m"))

    # miss vector
    ax.plot(*miss_2d, "kx", ms=10, mew=2.5, label=f"SC2 @ TCA\n({np.linalg.norm(miss_2d):.1f} m)")
    ax.plot([0], [0], "b+", ms=10, mew=2.5, label="SC1 @ TCA")

    # line from origin to miss point
    ax.annotate("", xy=miss_2d, xytext=(0,0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    sig_max = np.sqrt(np.max(np.linalg.eigvalsh(C_2d)))
    lim = max(4*sig_max, 3*HBR, np.linalg.norm(miss_2d)*1.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x̂ encounter plane (m)", fontsize=10)
    ax.set_ylabel("ŷ encounter plane (m)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=colors[name])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotate whether HBR is overlapped
    overlap = np.linalg.norm(miss_2d) < HBR + np.sqrt(np.max(np.linalg.eigvalsh(C_2d)))
    miss_dist = np.linalg.norm(miss_2d)
    ax.text(0.03, 0.03,
            f"miss = {miss_dist:.1f} m\nHBR = {HBR:.0f} m\n"
            f"{'← WITHIN 1σ ellipse!' if miss_dist < np.sqrt(np.max(np.linalg.eigvalsh(C_2d))) else ''}",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.9),
            va="bottom")

plt.tight_layout()
out = OUT_DIR / "comparison_crossing_vs_nearmiss.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Head-On vs Crossing — 3D RTN views with velocity arrows
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 7))
fig.suptitle("Head-On vs Crossing — 3D RTN Relative Motion with Velocity Arrows\n"
             "Both have v_rel ≈ 500 m/s; key difference is direction SC2 approaches from",
             fontsize=11, fontweight="bold")

# view angles chosen to show the N dimension clearly
view_angles = {"head_on": (30, -60), "crossing": (30, 30)}
titles = {
    "head_on":  "Head-On\nSC2 behind in -T, flies through in -N\n(coming from below the orbital plane)",
    "crossing": "Crossing\nSC2 ahead in +T, flies through in -N\n(coming from above the orbital plane)",
}

for idx, name in enumerate(["head_on", "crossing"]):
    d = data[name]
    traj = d["traj"]
    t = traj[:,0]
    dR, dT, dN = traj[:,1], traj[:,2], traj[:,3]
    tca_idx = np.argmin(np.abs(t - d["tca_h"]))

    ax = fig.add_subplot(1, 2, idx+1, projection="3d")

    # Full path colored by time
    cmap = plt.cm.plasma
    n = len(t)
    for i in range(n-1):
        frac = i/(n-1)
        ax.plot(dT[i:i+2], dR[i:i+2], dN[i:i+2],
                color=cmap(frac), lw=1.5, alpha=0.75)

    # SC1 at origin
    ax.scatter([0],[0],[0], s=120, color="#1565C0", zorder=10,
               edgecolors="white", linewidths=1.5, label="SC1")

    # SC2 at TCA
    ax.scatter([dT[tca_idx]],[dR[tca_idx]],[dN[tca_idx]], s=120,
               color="#B71C1C", zorder=10, edgecolors="white", linewidths=1.5,
               label=f"SC2 @ TCA ({d['miss_m']:.0f} m)")

    # --- SC1 velocity arrow: SC1 moves in +T direction (along-track) in RTN
    # In RTN frame SC1 is always at origin moving in +T, so arrow points +T
    arrow_len = max(abs(dT).max(), abs(dR).max(), abs(dN).max()) * 0.18
    ax.quiver(0, 0, 0, arrow_len, 0, 0,
              color="#1565C0", linewidth=2.5, arrow_length_ratio=0.3,
              label="SC1 velocity (+T)")

    # --- SC2 velocity arrow at TCA: use relative velocity from traj cols 4,5,6
    vR2, vT2, vN2 = traj[tca_idx,4], traj[tca_idx,5], traj[tca_idx,6]
    vmag = np.sqrt(vR2**2 + vT2**2 + vN2**2) + 1e-10
    ax.quiver(dT[tca_idx], dR[tca_idx], dN[tca_idx],
              vT2/vmag*arrow_len, vR2/vmag*arrow_len, vN2/vmag*arrow_len,
              color="#B71C1C", linewidth=2.5, arrow_length_ratio=0.3,
              label="SC2 rel. velocity")

    # Miss vector
    ax.plot([0, dT[tca_idx]], [0, dR[tca_idx]], [0, dN[tca_idx]],
            color="#2E7D32", lw=2, ls="--", label="Miss vector")

    ax.set_xlabel("T (along-track, km)", fontsize=8, labelpad=4)
    ax.set_ylabel("R (radial, km)", fontsize=8, labelpad=4)
    ax.set_zlabel("N (cross-track, km)", fontsize=8, labelpad=4)
    ax.set_title(titles[name], fontsize=9, fontweight="bold", color=colors[name])
    ax.view_init(*view_angles[name])
    ax.legend(fontsize=7.5, loc="upper left")
    ax.tick_params(labelsize=7)

    # Equal-ish axes
    rmax = max(abs(dT).max(), abs(dR).max(), abs(dN).max()) * 1.1
    ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax); ax.set_zlim(-rmax, rmax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, t[-1]))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.1, shrink=0.6)
    cb.set_label("Time (h)", fontsize=7)

plt.tight_layout()
out = OUT_DIR / "comparison_headon_vs_crossing.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Overtaking — 3D lapping spiral + 2D T-N view
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 7))
fig.suptitle("Overtaking — SC2 Slowly Lapping SC1 in RTN Frame\n"
             "SC2 starts ahead (+T), drifts back in N oscillations, passes, repeats (~18 laps over 24h)",
             fontsize=11, fontweight="bold")

d = data["overtaking"]
traj = d["traj"]
t, dR, dT, dN = traj[:,0], traj[:,1], traj[:,2], traj[:,3]
miss_m_arr = np.sqrt(dR**2 + dT**2 + dN**2) * 1e3
tca_idx = np.argmin(miss_m_arr)
cmap = plt.cm.RdYlGn_r
n = len(t)
arrow_len = max(abs(dT).max(), abs(dR).max(), abs(dN).max()) * 0.15

# Panel 1: 3D view — elev=20, azim=20 to show the spiral looping in T-N
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
for i in range(n-1):
    frac = i/(n-1)
    ax3d.plot(dT[i:i+2], dR[i:i+2], dN[i:i+2],
              color=cmap(frac), lw=1.5, alpha=0.8)

ax3d.scatter([0],[0],[0], s=120, color="#1565C0", zorder=10,
             edgecolors="white", linewidths=1.5, label="SC1")
ax3d.scatter([dT[tca_idx]],[dR[tca_idx]],[dN[tca_idx]], s=120,
             color="#B71C1C", zorder=10, edgecolors="white", linewidths=1.5,
             label=f"SC2 @ TCA ({d['miss_m']:.0f} m)")
ax3d.scatter([dT[0]],[dR[0]],[dN[0]], s=100, color="lime", marker="^",
             zorder=9, edgecolors="black", linewidths=0.8, label="t=0")

# SC1 velocity arrow: +T
ax3d.quiver(0,0,0, arrow_len,0,0,
            color="#1565C0", linewidth=2.5, arrow_length_ratio=0.3,
            label="SC1 vel (+T)")
# SC2 relative velocity at TCA
vR2, vT2, vN2 = traj[tca_idx,4], traj[tca_idx,5], traj[tca_idx,6]
vmag = np.sqrt(vR2**2+vT2**2+vN2**2)+1e-10
ax3d.quiver(dT[tca_idx], dR[tca_idx], dN[tca_idx],
            vT2/vmag*arrow_len, vR2/vmag*arrow_len, vN2/vmag*arrow_len,
            color="#B71C1C", linewidth=2.5, arrow_length_ratio=0.3,
            label="SC2 rel. vel @ TCA")

ax3d.set_xlabel("T (along-track, km)", fontsize=8, labelpad=4)
ax3d.set_ylabel("R (radial, km)", fontsize=8, labelpad=4)
ax3d.set_zlabel("N (cross-track, km)", fontsize=8, labelpad=4)
ax3d.set_title("3D view — lapping spiral\n(elev=25°, azim=160° shows N oscillation)",
               fontsize=9, fontweight="bold")
ax3d.view_init(elev=25, azim=160)
ax3d.legend(fontsize=7, loc="upper left")
ax3d.tick_params(labelsize=7)
rmax = max(abs(dT).max(), abs(dR).max(), abs(dN).max()) * 1.1
ax3d.set_xlim(-rmax, rmax); ax3d.set_ylim(-rmax, rmax); ax3d.set_zlim(-rmax, rmax)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, t[-1]))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax3d, fraction=0.025, pad=0.1, shrink=0.6)
cb.set_label("Time (h, green→red)", fontsize=7)

# Panel 2: T vs N bird's-eye — clearest view of the lapping
ax2d = fig.add_subplot(1, 2, 2)
for i in range(n-1):
    frac = i/(n-1)
    ax2d.plot(dT[i:i+2], dN[i:i+2], color=cmap(frac), lw=1.5, alpha=0.85)

ax2d.scatter(0, 0, s=120, color="#1565C0", zorder=10,
             edgecolors="white", linewidths=1.5, label="SC1")
ax2d.scatter(dT[tca_idx], dN[tca_idx], s=120, color="#B71C1C", zorder=10,
             edgecolors="white", linewidths=1.5, label=f"SC2 @ TCA ({d['miss_m']:.0f} m)")
ax2d.scatter(dT[0], dN[0], s=100, color="lime", marker="^", zorder=9,
             edgecolors="black", linewidths=0.8, label="t=0 (SC2 starts here)")

# SC1 velocity arrow (+T direction)
ax2d_arrow_len = abs(dT).max() * 0.12
ax2d.annotate("", xy=(ax2d_arrow_len, 0), xytext=(0, 0),
              arrowprops=dict(arrowstyle="-|>", color="#1565C0",
                              lw=2.5, mutation_scale=18))
ax2d.text(ax2d_arrow_len*0.5, 0.5, "SC1 vel", fontsize=8,
          color="#1565C0", ha="center")

# SC2 relative velocity at TCA
vT2_n, vN2_n = vT2/vmag, vN2/vmag
ax2d.annotate("",
              xy=(dT[tca_idx]+vT2_n*ax2d_arrow_len, dN[tca_idx]+vN2_n*ax2d_arrow_len),
              xytext=(dT[tca_idx], dN[tca_idx]),
              arrowprops=dict(arrowstyle="-|>", color="#B71C1C",
                              lw=2.5, mutation_scale=18))
ax2d.text(dT[tca_idx]+vT2_n*ax2d_arrow_len*1.3,
          dN[tca_idx]+vN2_n*ax2d_arrow_len*1.3,
          "SC2 rel. vel", fontsize=8, color="#B71C1C", ha="center")

ax2d.set_xlabel("T — Along-track (km)", fontsize=10)
ax2d.set_ylabel("N — Cross-track (km)", fontsize=10)
ax2d.set_title("T–N bird's-eye view\n(each loop = one lap of SC2 around SC1)",
               fontsize=9, fontweight="bold")
ax2d.legend(fontsize=8)
ax2d.grid(True, alpha=0.3)

sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, t[-1]))
sm2.set_array([])
cb2 = plt.colorbar(sm2, ax=ax2d, fraction=0.025, pad=0.04, shrink=0.8)
cb2.set_label("Time (hours, green=start, red=TCA)", fontsize=7.5)

plt.tight_layout()
out = OUT_DIR / "comparison_overtaking_lapping.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out.name}")

# ── 2D T-R/T-N overtaking panels ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Overtaking — SC2 Slowly Lapping SC1 in RTN Frame\n"
             "SC2 starts far ahead (+T), slowly drifts back, passes, and repeats",
             fontsize=11, fontweight="bold")

d = data["overtaking"]
traj = d["traj"]
t, dR, dT, dN = traj[:,0], traj[:,1], traj[:,2], traj[:,3]
miss_m_arr = np.sqrt(dR**2 + dT**2 + dN**2) * 1e3
tca_idx = np.argmin(miss_m_arr)
cmap = plt.cm.RdYlGn_r
n = len(t)

# T vs R view
ax = axes[0]
for i in range(n-1):
    frac = i/(n-1)
    ax.plot(dT[i:i+2], dR[i:i+2], color=cmap(frac), lw=1.5, alpha=0.8)
ax.scatter(dT[tca_idx], dR[tca_idx], s=150, color="red", zorder=10,
           edgecolors="white", linewidths=1.5, label=f"TCA (miss={d['miss_m']:.0f} m)")
ax.scatter(0, 0, s=120, color="blue", marker="+", linewidths=2.5, zorder=10, label="SC1")
ax.scatter(dT[0], dR[0], s=100, color="green", marker="^", zorder=9,
           edgecolors="white", linewidths=1.2, label="t=0 (24h before TCA)")
ax.set_xlabel("T — Along-track (km)", fontsize=10)
ax.set_ylabel("R — Radial (km)", fontsize=10)
ax.set_title("T–R plane view\n(along-track vs radial)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, t[-1]))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cb.set_label("Time (hours, green=early, red=near TCA)", fontsize=7.5)

# T vs N view — this shows the oscillation most clearly
ax = axes[1]
for i in range(n-1):
    frac = i/(n-1)
    ax.plot(dT[i:i+2], dN[i:i+2], color=cmap(frac), lw=1.5, alpha=0.8)
ax.scatter(dT[tca_idx], dN[tca_idx], s=150, color="red", zorder=10,
           edgecolors="white", linewidths=1.5, label=f"TCA (miss={d['miss_m']:.0f} m)")
ax.scatter(0, 0, s=120, color="blue", marker="+", linewidths=2.5, zorder=10, label="SC1")
ax.scatter(dT[0], dN[0], s=100, color="green", marker="^", zorder=9,
           edgecolors="white", linewidths=1.2, label="t=0")
ax.set_xlabel("T — Along-track (km)", fontsize=10)
ax.set_ylabel("N — Cross-track (km)", fontsize=10)
ax.set_title("T–N plane view\n(along-track vs cross-track — shows lapping)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, t[-1]))
sm2.set_array([])
cb2 = plt.colorbar(sm2, ax=ax, fraction=0.03, pad=0.04)
cb2.set_label("Time (hours)", fontsize=7.5)

plt.tight_layout()
out = OUT_DIR / "comparison_overtaking_2d.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out.name}")

print("\nAll comparison plots saved.")
