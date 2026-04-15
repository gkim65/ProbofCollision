"""
plot_conjunctions.py

Visualise the four conjunction scenario fixtures for sanity-checking and
communicating what each scenario looks like geometrically.

Four figures, one per scenario (crossing, head-on, overtaking, near-miss).
Each figure has four panels:

  Panel 1 — RTN trajectory (dR, dT, dN vs time)
      How the relative position evolves over the full 24-hour window.
      The dominant component at TCA (t=24h) identifies the conjunction type.

  Panel 2 — Miss distance vs time
      Scalar separation |r_rel| over the window.  Vertical line at the TCA
      found by find_tca.  Should dip to a minimum at exactly t=24h.

  Panel 3 — Encounter plane projection
      At TCA, show the 2D covariance as 1σ/2σ/3σ ellipses and the HBR disk.
      The ellipse shape explains the Pc value — compact for head-on (higher Pc),
      elongated for crossing (lower Pc at same miss distance).

  Panel 4 — RTN covariance cross-sections
      Position uncertainty ellipses in the R-T, R-N, and T-N planes.
      Shows the characteristic large-T, small-R-N anisotropy.

Run from the repo root:
    PYTHONPATH=src uv run python plots/plot_conjunctions.py

Output: plots/conjunctions_<name>.png  (one file per scenario, gitignored)

Performance note
----------------
find_tca is called once per scenario (4 total).  sample_rtn_trajectory is
called once per scenario (4 total, 49 samples each — cheap).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import brahe

from collision.conjunction import generate_conjunction, sample_rtn_trajectory
from collision.tca import find_tca, get_states_at_tca
from collision.covariance import generate_covariances

brahe.initialize_eop()

OUT_DIR = Path(__file__).parent
HBR = 10.0  # m


# ---------------------------------------------------------------------------
# Scenario definitions — same seeds as conftest.py
# ---------------------------------------------------------------------------

SCENARIOS = [
    dict(name="crossing",         r_mag=500.0,  v_mag=15.0,  conjunction_type="crossing",   seed=42),
    dict(name="head_on",          r_mag=200.0,  v_mag=500.0, conjunction_type="head-on",    seed=7),
    dict(name="overtaking",       r_mag=1000.0, v_mag=50.0,  conjunction_type="overtaking", seed=99),
    dict(name="near_miss",        r_mag=10.0,   v_mag=15.0,  conjunction_type="crossing",   seed=123),
]


# ---------------------------------------------------------------------------
# Helper: 2D covariance ellipse patch
# ---------------------------------------------------------------------------

def cov_ellipse(mean_2d, cov_2d, n_sigma, **kwargs):
    """Return a matplotlib Ellipse patch for a 2D Gaussian confidence region."""
    vals, vecs = np.linalg.eigh(cov_2d)
    # largest eigenvalue first
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_sigma * np.sqrt(vals)
    return Ellipse(xy=mean_2d, width=width, height=height, angle=angle, **kwargs)


# ---------------------------------------------------------------------------
# Helper: encounter-plane projection
# ---------------------------------------------------------------------------

def encounter_plane_projection(s1, s2, cov1, cov2):
    """
    Return miss_2d (m) and C_2d (m²) projected onto the encounter plane.
    Same logic as fowler_pc.
    """
    r_rel = s1[:3] - s2[:3]
    v_rel = s1[3:] - s2[3:]
    z_hat = v_rel / np.linalg.norm(v_rel)

    r_perp = r_rel - np.dot(r_rel, z_hat) * z_hat
    r_perp_mag = np.linalg.norm(r_perp)
    if r_perp_mag < 1e-10:
        arb = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arb, z_hat)) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        r_perp = arb - np.dot(arb, z_hat) * z_hat
        r_perp_mag = np.linalg.norm(r_perp)

    x_hat = r_perp / r_perp_mag
    y_hat = np.cross(z_hat, x_hat)
    B = np.array([x_hat, y_hat])

    C_pos = cov1[:3, :3] + cov2[:3, :3]
    C_2d = B @ C_pos @ B.T
    miss_2d = B @ r_rel
    return miss_2d, C_2d


# ---------------------------------------------------------------------------
# Main: one figure per scenario
# ---------------------------------------------------------------------------

for cfg in SCENARIOS:
    name  = cfg["name"]
    print(f"  {name} ...", end=" ", flush=True)

    sc = generate_conjunction(
        r_mag=cfg["r_mag"], v_mag=cfg["v_mag"],
        conjunction_type=cfg["conjunction_type"], seed=cfg["seed"],
    )

    # --- trajectory (cheap) ---
    traj = sample_rtn_trajectory(sc, n_samples=97)
    t_h        = traj[:, 0]
    dR, dT, dN = traj[:, 1], traj[:, 2], traj[:, 3]   # km
    miss_km    = np.sqrt(dR**2 + dT**2 + dN**2)

    # --- TCA search ---
    epoch_tca, miss_found = find_tca(
        sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"]
    )
    tca_hours = (epoch_tca - sc["epoch_start"]) / 3600.0

    s1, s2 = get_states_at_tca(
        sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"], epoch_tca
    )
    cov1, cov2 = generate_covariances(s1, s2)

    # --- encounter plane ---
    miss_2d, C_2d = encounter_plane_projection(s1, s2, cov1, cov2)

    # --- RTN covariance cross-sections (position block only) ---
    C_rtn_pos = np.diag([100.0**2, 500.0**2, 50.0**2])  # default RTN variances

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    title_base = name.replace("_", " ").title()

    # Panel 1: RTN trajectory
    ax = axes[0]
    ax.plot(t_h, dR, label="dR", color="steelblue")
    ax.plot(t_h, dT, label="dT", color="darkorange")
    ax.plot(t_h, dN, label="dN", color="forestgreen")
    ax.axvline(tca_hours, color="red", ls="--", lw=1, label="TCA")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Relative position (km)")
    ax.set_title(f"{title_base}\nRTN Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: miss distance vs time
    ax = axes[1]
    ax.plot(t_h, miss_km * 1e3, color="purple")  # back to metres
    ax.axvline(tca_hours, color="red", ls="--", lw=1, label=f"TCA ({miss_found:.1f} m)")
    ax.axhline(HBR, color="gray", ls=":", lw=1, label=f"HBR = {HBR:.0f} m")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Miss distance (m)")
    ax.set_title(f"{title_base}\nMiss Distance vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: encounter plane projection
    ax = axes[2]
    # sigma ellipses
    colors_ell = ["#1f77b4", "#ff7f0e", "#d62728"]
    alphas_ell = [0.35, 0.20, 0.10]
    for n_sig, color, alpha in zip([1, 2, 3], colors_ell, alphas_ell):
        ell = cov_ellipse(miss_2d, C_2d, n_sig,
                          fill=True, facecolor=color, alpha=alpha,
                          edgecolor=color, linewidth=1.2)
        ax.add_patch(ell)
        ell2 = cov_ellipse(miss_2d, C_2d, n_sig,
                           fill=False, edgecolor=color, linewidth=1.2,
                           label=f"{n_sig}σ")
        ax.add_patch(ell2)
    # HBR disk
    hbr_circle = plt.Circle((0, 0), HBR, color="red", fill=False,
                              linewidth=1.5, linestyle="--", label=f"HBR {HBR:.0f} m")
    ax.add_patch(hbr_circle)
    # miss vector dot
    ax.plot(*miss_2d, "kx", ms=8, mew=2, label="miss")
    # axis limits: 4σ or at least 3× HBR
    sig_max = np.sqrt(np.max(np.linalg.eigvalsh(C_2d)))
    lim = max(4 * sig_max, 3 * HBR, np.linalg.norm(miss_2d) * 1.2)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x̂ (encounter plane, m)")
    ax.set_ylabel("ŷ (encounter plane, m)")
    ax.set_title(f"{title_base}\nEncounter Plane (HBR = {HBR:.0f} m)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 4: RTN covariance cross-sections
    ax = axes[3]
    sigma = np.array([100.0, 500.0, 50.0])  # default R, T, N (m)
    labels_pairs = [("R", "T", 0, 1), ("R", "N", 0, 2), ("T", "N", 1, 2)]
    colors_rtn = ["steelblue", "darkorange", "forestgreen"]
    for (la, lb, i, j), color in zip(labels_pairs, colors_rtn):
        cov_2 = np.array([[sigma[i]**2, 0], [0, sigma[j]**2]])
        for n_sig, alpha in zip([1, 2, 3], [0.35, 0.20, 0.10]):
            ell = cov_ellipse([0, 0], cov_2, n_sig,
                              fill=True, facecolor=color, alpha=alpha,
                              edgecolor=color, linewidth=1.0)
            ax.add_patch(ell)
        ell_edge = cov_ellipse([0, 0], cov_2, 1,
                                fill=False, edgecolor=color, linewidth=1.5,
                                label=f"{la}-{lb} plane")
        ax.add_patch(ell_edge)
    lim_rtn = 600
    ax.set_xlim(-lim_rtn, lim_rtn); ax.set_ylim(-lim_rtn, lim_rtn)
    ax.set_aspect("equal")
    ax.set_xlabel("Component 1 (m)")
    ax.set_ylabel("Component 2 (m)")
    ax.set_title(f"{title_base}\nRTN Position Uncertainty (1σ)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / f"conjunctions_{name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out.name}")

print("Done.")
