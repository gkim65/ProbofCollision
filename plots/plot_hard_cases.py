"""
plot_hard_cases.py

Three-panel figure illustrating the hard cases for Pc methods.

Run from repo root:
    PYTHONPATH=src uv run python plots/plot_hard_cases.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import brahe

from collision.conjunction import generate_conjunction
from collision.tca import find_tca, get_states_at_tca
from collision.covariance import generate_covariances
from collision.fowler import fowler_pc
from collision.chan1997 import chan_pc
from collision.monte_carlo import monte_carlo_pc
from brahe import NumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfig

brahe.initialize_eop()


# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Hard Cases for Analytic $P_c$ Methods", fontsize=13, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Case A: Tiny Pc / importance sampling gap
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

sc = generate_conjunction(
    tca_hours=24.0, r_mag=500.0, v_mag=500.0,
    conjunction_type="crossing", seed=42,
)
epoch, _ = find_tca(sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"])
s1, s2 = get_states_at_tca(
    sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"], epoch
)

cov1_def, cov2_def = generate_covariances(s1, s2)
cov1_tight, cov2_tight = generate_covariances(
    s1, s2, pos_std_rtn=(10.0, 50.0, 5.0), vel_std_rtn=(0.01, 0.05, 0.005)
)

results = {}
for label, c1, c2 in [("Default\n(100/500/50 m)", cov1_def, cov2_def),
                       ("Tight\n(10/50/5 m)",    cov1_tight, cov2_tight)]:
    results[label] = {
        "Fowler": fowler_pc(s1, s2, c1, c2, 10.0),
        "Chan":   chan_pc(  s1, s2, c1, c2, 10.0),
        "MC":     monte_carlo_pc(s1, s2, c1, c2, 10.0, n_samples=1_000_000, seed=42)[0],
    }

# Replace zeros with a sentinel for the plot (to show a "< floor" bar)
FLOOR = 1e-14
labels_cov = list(results.keys())
methods = ["Fowler", "Chan", "MC"]
colors = {"Fowler": "#4878D0", "Chan": "#EE854A", "MC": "#6ACC65"}

x = np.arange(len(labels_cov))
width = 0.22
offsets = [-width, 0, width]

for i, method in enumerate(methods):
    vals = [max(results[lbl][method], FLOOR) for lbl in labels_cov]
    bars = ax.bar(x + offsets[i], vals, width, label=method,
                  color=colors[method], alpha=0.85)
    # annotate actual value above each bar
    for bar, lbl in zip(bars, labels_cov):
        raw = results[lbl][method]
        txt = "0" if raw == 0.0 else f"{raw:.0e}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 2,
            txt,
            ha="center", va="bottom", fontsize=7.5, rotation=0,
        )

ax.set_yscale("log")
ax.set_ylim(FLOOR / 5, 5e-2)
ax.set_xticks(x)
ax.set_xticklabels(labels_cov, fontsize=9)
ax.set_ylabel("$P_c$")
ax.set_title("Case A: Tiny $P_c$\n(500 m crossing, tight covariance)")
ax.legend(loc="lower right")
ax.axhline(1e-6, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(1.52, 1.4e-6, "MC floor\n($N=10^6$)", fontsize=7.5, color="gray", va="bottom")


# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Case B: Chan anisotropy error vs σ₂/σ₁
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]

sc_ho = generate_conjunction(
    tca_hours=24.0, r_mag=200.0, v_mag=500.0,
    conjunction_type="head-on", seed=7,
)
epoch_ho, _ = find_tca(sc_ho["epoch_start"], sc_ho["sc1_eci_t0"], sc_ho["sc2_eci_t0"])
s1h, s2h = get_states_at_tca(
    sc_ho["epoch_start"], sc_ho["sc1_eci_t0"], sc_ho["sc2_eci_t0"], epoch_ho
)


def enc_plane_ratio(s1, s2, cov1, cov2):
    C_pos = cov1[:3, :3] + cov2[:3, :3]
    v_rel = s1[3:] - s2[3:]
    z = v_rel / np.linalg.norm(v_rel)
    rr = s1[:3] - s2[:3]
    rp = rr - np.dot(rr, z) * z
    rpm = np.linalg.norm(rp)
    if rpm < 1e-10:
        arb = np.array([1.0, 0.0, 0.0])
        rp = arb - np.dot(arb, z) * z
        rpm = np.linalg.norm(rp)
    xh = rp / rpm
    yh = np.cross(z, xh)
    B = np.array([xh, yh])
    evs = np.linalg.eigvalsh(B @ C_pos @ B.T)
    return float(np.sqrt(evs[1] / evs[0]))


sN_vals = [10, 20, 30, 50, 75, 100, 200, 300, 500, 750, 1000, 2000, 5000]
ratios, errors_pct = [], []

for sN in sN_vals:
    c1, c2 = generate_covariances(s1h, s2h, pos_std_rtn=(10.0, 500.0, float(sN)))
    ratio = enc_plane_ratio(s1h, s2h, c1, c2)
    pc_f = fowler_pc(s1h, s2h, c1, c2, 10.0)
    pc_c = chan_pc(s1h, s2h, c1, c2, 10.0)
    if pc_f > 0 and pc_c > 0:
        ratios.append(ratio)
        errors_pct.append(abs(pc_c - pc_f) / pc_f * 100)

ax.semilogx(ratios, errors_pct, "o-", color="#4878D0", markersize=5, linewidth=1.5)
ax.axhline(1.0, color="#EE854A", linestyle="--", linewidth=1.0, label="1% threshold")
ax.axhline(0.2, color="#6ACC65", linestyle=":",  linewidth=1.0, label="0.2% (default cov)")
ax.set_xlabel("Encounter-plane axis ratio $\\sigma_2/\\sigma_1$")
ax.set_ylabel("$|P_c^\\mathrm{Chan} - P_c^\\mathrm{Fowler}| / P_c^\\mathrm{Fowler}$ (%)")
ax.set_title("Case B: Chan Anisotropy Error\nvs. Covariance Elongation")
ax.legend(loc="upper left")
ax.set_xlim(left=1)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
ax.grid(True, which="both", alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Case C: miss distance over 24 h for slow overtaking
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]

sc3 = generate_conjunction(
    tca_hours=24.0, r_mag=500.0, v_mag=5.0,
    conjunction_type="overtaking", seed=77,
)
epoch_start = sc3["epoch_start"]
sc1_t0, sc2_t0 = sc3["sc1_eci_t0"], sc3["sc2_eci_t0"]
duration = sc3["epoch_tca"] - epoch_start   # seconds

pc_cfg = NumericalPropagationConfig.default()
fc_cfg = ForceModelConfig.two_body()

N_GRID = 200
times_h, misses_km = [], []
for i in range(N_GRID + 1):
    t = epoch_start + duration * i / N_GRID
    p1 = NumericalOrbitPropagator(epoch_start, sc1_t0, pc_cfg, fc_cfg)
    p2 = NumericalOrbitPropagator(epoch_start, sc2_t0, pc_cfg, fc_cfg)
    p1.propagate_to(t)
    p2.propagate_to(t)
    r1 = np.array(p1.current_state()[:3])
    r2 = np.array(p2.current_state()[:3])
    times_h.append(duration * i / N_GRID / 3600)
    misses_km.append(float(np.linalg.norm(r1 - r2)) / 1e3)

times_h = np.array(times_h)
misses_km = np.array(misses_km)

local_mins = [
    i for i in range(1, len(misses_km) - 1)
    if misses_km[i] < misses_km[i - 1] and misses_km[i] < misses_km[i + 1]
]

ax.plot(times_h, misses_km, color="#4878D0", linewidth=1.2, label="Miss distance")
ax.scatter(
    times_h[local_mins], misses_km[local_mins],
    color="#EE854A", zorder=5, s=30,
    label=f"{len(local_mins)} local minima\n(ignored by find\\_tca)",
)
global_idx = int(np.argmin(misses_km))
ax.scatter(
    times_h[global_idx], misses_km[global_idx],
    color="#D42B2B", zorder=6, s=60, marker="*",
    label=f"Global min\n(find\\_tca result)",
)
ax.axhline(0.01, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(0.3, 0.013, "HBR = 10 m", fontsize=7.5, color="gray")

ax.set_xlabel("Time from $T_0$ (hours)")
ax.set_ylabel("Miss distance (km)")
ax.set_title(f"Case C: Multiple Close Approaches\n($v_\\mathrm{{rel}}=5$ m/s overtaking, {len(local_mins)} minima in 24 h)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)


# ── save ──────────────────────────────────────────────────────────────────────
fig.tight_layout()
out = "plots/hard_cases.png"
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
plt.show()
