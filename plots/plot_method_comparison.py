"""
plot_method_comparison.py

Five-panel figure comparing all four Pc methods across scenarios and regimes.

Panels
------
1. All 5 standard scenarios: Fowler / Chan / MC-2D (1M) / MC-3D (100k) side by side
2. v_rel sweep (crossing geometry, 200 m miss): how each method responds to closing speed
3. Case A: Tiny-Pc regime — where MC-2D and MC-3D both go blind
4. Case C: Slow overtaking — 15 minima, Pc comparison at global-min TCA
5. MC-3D convergence: Pc vs N for head-on (where 3D ≠ 2D) and crossing (where they agree)

Run from repo root:
    PYTHONPATH=src uv run python plots/plot_method_comparison.py
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
from collision.monte_carlo_3d import monte_carlo_3d_pc
from brahe import (NumericalOrbitPropagator, NumericalPropagationConfig,
                   ForceModelConfig)

brahe.initialize_eop()

# ── shared style ──────────────────────────────────────────────────────────────
COLORS = {
    "Fowler":  "#4878D0",
    "Chan":    "#EE854A",
    "MC-2D":   "#6ACC65",
    "MC-3D":   "#D65F5F",
}
N_2D = 1_000_000   # samples for MC-2D throughout
N_3D = 100_000     # samples for MC-3D throughout (increased from 10k)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
})

fig = plt.figure(figsize=(18, 12))
# 2×3 grid; panel 5 spans bottom-right two cells
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1:])   # wide convergence panel

fig.suptitle(
    f"Pc Method Comparison — Fowler / Chan / MC-2D ($N={N_2D//1_000_000}$M) / MC-3D ($N={N_3D//1_000}$k)",
    fontsize=13, fontweight="bold",
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _scenario(conj_type, r_mag, v_mag, seed, near_miss=False):
    sc = generate_conjunction(tca_hours=24.0, r_mag=r_mag, v_mag=v_mag,
                              conjunction_type=conj_type, seed=seed)
    epoch, miss = find_tca(sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"])
    if near_miss:
        s1, s2 = sc["sc1_eci_tca"], sc["sc2_eci_tca"]
    else:
        s1, s2 = get_states_at_tca(
            sc["epoch_start"], sc["sc1_eci_t0"], sc["sc2_eci_t0"], epoch)
    vrel = float(np.linalg.norm(s1[3:] - s2[3:]))
    c1, c2 = generate_covariances(s1, s2)
    return s1, s2, c1, c2, epoch, vrel, miss


def _run_all(s1, s2, c1, c2, epoch, hbr=10.0, seed=42):
    pf          = fowler_pc(s1, s2, c1, c2, hbr)
    pc_         = chan_pc(  s1, s2, c1, c2, hbr)
    p2d, lo2, hi2 = monte_carlo_pc(s1, s2, c1, c2, hbr,
                                    n_samples=N_2D, seed=seed)
    p3d, lo3, hi3 = monte_carlo_3d_pc(s1, s2, c1, c2, epoch, hbr,
                                       n_samples=N_3D, seed=seed)
    return dict(Fowler=pf, Chan=pc_,
                **{"MC-2D": p2d, "MC-2D_lo": lo2, "MC-2D_hi": hi2,
                   "MC-3D": p3d, "MC-3D_lo": lo3, "MC-3D_hi": hi3})


# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — 5 standard scenarios
# ══════════════════════════════════════════════════════════════════════════════
scenario_defs = [
    ("crossing\n(15 m/s)",   "crossing",   500,  15,  42, False),
    ("head-on\n(15 km/s)",   "head-on",    200, 500,   7, False),
    ("overtaking\n(50 m/s)", "overtaking", 1000, 50,  99, False),
    ("near-miss\n(15 m/s)",  "crossing",    10,  15, 123, True),
    ("high-Pc\n(500 m/s)",   "crossing",   200, 500,  42, False),
]
methods = ["Fowler", "Chan", "MC-2D", "MC-3D"]
x = np.arange(len(scenario_defs))
width = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

all_results = []
for _, ctype, r, v, seed, nm in scenario_defs:
    s1, s2, c1, c2, ep, _, _ = _scenario(ctype, r, v, seed, nm)
    all_results.append(_run_all(s1, s2, c1, c2, ep))

for i, method in enumerate(methods):
    vals = [max(r[method], 1e-15) for r in all_results]
    ax1.bar(x + offsets[i], vals, width, label=method,
            color=COLORS[method], alpha=0.85)

ax1.set_yscale("log")
ax1.set_ylim(1e-5, 0.5)
ax1.set_xticks(x)
ax1.set_xticklabels([s[0] for s in scenario_defs], fontsize=8)
ax1.set_ylabel("$P_c$")
ax1.set_title(f"Panel 1: 5 Standard Scenarios\n"
              f"(MC-2D $N={N_2D//1_000_000}$M, MC-3D $N={N_3D//1_000}$k, HBR=10 m)")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, axis="y", alpha=0.3)

# Annotate head-on bar with "3D ≠ 2D" note
ax1.annotate("3D\n$\\neq$2D", xy=(1 + offsets[3], all_results[1]["MC-3D"] * 1.3),
             fontsize=7, color=COLORS["MC-3D"], ha="center")


# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — v_rel sweep, crossing, 200 m miss
# ══════════════════════════════════════════════════════════════════════════════
v_mags = [5, 15, 50, 150, 500, 2000, 7500]
vrels, pf_v, pc_v, p2d_v, p3d_v = [], [], [], [], []
lo2_v, hi2_v, lo3_v, hi3_v = [], [], [], []

for vm in v_mags:
    s1, s2, c1, c2, ep, vrel, _ = _scenario("crossing", 200, vm, 42)
    res = _run_all(s1, s2, c1, c2, ep)
    vrels.append(vrel)
    pf_v.append(res["Fowler"]); pc_v.append(res["Chan"])
    p2d_v.append(res["MC-2D"]); p3d_v.append(res["MC-3D"])
    lo2_v.append(res["MC-2D_lo"]); hi2_v.append(res["MC-2D_hi"])
    lo3_v.append(res["MC-3D_lo"]); hi3_v.append(res["MC-3D_hi"])

vrels = np.array(vrels)
ax2.plot(vrels, pf_v, "o-",  color=COLORS["Fowler"], lw=1.5, ms=5, label="Fowler")
ax2.plot(vrels, pc_v, "s--", color=COLORS["Chan"],   lw=1.5, ms=5, label="Chan")
ax2.plot(vrels, p2d_v, "^-", color=COLORS["MC-2D"],  lw=1.5, ms=5,
         label=f"MC-2D ($N={N_2D//1_000_000}$M)")
ax2.fill_between(vrels, lo2_v, hi2_v, color=COLORS["MC-2D"], alpha=0.15)
ax2.plot(vrels, p3d_v, "D-", color=COLORS["MC-3D"],  lw=1.5, ms=5,
         label=f"MC-3D ($N={N_3D//1_000}$k)")
ax2.fill_between(vrels, lo3_v, hi3_v, color=COLORS["MC-3D"], alpha=0.15)

ax2.set_xscale("log")
ax2.set_xlabel("Relative speed at TCA (m/s)")
ax2.set_ylabel("$P_c$")
ax2.set_title("Panel 2: $P_c$ vs. Closing Speed\n(crossing, 200 m miss)")
ax2.legend(loc="lower left", fontsize=7.5)
ax2.grid(True, which="both", alpha=0.3)
ax2.axvline(2000, color="gray", ls=":", lw=1.0)
ax2.text(2200, min(p3d_v) * 1.5, "↑ 3D\ndiverges", fontsize=7, color="gray")


# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Case A: Tiny-Pc regime
# ══════════════════════════════════════════════════════════════════════════════
s1a, s2a, _, _, epa, _, _ = _scenario("crossing", 500, 500, 42)
cov_cases = [
    ("Default\n(100/500/50 m)", generate_covariances(s1a, s2a)),
    ("Tight\n(10/50/5 m)",      generate_covariances(s1a, s2a,
                                    pos_std_rtn=(10., 50., 5.),
                                    vel_std_rtn=(0.01, 0.05, 0.005))),
]
FLOOR = 1e-15
x3 = np.arange(len(cov_cases))
offsets3 = np.array([-1.5, -0.5, 0.5, 1.5]) * width

cov_results = []
for _, (c1, c2) in cov_cases:
    pf  = fowler_pc(s1a, s2a, c1, c2, 10.)
    pc_ = chan_pc(  s1a, s2a, c1, c2, 10.)
    p2d, _, _ = monte_carlo_pc(s1a, s2a, c1, c2, 10., n_samples=N_2D, seed=42)
    p3d, _, _ = monte_carlo_3d_pc(s1a, s2a, c1, c2, epa, 10., n_samples=N_3D, seed=42)
    cov_results.append({"Fowler": pf, "Chan": pc_, "MC-2D": p2d, "MC-3D": p3d})

for i, method in enumerate(methods):
    vals = [max(r[method], FLOOR) for r in cov_results]
    bars = ax3.bar(x3 + offsets3[i], vals, width, label=method,
                   color=COLORS[method], alpha=0.85)
    for bar, r in zip(bars, cov_results):
        raw = r[method]
        txt = "0" if raw == 0.0 else f"{raw:.0e}"
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 3,
                 txt, ha="center", va="bottom", fontsize=6.5, rotation=0)

ax3.set_yscale("log")
ax3.set_ylim(FLOOR / 5, 5.0)
ax3.set_xticks(x3)
ax3.set_xticklabels([c[0] for c in cov_cases], fontsize=9)
ax3.set_ylabel("$P_c$")
ax3.set_title(f"Panel 3: Case A — Tiny $P_c$ Regime\n"
              f"(500 m crossing; MC-2D $N={N_2D//1_000_000}$M, MC-3D $N={N_3D//1_000}$k)")
ax3.legend(loc="lower right", fontsize=8)
ax3.axhline(1e-6, color="gray", ls="--", lw=0.8, alpha=0.7)
ax3.text(1.52, 1.5e-6, f"MC floor\n($N={N_2D//1_000_000}$M)", fontsize=7, color="gray")
ax3.grid(True, axis="y", alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# Panel 4 — Case C: slow overtaking, miss distance + all-method Pc at global min
# ══════════════════════════════════════════════════════════════════════════════
ax4r = ax4.twinx()

sc3 = generate_conjunction(tca_hours=24.0, r_mag=500.0, v_mag=5.0,
                           conjunction_type="overtaking", seed=77)
ep_start = sc3["epoch_start"]
sc1_t0, sc2_t0 = sc3["sc1_eci_t0"], sc3["sc2_eci_t0"]
duration = sc3["epoch_tca"] - ep_start

pc_cfg = NumericalPropagationConfig.default()
fc_cfg = ForceModelConfig.two_body()
N_GRID = 200
times_h, misses_km = [], []
for i in range(N_GRID + 1):
    t = ep_start + duration * i / N_GRID
    p1 = NumericalOrbitPropagator(ep_start, sc1_t0, pc_cfg, fc_cfg)
    p2 = NumericalOrbitPropagator(ep_start, sc2_t0, pc_cfg, fc_cfg)
    p1.propagate_to(t); p2.propagate_to(t)
    r1 = np.array(p1.current_state()[:3])
    r2 = np.array(p2.current_state()[:3])
    times_h.append(duration * i / N_GRID / 3600)
    misses_km.append(float(np.linalg.norm(r1 - r2)) / 1e3)
times_h = np.array(times_h); misses_km = np.array(misses_km)
local_mins = [i for i in range(1, len(misses_km) - 1)
              if misses_km[i] < misses_km[i-1] and misses_km[i] < misses_km[i+1]]
global_idx = int(np.argmin(misses_km))

ax4.plot(times_h, misses_km, color="#4878D0", lw=1.2, alpha=0.6, label="Miss distance")
ax4.scatter(times_h[local_mins], misses_km[local_mins], color="#EE854A", s=20, zorder=5,
            label=f"{len(local_mins)} ignored minima")
ax4.scatter(times_h[global_idx], misses_km[global_idx], color="#D42B2B", s=80,
            marker="*", zorder=6, label="find\\_tca result")
ax4.set_xlabel("Time from $T_0$ (hours)")
ax4.set_ylabel("Miss distance (km)", color="#4878D0")
ax4.tick_params(axis="y", labelcolor="#4878D0")

# Pc at global-min TCA
ep3, _ = find_tca(sc3["epoch_start"], sc1_t0, sc2_t0)
s1c, s2c = get_states_at_tca(sc3["epoch_start"], sc1_t0, sc2_t0, ep3)
c1c, c2c = generate_covariances(s1c, s2c)
res_c = _run_all(s1c, s2c, c1c, c2c, ep3)

method_pcs = [(m, res_c[m]) for m in methods]
xoff = [-1.2, -0.4, 0.4, 1.2]
for j, (mname, pv) in enumerate(method_pcs):
    ax4r.scatter(times_h[global_idx] + xoff[j], pv,
                 color=COLORS[mname], marker="D", s=55, zorder=10,
                 label=f"{mname}: {pv:.2e}")
    if mname in ("MC-2D", "MC-3D"):
        lo_key, hi_key = f"{mname}_lo", f"{mname}_hi"
        ax4r.errorbar(times_h[global_idx] + xoff[j], pv,
                      yerr=[[pv - res_c[lo_key]], [res_c[hi_key] - pv]],
                      fmt="none", color=COLORS[mname], capsize=3)

ax4r.set_ylabel("$P_c$ at global-min TCA")
ax4r.set_ylim(0, max(pv for _, pv in method_pcs) * 3.5)
ax4r.legend(loc="upper left", fontsize=7, title="$P_c$ at TCA")
ax4.set_title(f"Panel 4: Case C — Slow Overtaking\n"
              f"($v_\\mathrm{{rel}}=5$ m/s; {len(local_mins)} minima, Fowler/Chan/MC only for global min)")
ax4.legend(loc="upper right", fontsize=7.5)
ax4.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# Panel 5 — MC-3D convergence: Pc vs N
#   Two scenarios: head-on (3D ≠ 2D) and crossing (3D ≈ 2D)
# ══════════════════════════════════════════════════════════════════════════════
N_vals = [500, 1_000, 2_000, 5_000, 10_000, 30_000, 50_000, 100_000]

conv_scenarios = [
    ("head-on\n(3D ≠ 2D baseline)",  "head-on",  200, 500,  7, False),
    ("crossing\n(3D ≈ 2D baseline)", "crossing", 200, 500, 42, False),
]

for idx, (label, ctype, r_mag, v_mag, seed, nm) in enumerate(conv_scenarios):
    s1, s2, c1, c2, ep, vrel, _ = _scenario(ctype, r_mag, v_mag, seed, nm)
    # Analytic baselines
    pf_base = fowler_pc(s1, s2, c1, c2, 10.)
    pc_base = chan_pc(  s1, s2, c1, c2, 10.)
    p2d_base, _, _ = monte_carlo_pc(s1, s2, c1, c2, 10., n_samples=N_2D, seed=42)

    p3d_pts, lo_pts, hi_pts = [], [], []
    for N in N_vals:
        p3d, lo, hi = monte_carlo_3d_pc(s1, s2, c1, c2, ep, 10., n_samples=N, seed=42)
        p3d_pts.append(p3d); lo_pts.append(lo); hi_pts.append(hi)

    color_offset = "#4878D0" if idx == 0 else "#6ACC65"   # blue for head-on, green for crossing

    ax5.semilogx(N_vals, p3d_pts, "o-", color=color_offset, lw=1.5, ms=5,
                 label=f"MC-3D — {label.split(chr(10))[0]}")
    ax5.fill_between(N_vals, lo_pts, hi_pts, color=color_offset, alpha=0.15)
    # Baselines as horizontal lines
    ls_fowler = "-" if idx == 0 else "--"
    ax5.axhline(pf_base, color=color_offset, ls=ls_fowler, lw=1.0, alpha=0.8,
                label=f"Fowler — {label.split(chr(10))[0]} ({pf_base:.2e})")
    ax5.axhline(p2d_base, color=color_offset, ls=":", lw=1.2, alpha=0.8,
                label=f"MC-2D ($N={N_2D//1_000_000}$M) — {label.split(chr(10))[0]} ({p2d_base:.2e})")

ax5.set_xlabel("MC-3D sample count $N$")
ax5.set_ylabel("$P_c$")
ax5.set_title("Panel 5: MC-3D Convergence vs. $N$\n"
              "Head-on (where 3D $\\neq$ 2D baseline) vs. crossing (where 3D $\\approx$ 2D baseline)")
ax5.legend(loc="right", fontsize=7.5, ncol=1)
ax5.grid(True, which="both", alpha=0.3)

# Annotate the convergence plateau
ax5.annotate("Head-on: 3D converges\nto ~1.5e-3, not ~5e-3",
             xy=(50_000, 1.5e-3), xytext=(8_000, 2.8e-3),
             fontsize=7.5, color="#4878D0",
             arrowprops=dict(arrowstyle="->", color="#4878D0", lw=0.8))


# ── save ──────────────────────────────────────────────────────────────────────
out = "plots/method_comparison.png"
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
plt.show()
