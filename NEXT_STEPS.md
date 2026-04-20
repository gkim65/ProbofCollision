# Next Steps: Probability of Collision Implementation

Handoff document for the next development session.
Stop putting in the coauthored by claude code in the github messages.

---

## What Exists Already

### Code (`src/collision/`)

| File | Purpose |
|------|---------|
| `conjunction.py` | Generate synthetic LEO conjunction scenarios. Back-propagates from a designed TCA. Supports crossing, head-on, overtaking, near-miss. |
| `tca.py` | Find TCA via coarse grid (200 steps) + Brent's method. ~3–4 s per call, 1-second precision. |
| `covariance.py` | Build 6×6 ECI covariances from diagonal RTN 1-sigma inputs. Default: σ_R=100m, σ_T=500m, σ_N=50m. Covariance constructed at TCA, not propagated from T=0. |
| `fowler.py` | Fowler (1993) analytic Pc. 2D Gaussian over HBR disk via `scipy.integrate.dblquad`. ~94 ms/call. |
| `chan1997.py` | Chan (1997) analytic Pc. Same encounter-plane projection as Fowler; evaluates via noncentral chi-squared CDF. ~0.09 ms/call (~1000× faster). < 0.2% agreement with Fowler on all scenarios. Stable at Pc < 10⁻¹⁰. |
| `monte_carlo.py` | Monte Carlo Pc baseline (2D encounter-plane). Samples N(miss_2d, C_2d) in the encounter plane, counts hits inside HBR circle. ~0.5–2 s at N=10⁶. Agrees with Chan to within 3× at Pc ~ 10⁻⁴. |
| `monte_carlo_3d.py` | 3D trajectory-integrated MC. Perturbs full 6D states from 6×6 joint covariance, propagates ±120 s around TCA via two-body dynamics, finds true minimum miss via Brent's method, counts hits < HBR. ~5 s at N=10,000. |

### Tests (`tests/`)

| File | Count | What it covers |
|------|-------|----------------|
| `conftest.py` | — | Session-scoped fixtures: 5 scenarios, 5 TCA results, 5 covariance pairs, 4 MC-2D fixtures (N=1M), 5 MC-3D fixtures (N=10k, all scenarios), 4 hard-case fixtures |
| `test_conjunction.py` | 26 | Output structure, geometry accuracy, physical sanity, RTN trajectory |
| `test_tca.py` | 13 | TCA timing, miss distance accuracy, state retrieval, edge cases |
| `test_fowler.py` | 31 | Covariance generation, Pc properties, limit cases, magnitude ranges, covariance-size effects |
| `test_chan1997.py` | 28 | Chan vs Fowler agreement (< 1%), properties, limits, degenerate inputs |
| `test_monte_carlo.py` | 21 | Return type, reproducibility, CI width, vs Chan (3× tol), monotonicity, degenerate inputs |
| `test_monte_carlo_3d.py` | 20 | Return type, reproducibility, CI width, vs MC-2D (5× tol), monotonicity, degenerate inputs. All 5 scenarios covered including crossing and overtaking. |
| `test_hard_cases.py` | 19 (1 skip) | Case A tiny Pc, Case B anisotropy error, Case C multiple close approaches |
| `test_grazing.py` | 7 | Grazing geometry: miss=HBR=10m, σ=1m (σ≪HBR). All 2D methods give Pc~0.5. |
| **Total** | **165 (1 skip)** | All pass in ~175 seconds |

### Key fixtures in `conftest.py`

| Fixture | Scenario | Notes |
|---------|----------|-------|
| `crossing_scenario/tca/covs` | 500 m miss, 15 m/s, crossing | Pc ~ 3.89e-4 |
| `head_on_scenario/tca/covs` | 195 m miss, retrograde orbit, v_rel ≈ 15185 m/s | Pc ~ 4.98e-3 |
| `overtaking_scenario/tca/covs` | 1000 m miss, 50 m/s slow catch-up | Pc ~ 1.86e-4 |
| `near_miss_scenario/tca/covs` | 10 m miss, 15 m/s crossing | Pc ~ 4.99e-4 |
| `high_pc_crossing_scenario/tca/covs` | 201 m miss, 500 m/s crossing, seed=42 | Pc ~ 4.80e-4 |
| `crossing_mc / head_on_mc / near_miss_mc / high_pc_crossing_mc` | MC-2D at N=1M | session-scoped, seed=42 |
| `crossing_mc3d / overtaking_mc3d / head_on_mc3d / near_miss_mc3d / high_pc_mc3d` | MC-3D at N=10k | session-scoped, seed=42. All 5 scenarios covered. |
| `tiny_pc_scenario/tca / tiny_pc_covs_tight` | 500 m crossing, 10× tighter cov | Chan Pc ~ 5e-12 (hard case A) |
| `slow_overtaking_scenario` | 500 m, v_rel=5 m/s overtaking | 15 local minima in 24h (hard case C) |

### Key findings (documented in FINDINGS.md)

1. **Head-on Pc is overestimated by 2D methods.** For head-on conjunctions r_rel ∥ v_rel, so the full miss distance projects to ~0.003 m in the encounter plane → near-zero projected miss → high 2D Pc (~5e-3). MC-3D(N=1M) = 1.28e-3. The 2D methods overestimate by ~4×, 27σ outside MC-3D's CI.

2. **Chan anisotropy error plateaus at ~6%.** Once σ₂/σ₁ ≥ 5, Chan's error vs Fowler stabilizes near 6% and does not grow further at σ₂/σ₁=500. Default covariance gives ~0.2%.

3. **Tiny Pc needs importance sampling.** At Chan Pc ~ 5e-12, plain MC at N=1M returns 0. Importance sampling is the only way to validate Chan in this regime.

4. **Multiple close approaches.** At v_rel=5 m/s, 15 local minima exist in 24h. find_tca returns only the global minimum. True Pc = 1 - ∏(1 - Pc_i) is out of scope.

5. **Near-miss and crossing have nearly equal Pc.** Both use N-dominant v_rel and identical default covariance. σ_T=500m covariance is so elongated that its tail reaches the 10m HBR disk even at 500m miss → both give Pc~4e-4. Miss distance alone does not determine Pc when covariance is highly elongated.

### Plots (`plots/`)

| Script | What it shows |
|--------|---------------|
| `plot_conjunctions.py` | 4-panel per scenario: RTN trajectory, miss distance, encounter plane, RTN covariance cross-sections |
| `plot_3d_orbits.py` | 3D RTN relative motion for all scenarios |
| `plot_3d_eci.py` | Full ECI orbit + zoomed TCA view per scenario |
| `plot_scenario_comparison.py` | Side-by-side scenario comparisons |
| `plot_pc_findings.py` | Pc sensitivity: vs covariance scale, vs miss distance |
| `plot_hard_cases.py` | 3-panel: Case A bar chart, Case B error% vs σ ratio, Case C miss distance over 24h |
| `plot_method_comparison.py` | 5-panel: all 4 methods × 5 scenarios, v_rel sweep, Case A, Case C, MC-3D convergence vs N |

Run plots with: `PYTHONPATH=src uv run python plots/<script>.py`

### Report (`report/report.tex`)

Compiles to `report.pdf` with `pdflatex report.tex` × 2. All sections complete and accurate:
1. Introduction
2. Background
3. Synthetic Conjunction Generation
4. Finding TCA
5. Covariance Model
6. Fowler (1993)
7. Chan (1997)
8. Monte Carlo 2D
9. Monte Carlo 3D (head-on overestimation finding, full 5-scenario comparison table, method_comparison.png)
10. Hard Cases (A/B/C)
11. Findings (3 findings + Scenario Comparison with fig:comparison)
12. Test Suite (165 tests, 4 test tables)
13. Future Work

**Report accuracy audit completed this session.** All figures are referenced in prose. All v_rel values, Pc values, and test counts are correct.

---

## Observed Pc Reference Values

Default covariance (pos σ = 100/500/50 m R/T/N), HBR = 10 m:

| Scenario | Miss (m) | v_rel (m/s) | Fowler Pc | Chan Pc | MC-2D Pc | MC-3D(1M) Pc |
|----------|----------|-------------|-----------|---------|----------|--------------|
| crossing | 500 | 15 | 3.89e-4 | 3.89e-4 | 3.71e-4 | 3.91e-4 |
| high-Pc crossing | 201 | 500 | 4.80e-4 | 4.80e-4 | 4.64e-4 | 5.14e-4 |
| head-on | 195 | 15185 | 4.98e-3 | 4.98e-3 | 4.98e-3 | 1.28e-3 |
| overtaking | 1000 | 50 | 1.85e-4 | 1.86e-4 | 1.87e-4 | 1.64e-4 |
| near-miss | 10 | 15 | 5.00e-4 | 4.99e-4 | 4.78e-4 | 5.10e-4 |

MC-3D(1M) CIs (from last session run): crossing=[3.515e-4,4.305e-4], head-on=[1.208e-3,1.352e-3], overtaking=[1.384e-4,1.896e-4], near-miss=[4.648e-4,5.552e-4], high-Pc=[4.687e-4,5.593e-4].

---

## What Needs to Be Built Next

### Priority 1 — Importance Sampling (`src/collision/importance_sampling.py`)

Extends MC to Pc < 10⁻⁶. Draw from proposal distribution centered near HBR boundary, reweight by p(x)/q(x). Only method that can validate Chan at Pc < 1e-8. This is the only thing that can confirm Hard Case A results.

### Priority 2 — Patera (2001) Line Integral (`src/collision/patera2001.py`)

Reference: Patera (2001). "General Method for Calculating Satellite Collision Probability." JGCD.
Compare against Fowler/Chan on all 5 scenarios + hard cases. Add a new column to tab:all_methods.

### Priority 3 — Library-Grade Refactor

- Rename `fowler.py` → `fowler1993.py`
- Rename `generate_covariances` → `build_covariances_from_rtn`
- Add `py.typed` marker, complete NumPy-style docstrings

---

## Known Simplifications

### Covariance at TCA is not propagated from T=0
`generate_covariances` constructs covariances directly at TCA. A real pipeline propagates via STM — along-track uncertainty inflates substantially over 24 hours.

### Chan anisotropy correction is approximate
< 0.2% accurate for all tested LEO scenarios. Error grows when HBR ≈ σ₁ (large HBR regime).

---

## Performance Reference

| Operation | Time |
|-----------|------|
| Full test suite (165 tests) | ~175 s |
| Single `find_tca` call | ~3–4 s |
| `generate_covariances` | < 1 ms |
| `fowler_pc` (dblquad) | ~94 ms |
| `chan_pc` (ncx2.cdf) | ~0.09 ms |
| `monte_carlo_pc` N=10⁶ (2D) | ~0.5–2 s |
| `monte_carlo_3d_pc` N=10,000 | ~5 s |
| `monte_carlo_3d_pc` N=100,000 | ~50 s |
| `monte_carlo_3d_pc` N=1,000,000 | ~500 s |
