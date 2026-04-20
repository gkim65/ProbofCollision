# Development Notes: Probability of Collision

This document explains the codebase structure, how to use each module, and what every test is checking and why.

---

## Project Structure

```
src/collision/
    conjunction.py      — generate synthetic conjunction scenarios
    tca.py              — find Time of Closest Approach from initial states
    covariance.py       — generate RTN covariances and rotate to ECI
    fowler.py           — Fowler (1993) analytic Pc via 2D Gaussian integral
    chan1997.py         — Chan (1997) analytic Pc via noncentral chi-squared CDF
    monte_carlo.py      — MC-2D: encounter-plane sampling baseline
    monte_carlo_3d.py   — MC-3D: trajectory-integrated full 3D sampling

tests/
    conftest.py             — shared fixtures (scenarios, TCA results, covariances, MC results)
    test_conjunction.py     — 26 tests
    test_tca.py             — 13 tests
    test_fowler.py          — 31 tests
    test_chan1997.py        — 28 tests
    test_monte_carlo.py     — 21 tests
    test_monte_carlo_3d.py  — 20 tests
    test_hard_cases.py      — 19 tests (1 skip)
    test_grazing.py         — 7 tests

plots/
    plot_conjunctions.py        — 4-panel per-scenario: RTN trajectory, miss distance, encounter plane, RTN covariance
    plot_3d_orbits.py           — 3D RTN relative motion (SC2 relative to SC1 at origin)
    plot_3d_eci.py              — ECI 3D view: full orbit + zoomed TCA panel per scenario
    plot_scenario_comparison.py — side-by-side comparison figures across scenarios
    plot_pc_findings.py         — Pc sensitivity: vs covariance scale, vs miss distance, geometry comparison
    plot_hard_cases.py          — 3-panel: Case A bar chart, Case B error% vs σ ratio, Case C miss distance
    plot_method_comparison.py   — 5-panel: all 4 methods × 5 scenarios, v_rel sweep, hard cases, convergence
```

Run plots with: `PYTHONPATH=src uv run python plots/<script>.py`

---

## `conjunction.py` — Scenario Generator

### What it does

Creates realistic synthetic conjunction scenarios where you know the **ground truth** — exact miss distance, exact TCA epoch, and states at TCA for both spacecraft. This lets you verify that your Pc methods produce sensible answers.

The key idea: instead of picking initial conditions and hoping a close approach happens, it works **backwards from TCA**:

1. Define SC1's orbit at TCA via Keplerian elements
2. Place SC2 relative to SC1 at TCA using **RTN coordinates** (Radial-Tangential-Normal — a frame centered on SC1 that moves with it)
3. Propagate both spacecraft backwards 24 hours using brahe's numerical propagator to get T=0 initial conditions

Because you designed the TCA geometry explicitly, you always know the ground truth.

### RTN frame explained

- **R (Radial)**: points from Earth's center toward SC1
- **T (Tangential/Along-track)**: points in the direction SC1 is moving
- **N (Normal/Cross-track)**: perpendicular to the orbital plane (R × T)

The conjunction types and their physical geometry:

| Type | Miss vector at TCA | Velocity direction | Physical meaning |
|------|--------------------|--------------------|-----------------|
| `"crossing"` | +T (along-track) | −N (cross-track) | Two orbits in different planes crossing — most common real-world type. |
| `"head-on"` | +T (along-track) | retrograde | SC2 in retrograde orbit, built directly in ECI. v_rel ≈ 15 km/s. |
| `"overtaking"` | +T (along-track) | −T (along-track) | SC2 slightly ahead in same orbital plane, SC1 catching up. v_rel ~15–50 m/s. |
| `"near_miss"` | +T (along-track) | −N (cross-track) | Same geometry as crossing but r_mag = 10 m — stress test. |

**Head-on is special:** it cannot be built in RTN. Instead `_sc2_eci_head_on()` constructs SC2 directly in ECI by flipping the along-track velocity component, producing a true retrograde orbit.

### Main function: `generate_conjunction()`

```python
from collision.conjunction import generate_conjunction

scenario = generate_conjunction(
    tca_hours=24.0,          # how far before TCA to set T=0
    r_mag=500.0,             # miss distance at TCA (m)
    v_mag=15.0,              # relative speed at TCA (m/s)
    conjunction_type="crossing",
    seed=42,                 # for reproducibility
)
```

Returns a dict with keys: `epoch_start`, `epoch_tca`, `sc1_eci_t0`, `sc2_eci_t0`, `sc1_eci_tca`, `sc2_eci_tca`, `miss_distance`, `rel_speed`.

---

## `tca.py` — TCA Finder

### Algorithm

**Step 1 — Coarse grid:** 200 evenly-spaced time nodes across the search window. Records miss distance at each.

**Step 2 — Fine minimization:** Brent's method within ±1 coarse step around best node. Achieves ~1-second timing precision.

**Note on slow overtaking:** Very low relative speeds (~5 m/s) produce multiple near-approach points in 24 hours. `find_tca` returns only the global minimum. This is documented as a known limitation (Case C in hard cases).

### Main functions

```python
from collision.tca import find_tca, get_states_at_tca

epoch_tca, miss_distance = find_tca(
    scenario["epoch_start"],
    scenario["sc1_eci_t0"],
    scenario["sc2_eci_t0"],
    window_hours=24.0,
    coarse_steps=200,
)

sc1_tca, sc2_tca = get_states_at_tca(
    scenario["epoch_start"],
    scenario["sc1_eci_t0"],
    scenario["sc2_eci_t0"],
    epoch_tca,
)
```

---

## `covariance.py` — RTN Covariance Generator

Builds 6×6 ECI covariance from diagonal RTN 1-sigma inputs. Default: σ_R=100m, σ_T=500m, σ_N=50m.

```python
from collision.covariance import generate_covariances

cov1, cov2 = generate_covariances(
    sc1_eci_tca, sc2_eci_tca,
    pos_std_rtn=(100.0, 500.0, 50.0),  # 1-sigma (R, T, N) in metres
    vel_std_rtn=(0.1, 0.5, 0.05),      # 1-sigma (R, T, N) in m/s
)
```

**Key simplification:** constructs the covariance at TCA from fixed assumptions. A real pipeline would propagate via STM from T=0.

---

## `fowler.py` — Analytic Pc (Fowler 1993)

Computes Pc by projecting the 3D problem onto the 2D encounter plane (perpendicular to v_rel), then numerically integrating the 2D Gaussian over the HBR disk.

```python
from collision.fowler import fowler_pc

pc = fowler_pc(sc1_eci, sc2_eci, cov1, cov2, hard_body_radius=10.0)
```

**Algorithm:**
1. Compute `r_rel = r1 − r2`, `v_rel = v1 − v2`
2. Build encounter-plane basis: `z_hat = v_rel/|v_rel|`, `x_hat` from r_rel component, `y_hat = z_hat × x_hat`
3. Project combined position covariance: `C_2d = B @ (C1[:3,:3] + C2[:3,:3]) @ B.T`
4. Project miss vector: `miss_2d = B @ r_rel`
5. Integrate 2D Gaussian over HBR disk via `scipy.integrate.dblquad` (~94 ms)

**Limitation:** at very small Pc (< 1e-10), `dblquad` underflows to 0.0. Use Chan for numerical stability.

---

## `chan1997.py` — Analytic Pc (Chan 1997)

Solves the same 2D Gaussian integral as Fowler but via the noncentral chi-squared CDF. ~1000× faster and numerically stable to Pc < 10⁻¹².

```python
from collision.chan1997 import chan_pc

pc = chan_pc(sc1_eci, sc2_eci, cov1, cov2, hard_body_radius=10.0)
```

**Algorithm:**
1. Same encounter-plane projection as Fowler
2. Diagonalize the 2D combined covariance: eigenvalues σ₁² ≤ σ₂²
3. Compute noncentrality parameter λ = (miss_x/σ₁)² + (miss_y/σ₁)² × (σ₁/σ₂)
4. Pc = (σ₁/σ₂) × ncx2.cdf(HBR²/σ₁², df=2, nc=λ)

**Anisotropy behavior:** Chan's leading-term approximation errors plateau at ~6% when σ₂/σ₁ ≥ 5, rather than growing unboundedly. Default covariance gives ~0.2% error.

**Agreement:** < 1% with Fowler on all tested scenarios (< 0.2% typical).

---

## `monte_carlo.py` — MC-2D Baseline

Samples from the 2D encounter-plane Gaussian and counts hits inside the HBR circle. Directly comparable to Fowler and Chan (same projection, same question).

```python
from collision.monte_carlo import monte_carlo_pc

pc, ci_low, ci_high = monte_carlo_pc(
    sc1_eci, sc2_eci, cov1, cov2,
    hard_body_radius=10.0,
    n_samples=1_000_000,
    seed=42,
)
```

**Returns:** `(pc, ci_low, ci_high)` — 95% confidence interval via normal approximation.

**Performance:** ~0.5–2 s at N=10⁶. Agrees with Chan to within 3× at Pc ~ 10⁻⁴.

**Limitation:** returns 0 when Pc < ~1/N. At Chan Pc ~ 5e-12, even N=10⁹ would give < 0.005 expected hits. Importance sampling is required for this regime.

---

## `monte_carlo_3d.py` — MC-3D Trajectory-Integrated

Perturbs full 6D states and propagates trajectories. Answers a stricter question than 2D: does the 3D trajectory come within HBR at any point in ±120 s around TCA?

```python
from collision.monte_carlo_3d import monte_carlo_3d_pc

pc, ci_low, ci_high = monte_carlo_3d_pc(
    sc1_eci, sc2_eci, cov1, cov2, epoch_tca,
    hard_body_radius=10.0,
    n_samples=10_000,
    seed=42,
)
```

**Algorithm:**
1. Build block-diagonal 12×12 joint covariance from cov1, cov2
2. Sample N perturbed 6D state pairs
3. For each pair: propagate ±120 s via two-body dynamics (200-step coarse grid + Brent refinement)
4. Count pairs where minimum 3D miss < HBR

**Why Brent is essential:** at 15 km/s head-on, the collision window is 0.66 ms. A 10 s coarse grid would never detect it.

**Head-on overestimation finding:** For head-on conjunctions, r_rel ∥ v_rel. The 2D encounter plane projects the 195 m miss to ~0.003 m (near zero), giving 2D Pc ~5e-3. The 3D method requires the trajectory to actually come within HBR in all 3 dimensions, giving Pc ~1.28e-3 at N=1M. The **2D methods overestimate by ~4× for head-on**. MC-3D(1M) CI = [1.208e-3, 1.352e-3], which excludes the 2D result by 27σ — this is a confirmed geometric difference, not a sampling artifact.

**Performance:** ~5 s at N=10k, ~50 s at N=100k, ~500 s at N=1M.

---

## Method Comparison Summary

| Method | What it computes | Speed | Regime |
|--------|-----------------|-------|--------|
| Fowler | 2D Gaussian integral, numerical | ~94 ms | All regimes; underflows at Pc < 1e-10 |
| Chan | 2D Gaussian integral, ncx2 CDF | ~0.09 ms | All regimes, stable to 1e-12 |
| MC-2D | 2D encounter-plane sampling | ~1 s (N=1M) | Pc > ~1e-6; comparable to Fowler/Chan |
| MC-3D | 3D trajectory-integrated | ~5 s (N=10k) | Strict 3D check; differs for head-on |

**When methods disagree:**
- Head-on (r_rel ∥ v_rel): MC-3D is more physically correct
- Tiny Pc (< 1e-8): only Chan is reliable; MC methods need importance sampling
- Anisotropic covariance (σ₂/σ₁ >> 5): Chan has ~6% fixed error vs Fowler

---

## Test Suite

Run with: `PYTHONPATH=src uv run pytest tests/ -v`

165 tests (1 skip) complete in ~175 seconds. Session-scoped fixtures mean expensive propagations happen once per `pytest` run.

### `conftest.py` — Shared Fixtures

**Scenarios:**
| Fixture | Parameters |
|---------|-----------|
| `crossing_scenario` | 500 m miss, 15 m/s, crossing, seed=42 |
| `head_on_scenario` | 200 m miss, v_rel ≈ 15 km/s, head-on, seed=7 |
| `overtaking_scenario` | 1000 m miss, 50 m/s, overtaking, seed=99 |
| `near_miss_scenario` | 10 m miss, 15 m/s, crossing, seed=123 |
| `high_pc_crossing_scenario` | 200 m miss, 500 m/s, crossing, seed=42 |
| `tiny_pc_scenario` | 500 m miss, 500 m/s, crossing, seed=42 (hard case A) |
| `slow_overtaking_scenario` | 500 m miss, 5 m/s, overtaking, seed=77 (hard case C) |

**TCA results:** `crossing_tca`, `head_on_tca`, `overtaking_tca`, `near_miss_tca`, `high_pc_crossing_tca`, `tiny_pc_tca`

**Covariances:** `crossing_covs`, `head_on_covs`, `overtaking_covs`, `near_miss_covs`, `high_pc_crossing_covs`, `tiny_pc_covs_tight`

**MC-2D (N=1M):** `crossing_mc`, `head_on_mc`, `near_miss_mc`, `high_pc_crossing_mc`

**MC-3D (N=10k):** `crossing_mc3d`, `overtaking_mc3d`, `head_on_mc3d`, `near_miss_mc3d`, `high_pc_mc3d`

---

### `test_hard_cases.py` — Hard Cases

#### `TestCaseA_TinyPc` — Pc < 1e-10 stress test

| Test | Checks |
|------|--------|
| `test_chan_pc_is_positive` | Chan returns positive value (numerically stable) |
| `test_chan_pc_is_tiny` | Chan Pc < 1e-6 with tight covariance |
| `test_fowler_underflows_or_positive` | Fowler returns 0 (underflow) or tiny positive < 1e-4 |
| `test_fowler_underflows_to_zero` | (skipped if Fowler nonzero) Verifies dblquad underflow |
| `test_mc_returns_zero` | MC-2D at N=1M returns 0 — insufficient samples |
| `test_chan_greater_than_mc` | Chan > MC when MC=0 (Chan is correct, MC is wrong) |
| `test_chan_fowler_ratio_when_fowler_nonzero` | Agreement < 100× when both nonzero |

#### `TestCaseB_ElongatedCovariance` — anisotropy error

| Test | Checks |
|------|--------|
| `test_b1_wide_ellipse_chan_positive` | Wide σ_T = 5000 m → crossing Pc positive |
| `test_b1_wide_ellipse_chan_fowler_ratio` | Chan/Fowler ratio within 100× |
| `test_b1_anisotropy_is_high` | σ₂/σ₁ > 10 confirmed |
| `test_b2_head_on_compact_chan_fowler_agree` | Head-on compact covariance: < 1% error |
| `test_b2_head_on_ratio_small` | σ₂/σ₁ < 5 |
| `test_b3_chan_vs_fowler_moderate_anisotropy` | σ₂/σ₁ ~ 10: error < 10% (plateaus at ~6%) |
| `test_b4_chan_vs_fowler_high_anisotropy` | σ₂/σ₁ ~ 100: error < 100% (bounded plateau) |

#### `TestCaseC_MultipleCloseApproaches` — multiple minima

| Test | Checks |
|------|--------|
| `test_slow_overtaking_tca_found` | find_tca returns a valid epoch |
| `test_slow_overtaking_single_minimum_documented` | Miss distance is positive (single result) |
| `test_slow_overtaking_fowler_finite` | Fowler Pc ∈ [0, 1] for found TCA |
| `test_slow_overtaking_chan_finite` | Chan Pc ∈ [0, 1] for found TCA |
| `test_slow_overtaking_chan_fowler_agree` | Chan vs Fowler < 1% for found TCA |

---

## Design Decisions Worth Knowing

**Why crossing at 500 m/15 m/s produces Pc ~ 4e-4 despite large miss?**
The encounter plane is perpendicular to v_rel (≈ N direction for crossing). The large σ_T=500 m does NOT project onto the encounter plane — it's perpendicular to it. Only σ_R=100 m and σ_N=50 m project in, giving a compact 2D covariance → moderate Pc.

**Why head-on at 195 m produces Pc ~ 5e-3 (2D) but ~1.28e-3 (3D)?**
Head-on: v_rel ≈ 2× orbital speed, so r_rel (≈200 m along T) is nearly parallel to v_rel. The encounter-plane projection of r_rel ≈ 0 → zero projected miss → high 2D Pc. MC-3D requires the 3D trajectory to come within HBR, accounting for the full 200 m separation → lower Pc.

**Why does the overtaking fixture use 50 m/s not 5 m/s?**
At 5 m/s, 15 local minima appear in 24h. 50 m/s gives an unambiguous single TCA matching the planted one, making test assertions straightforward.

**Why is covariance constructed at TCA not propagated from T=0?**
Sufficient for testing Pc algorithms in isolation. A real pipeline would propagate via STM, inflating along-track uncertainty substantially over 24 hours.

**Why does Chan anisotropy error plateau at ~6% not grow unboundedly?**
Chan's leading-term correction uses σ₁/σ₂ to rescale the CDF. Once σ₂/σ₁ >> 1, the CDF argument saturates and the correction's relative error stabilizes. Empirically: error peaks near σ₂/σ₁ = 5 and stays at ~6% through σ₂/σ₁ = 500.
