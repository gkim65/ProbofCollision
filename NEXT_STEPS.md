# Next Steps: Probability of Collision Implementation

Handoff document for the next development session.

---

## What Exists Already

### Code (`src/collision/`)

| File | Purpose |
|------|---------|
| `conjunction.py` | Generate synthetic LEO conjunction scenarios with known ground truth. Back-propagates from a designed TCA to produce T=0 initial conditions. |
| `tca.py` | Find TCA from T=0 initial conditions. Sequential coarse grid + Brent fine search. Returns TCA epoch and miss distance. |
| `covariance.py` | Build 6×6 ECI covariances from diagonal RTN 1-sigma inputs. Rotates via `brahe.rotation_rtn_to_eci`. Constructs at TCA (not propagated from T=0 — see Known Simplifications). |
| `fowler.py` | Fowler (1993) analytic Pc. Projects combined covariance onto encounter plane, integrates 2D Gaussian over hard-body disk via `scipy.integrate.dblquad`. |

### Tests (`tests/`)

| File | Count | What it covers |
|------|-------|----------------|
| `conftest.py` | — | Session-scoped fixtures: 5 scenarios, 5 TCA results, 5 covariance pairs |
| `test_conjunction.py` | 26 | Output structure, geometry accuracy, physical sanity, RTN trajectory |
| `test_tca.py` | 13 | TCA timing, miss distance accuracy, state retrieval, edge cases |
| `test_fowler.py` | 31 | Covariance generation, Pc properties, limit cases, magnitude ranges, covariance-size effects |
| **Total** | **70** | All pass in ~35 seconds |

### Plots (`plots/`)

| File | What it shows |
|------|---------------|
| `plot_pc_findings.py` | 3-panel: Pc vs covariance scale, Pc vs miss distance (vary v_rel), crossing vs head-on geometry comparison |
| `pc_findings.png` | Output of above (gitignored — regenerate with `PYTHONPATH=src uv run python plots/plot_pc_findings.py`) |

### Key fixtures in `conftest.py`

| Fixture | Scenario |
|---------|----------|
| `crossing_scenario/tca/covs` | 500 m miss, 15 m/s — very low Pc (~2e-14), good for TCA geometry tests |
| `head_on_scenario/tca/covs` | 200 m miss, 500 m/s — Pc ~5e-4, operationally interesting |
| `overtaking_scenario/tca/covs` | 1000 m miss, 50 m/s — Pc ~2e-4 |
| `near_miss_scenario/tca/covs` | 10 m miss, 15 m/s — Pc ~1e-3, stress test |
| `high_pc_crossing_scenario/tca/covs` | 200 m miss, 500 m/s crossing — Pc ~3e-5, only realistic crossing with interesting Pc |

Run tests: `uv run pytest tests/ -v`

---

## What Needs to Be Built Next

### Priority 1 — Conjunction Visualization (`plots/plot_conjunctions.py`)

Before moving to Monte Carlo, add a plot script that visualizes the conjunction scenarios. This makes the test suite more interpretable and helps debug future Pc methods.

**Suggested figures (one script, multi-panel output):**

1. **RTN trajectory** — for each scenario, plot dR, dT, dN vs time over the 24-hour window using `sample_rtn_trajectory()`. Shows the characteristic shape of each conjunction type (crossing: N component passes through zero; head-on: T component passes through zero).

2. **Miss distance vs time** — scalar separation |r_rel| over the window with a vertical line at the found TCA. Confirms the TCA finder is landing at the actual minimum.

3. **Encounter plane projection** — at TCA, show the 2D Gaussian contours (±1σ, ±2σ, ±3σ ellipses from `C_2d`) and the HBR disk projected onto the encounter plane. Directly illustrates why head-on gives higher Pc than crossing — the ellipse is compact for head-on, elongated for crossing.

4. **Covariance ellipses in RTN** — show the position uncertainty ellipsoid projected onto the R-T, R-N, and T-N planes. Illustrates anisotropy (large T, small R and N).

`sample_rtn_trajectory()` in `conjunction.py` already returns the data for plots 1 and 2. The encounter plane data for plot 3 can be computed from the TCA states and covariances already in the fixtures using the same projection logic as `fowler_pc`.

---

### Priority 2 — Monte Carlo (`src/collision/monte_carlo.py`)

Estimates Pc by sampling from the joint position distribution at TCA.

**Algorithm:**
1. Compute `r_rel`, `C_pos = C1[:3,:3] + C2[:3,:3]` at TCA
2. Sample N relative position vectors from `N(r_rel, C_pos)`
3. Count samples where `|r_sample| < hard_body_radius`
4. `Pc_mc ≈ count / N`

**Implementation notes:**
- Use `np.random.default_rng(seed).multivariate_normal` for reproducibility
- Return `(pc_estimate, ci_low, ci_high)` — Wilson or normal confidence interval
- N=10⁶ is sufficient for Pc~10⁻⁴; need N=10⁸ for Pc~10⁻⁶ (impractical — this is why MCMC matters)

**Tests (`tests/test_monte_carlo.py`):**
- Pc ∈ [0, 1]
- Agrees with Fowler within ~2–3× for head-on and near-miss scenarios (large enough Pc for Monte Carlo to be accurate at N=10⁶)
- Fixed seed gives identical results
- Larger N reduces confidence interval width
- Put N=10⁶ runs in session-scoped fixtures to avoid re-running per test

---

### Priority 3 — MCMC (`src/collision/mcmc.py`)

Importance sampling or Metropolis-Hastings to concentrate samples near the hard-body sphere. Efficient for Pc values too small for naive Monte Carlo.

**Notes:**
- Good starting point: importance sampling with a proposal centered near the HBR boundary rather than at r_rel
- Alternative: `emcee` ensemble sampler (pure Python, no extra dependencies beyond scipy)
- Main advantage: accurate Pc estimates at 10⁻⁶ to 10⁻¹⁰ without requiring N=10⁸ samples

---

## Known Simplifications

### Covariance at TCA is not propagated from T=0

`generate_covariances` constructs covariances directly at TCA from fixed RTN 1-sigma assumptions. In a real pipeline, the covariance at T=0 would be propagated forward to TCA via a state transition matrix (STM) — along-track uncertainty inflates substantially over 24 hours.

This is fine for testing Pc methods in isolation. When building a pipeline that ingests real catalog data:
- Implement `propagate_covariance(epoch_start, eci_state, cov_t0, epoch_tca)` using a finite-difference STM Jacobian around the propagated trajectory
- Wire it in place of `generate_covariances`

### Fowler integration is numerical, not the Chan series

`fowler.py` uses `scipy.integrate.dblquad`. The Chan (1997) series expansion is a faster closed-form alternative. For very small Pc (< 10⁻¹⁰), `dblquad` can lose precision; the series expansion handles these cases better. Consider replacing if precision at very small Pc becomes an issue.

---

## Performance Reference

| Operation | Time |
|-----------|------|
| Full test suite (70 tests) | ~35 s |
| Single `find_tca` call | ~3–4 s (200 propagations) |
| `generate_covariances` | < 1 ms |
| `fowler_pc` (dblquad) | ~50–200 ms depending on Pc magnitude |
| `plot_pc_findings.py` | ~75 s |

Session-scoped fixtures in `conftest.py` mean each expensive operation runs once per `pytest` session. Always add new expensive computations (Monte Carlo with N=10⁶, etc.) to session-scoped fixtures.

---

## Observed Pc Reference Values

Default covariance (pos σ = 100/500/50 m R/T/N), HBR = 10 m:

| Scenario | Miss (m) | v_rel (m/s) | Pc |
|----------|----------|-------------|----|
| slow crossing | 500 | 15 | ~2e-14 |
| high-Pc crossing | 204 | 500 | ~3e-5 |
| head-on | 201 | 500 | ~5e-4 |
| overtaking | 1000 | 50 | ~2e-4 |
| near-miss | 10 | 15 | ~1e-3 |

See FINDINGS.md for why slow crossing Pc is so much lower than head-on despite similar miss distances.
