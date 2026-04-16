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

All outputs are gitignored PNGs — regenerate with `PYTHONPATH=src uv run python plots/<script>.py`.

| Script | Output(s) | What it shows |
|--------|-----------|---------------|
| `plot_conjunctions.py` | `conjunctions_<name>.png` × 4 | 4-panel per scenario: RTN trajectory, miss distance vs time, encounter plane with σ-ellipses and HBR disk, RTN covariance cross-sections |
| `plot_3d_orbits.py` | `conjunctions_3d_orbits.png`, `conjunctions_3d_<name>.png` × 4 | 3D RTN relative motion — SC1 at origin, SC2 path colored light→dark red as time approaches TCA |
| `plot_3d_eci.py` | `conjunctions_eci_<name>.png` × 4, `conjunctions_eci_all.png` | Per scenario: left = full ECI orbit + Earth sphere + velocity arrows; right = zoomed ±2–10 sec around TCA showing the two orbit arcs separating and the miss vector |
| `plot_scenario_comparison.py` | `comparison_*.png` × 4 | Side-by-side comparisons: miss distance all scenarios, crossing vs near-miss encounter plane, head-on vs crossing RTN 3D, overtaking arc geometry |
| `plot_pc_findings.py` | `pc_findings.png` | Pc sensitivity: vs covariance scale, vs miss distance at varying v_rel, crossing vs head-on geometry |

### Key fixtures in `conftest.py`

| Fixture | Scenario |
|---------|----------|
| `crossing_scenario/tca/covs` | 500 m miss, 7000 m/s N-dominant crossing — very low Pc, good for TCA geometry tests |
| `head_on_scenario/tca/covs` | 200 m miss, retrograde orbit, v_rel ≈ 15 km/s — operationally interesting Pc |
| `overtaking_scenario/tca/covs` | 1000 m miss, 15 m/s slow catch-up |
| `near_miss_scenario/tca/covs` | 10 m miss, 500 m/s crossing — stress test, Pc ~1e-3 |
| `high_pc_crossing_scenario/tca/covs` | 200 m miss, 500 m/s crossing — only realistic crossing with operationally-interesting Pc |

Run tests: `uv run pytest tests/ -v`

---

## What Needs to Be Built Next

### Priority 1 — Library-Grade Module Interfaces

Before adding new Pc methods, clean up the existing modules so they are suitable for eventual contribution to brahe or standalone library release.

**`conjunction.py`:**
- Remove hardcoded module-level constants (`TCA_HOURS`, `R_ALT`, `INCL`, etc.) — callers should always pass explicit values
- `SC1_OE_AT_TCA` and `EPOCH_TCA` should become named defaults in function signatures, not globals
- Rename internal helpers with clearer names (`_sc2_rtn_at_tca` → `_place_sc2_rtn`)
- Add `__all__` listing the public API

**`tca.py`:**
- Expose `_miss_distance_from` as a private but documented helper (useful for external testing)
- Add type annotations throughout
- Consider a `TCAResult` named tuple instead of a bare `(epoch, float)` return

**`covariance.py`:**
- Rename `generate_covariances` → `build_covariances_from_rtn` (more descriptive)
- Accept a single spacecraft state + return a single covariance (let callers combine) — current 2-spacecraft API is convenient but limits composability
- Document units explicitly in function signature (m, m/s → m², m²/s²)

**`fowler.py`:**
- Rename to `fowler1993.py` to make the method + year naming pattern explicit (all future Pc methods will follow `author_year.py`)
- Extract `_build_encounter_plane_basis` as a standalone testable helper
- Add `__all__`

**General:**
- Add `py.typed` marker file to `src/collision/` for mypy compatibility
- Ensure all public functions have complete NumPy-style docstrings (Args, Returns, Raises, Notes)

---

### Priority 2 — Chan (1997) Series Expansion (`src/collision/chan1997.py`)

Closed-form series expansion for the Fowler 2D integral. Faster than `dblquad` and numerically stable at very small Pc (< 10⁻¹⁰) where double-precision quadrature loses accuracy.

**Reference:** Chan, F.K. (1997). "Spacecraft Collision Probability." AAS 97-173.

**Algorithm:**
The Chan series expresses Pc as a sum over modified Bessel functions of the projected 2D Gaussian. For well-conditioned (non-degenerate) covariances it converges in ~10–20 terms.

1. Project covariance and miss vector onto encounter plane (same as Fowler)
2. Diagonalize `C_2d` via eigendecomposition → principal axes `(σ₁, σ₂)` and rotation angle θ
3. Transform miss vector into principal frame
4. Sum the series: `Pc = exp(-u²/2σ₁² - v²/2σ₂²) * Σ Aₙ Iₙ(...)` (see Chan 1997 eq. 15)
5. Truncate when term < 10⁻¹⁵ × running sum

**Why add this:**
- ~10× faster than `dblquad` for typical Pc values
- Handles Pc < 10⁻¹⁰ correctly (quadrature underflows)
- Standard reference method — needed for any methods-comparison paper
- Natural validation target: `chan1997_pc` and `fowler1993_pc` should agree to < 0.1% for all scenarios

**Tests (`tests/test_chan1997.py`):**
- Agrees with Fowler to within 0.1% for all 5 scenario fixtures
- Handles near-zero Pc (< 10⁻¹⁰) without underflow
- Handles large Pc (> 0.9) correctly
- Symmetric: swapping SC1/SC2 gives same result
- Raises `ValueError` on zero relative speed (same contract as Fowler)

---

### Priority 3 — Monte Carlo (`src/collision/monte_carlo.py`)

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

### Priority 4 — MCMC / Importance Sampling (`src/collision/importance_sampling.py`)

Importance sampling to concentrate samples near the hard-body sphere. Efficient for Pc values too small for naive Monte Carlo (< 10⁻⁶).

**Algorithm:**
Rather than sampling from `N(r_rel, C_pos)` (the prior), draw samples from a proposal distribution `q(x)` centered near the hard-body boundary and reweight:

`Pc ≈ (1/N) Σ [|xᵢ| < HBR] * p(xᵢ) / q(xᵢ)`

Good proposals: uniform disk of radius 2×HBR centered at the closest point on the HBR surface to r_rel, or a Gaussian centered at that point.

**Notes:**
- `emcee` ensemble sampler is an alternative (pure Python, no extra dependencies beyond scipy)
- Main advantage: accurate Pc estimates at 10⁻⁶ to 10⁻¹⁰ without requiring N=10⁸ samples
- Validation: compare against Chan series at low Pc where Monte Carlo fails

---

### Priority 5 — Patera (2001) Line Integral (`src/collision/patera2001.py`)

Alternative analytic method that avoids the 2D Gaussian projection by instead integrating along the miss distance vector.

**Reference:** Patera, R.P. (2001). "General Method for Calculating Satellite Collision Probability." Journal of Guidance, Control, and Dynamics.

**Why add this:** Patera and Fowler/Chan are the two main analytic approaches in the literature. Comparing them on the same scenario suite is a core part of a survey paper — they should agree for typical scenarios and diverge in known edge cases (e.g., highly elongated covariances).

---

### Priority 6 — Covariance Propagation via STM (`src/collision/covariance_propagation.py`)

Propagate a covariance from T=0 to TCA using a state transition matrix (STM), which is the physically correct approach for real conjunction data.

**Algorithm:**
1. Numerically propagate the reference trajectory from T=0 to TCA
2. Compute the Jacobian `Φ = ∂x_TCA/∂x_0` via finite differences (perturb each of the 6 state components, propagate, measure the change)
3. `C_TCA = Φ @ C_0 @ Φ.T`

**Why this matters:** Along-track uncertainty inflates by 10–100× over 24 hours due to differential drag and gravitational perturbations. The current `generate_covariances` constructs covariances directly at TCA, bypassing this inflation. A real pipeline must propagate covariances forward.

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
