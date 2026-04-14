# Next Steps: Probability of Collision Implementation

This document is a handoff for the next development session. It describes what has been built, what comes next, and the specific implementation plan for each Pc method.

---

## What Exists Already

### Code (`src/collision/`)

| File | Purpose |
|------|---------|
| `conjunction.py` | Generate synthetic LEO conjunction scenarios with known ground truth. Back-propagates from a designed TCA to produce T=0 initial conditions. |
| `tca.py` | Find TCA from T=0 initial conditions. Sequential coarse grid + Brent fine search. Returns TCA epoch and miss distance. |

### Tests (`tests/`)
- `conftest.py` — session-scoped fixtures for 4 scenarios + cached TCA results
- `test_conjunction.py` — 26 tests
- `test_tca.py` — 13 tests
- All 39 tests pass in ~26 seconds

### Dependencies (`pyproject.toml`)
```
brahe>=1.3.3
numpy>=2.4.4
scipy>=1.15.0
pytest>=9.0.3 (dev)
```

Run tests: `uv run pytest tests/ -v`

---

## What Needs to Be Built

Three Pc methods, in this order:

### 1. Fowler Method (`src/collision/fowler.py`) ← do this first

The analytic 2D-projection method from Fowler (1993). Fast and widely used in operations.

**Inputs (at TCA):**
- `sc1_eci`: `[6,]` ECI state of SC1 (m, m/s)
- `sc2_eci`: `[6,]` ECI state of SC2 (m, m/s)
- `cov1`: `[6,6]` position-velocity covariance of SC1 in ECI (m², m²/s², m·m/s)
- `cov2`: `[6,6]` position-velocity covariance of SC2 in ECI (m², m²/s², m·m/s)
- `hard_body_radius`: combined physical radius of both objects (m) — typically 5–20 m

**Algorithm:**
1. Compute relative position `r_rel = r1 - r2` and velocity `v_rel = v1 - v2` at TCA
2. Build the **encounter plane** — the plane perpendicular to `v_rel`:
   - `z_hat = v_rel / |v_rel|`
   - `x_hat = r_rel - (r_rel · z_hat) * z_hat`, then normalize
   - `y_hat = z_hat × x_hat`
3. Extract 3×3 position blocks from each covariance, combine: `C_pos = C1[:3,:3] + C2[:3,:3]`
4. Project combined covariance onto encounter plane (2×2 matrix):
   - `B = np.array([x_hat, y_hat])` — shape (2, 3)
   - `C_2d = B @ C_pos @ B.T` — shape (2, 2)
5. Project miss vector onto encounter plane:
   - `miss_2d = B @ r_rel` — shape (2,)
6. Integrate 2D Gaussian over hard-body disk:
   - `Pc = ∫∫_{x²+y²≤R²} N(x | miss_2d, C_2d) dx dy`
   - Use `scipy.integrate.dblquad` for the numerical integral
   - Alternatively use the Chan (1997) series expansion for speed

**Reference:** Fowler, W.T. (1993). "A Probability of Collision Analysis for Conjunction Assessments." AAS 93-177.

**What to add to the scenario:**
`conjunction.py` does not currently generate covariances. You'll need to either:
- Add a `generate_covariances()` helper that produces realistic diagonal position/velocity covariances in the RTN frame and rotates them to ECI, or
- Accept covariances as explicit inputs to the Fowler function and use simple test covariances in the tests

Typical LEO position uncertainty: 100–1000 m (1σ) along-track, 10–100 m cross-track/radial.

**Tests to write (`tests/test_fowler.py`):**
- Zero covariance → Pc should be 0 if miss > HBR, 1 if miss < HBR
- Very large covariance → Pc approaches `(π R²) / (2π det(C)^0.5)` (geometric limit)
- Symmetric covariance + zero miss → Pc depends only on HBR vs covariance spread
- Pc is in [0, 1]
- Pc increases as miss distance decreases (monotonicity)
- Pc increases as HBR increases
- Compare across all 4 scenario fixtures

---

### 2. Monte Carlo Method (`src/collision/monte_carlo.py`) ← do this second

Estimates Pc by sampling from the joint position distribution at TCA and counting how many samples fall within the hard-body radius.

**Algorithm:**
1. Sample `N` relative position vectors from `N(r_rel, C_pos_combined)`
2. Count samples where `|r_sample| < hard_body_radius`
3. `Pc ≈ count / N`

**Notes:**
- Needs large N for small Pc values (e.g. N=10⁶ for Pc~10⁻⁴)
- Can use `np.random.multivariate_normal` for sampling
- Should return both the estimate and a confidence interval

**Tests:** compare Monte Carlo result to Fowler for the same inputs — they should agree within statistical uncertainty. Use a fixed random seed for reproducibility.

---

### 3. MCMC Method (`src/collision/mcmc.py`) ← do this third

Uses Markov Chain Monte Carlo (importance sampling or Metropolis-Hastings) to concentrate samples near the hard-body sphere. More efficient than naive Monte Carlo for small Pc values.

**Notes:**
- Good library options: `scipy` (manual MH), or `emcee` (ensemble sampler)
- The target distribution is the collision indicator weighted by the Gaussian
- Main advantage over Monte Carlo: doesn't waste samples far from the collision region

---

## Covariance Generation (needed before any Pc method works)

You need to add covariance generation to `conjunction.py` or as a separate helper. Here's a suggested approach:

```python
def generate_covariances(sc1_eci_tca, sc2_eci_tca,
                         pos_std_rtn=(100.0, 500.0, 50.0),   # R, T, N in meters
                         vel_std_rtn=(0.1, 0.5, 0.05)):      # R, T, N in m/s
    """
    Generate diagonal RTN covariances and rotate to ECI.
    Returns cov1, cov2 as (6,6) arrays.
    """
```

RTN covariances are diagonal in the RTN frame (position errors are mostly along-track),
then rotate to ECI using `brahe.rotation_eci_to_rtn(sc1_eci_tca)`.

---

## Key brahe Functions Available

```python
from brahe import (
    state_eci_to_rtn,        # convert relative ECI state to RTN
    rotation_eci_to_rtn,     # rotation matrix ECI → RTN (for covariance rotation)
    rotation_rtn_to_eci,     # inverse
    state_koe_to_eci,        # Keplerian elements → ECI state
    NumericalOrbitPropagator,
    NumericalPropagationConfig,
    ForceModelConfig,
    Epoch,
)
```

---

## Suggested File Layout When Done

```
src/collision/
    __init__.py
    conjunction.py    ✓ done
    tca.py            ✓ done
    covariance.py     ← new: generate/rotate covariances
    fowler.py         ← new: analytic 2D Gaussian integral
    monte_carlo.py    ← new: naive Monte Carlo
    mcmc.py           ← new: MCMC importance sampling

tests/
    conftest.py       ✓ done (will need covariance fixtures added)
    test_conjunction.py  ✓ done
    test_tca.py          ✓ done
    test_fowler.py    ← new
    test_monte_carlo.py  ← new
    test_mcmc.py         ← new
```

---

## Context on Test Performance

The test suite runs in ~26 seconds. The expensive operations (24-hour orbit propagations) are cached in session-scoped fixtures in `conftest.py`. When adding new tests:
- Put any new expensive computations (e.g. Monte Carlo with N=10⁶) in session-scoped fixtures
- Don't call `find_tca` directly in individual tests — use the cached `*_tca` fixtures
