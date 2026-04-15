# Development Notes: Probability of Collision

This document explains the codebase structure, how to use each module, and what every test is checking and why.

---

## Project Structure

```
src/collision/
    conjunction.py   — generate synthetic conjunction scenarios
    tca.py           — find Time of Closest Approach from initial states
    covariance.py    — generate RTN covariances and rotate to ECI
    fowler.py        — Fowler (1993) analytic Pc via 2D Gaussian integral

tests/
    conftest.py         — shared fixtures (scenarios, TCA results, covariances)
    test_conjunction.py
    test_tca.py
    test_fowler.py
```

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

The three conjunction types place SC2 at TCA along different RTN axes:

| Type | Relative position at TCA | Physical meaning |
|------|--------------------------|-----------------|
| `"crossing"` | +N direction | SC2 comes from below/above the orbital plane |
| `"head-on"` | −T direction | SC2 approaches from behind (opposing traffic) |
| `"overtaking"` | +T direction | SC2 is ahead, being caught up to |

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

Returns a dict:

| Key | Type | Description |
|-----|------|-------------|
| `epoch_start` | `Epoch` | T=0 (24h before TCA) |
| `epoch_tca` | `Epoch` | Time of closest approach |
| `sc1_eci_t0` | `ndarray [6]` | SC1 ECI state at T=0 (m, m/s) |
| `sc2_eci_t0` | `ndarray [6]` | SC2 ECI state at T=0 (m, m/s) |
| `sc1_eci_tca` | `ndarray [6]` | SC1 ECI state at TCA (m, m/s) |
| `sc2_eci_tca` | `ndarray [6]` | SC2 ECI state at TCA (m, m/s) |
| `miss_distance` | `float` | Separation at TCA (m) |
| `rel_speed` | `float` | Relative speed at TCA (m/s) |

### Helper: `sample_rtn_trajectory()`

Propagates both spacecraft forward from T=0 and records their RTN relative state at `n_samples` evenly-spaced times. Useful for plotting how the conjunction geometry evolves over the 24-hour window.

```python
from collision.conjunction import sample_rtn_trajectory

traj = sample_rtn_trajectory(scenario, n_samples=49)
# shape (49, 7): [t_hours, dR_km, dT_km, dN_km, dVR_kms, dVT_kms, dVN_kms]
```

---

## `tca.py` — TCA Finder

### What it does

In a real scenario you only have T=0 states — you don't know when closest approach occurs. `find_tca` searches a time window for the minimum miss distance.

### Algorithm

**Step 1 — Coarse grid:**
A single pair of propagators is stepped forward through 200 evenly-spaced time nodes across the search window. Miss distance is recorded at each node. This is O(T) total integration work — much faster than creating fresh propagators on every evaluation.

**Step 2 — Fine minimization:**
Brent's method (scipy `minimize_scalar`) searches within ±1 coarse step around the best node, anchored at that node's state so it only needs to integrate a short arc (~7 minutes). Achieves ~1-second timing precision.

### Main function: `find_tca()`

```python
from collision.tca import find_tca

epoch_tca, miss_distance = find_tca(
    scenario["epoch_start"],
    scenario["sc1_eci_t0"],
    scenario["sc2_eci_t0"],
    window_hours=24.0,   # search window
    coarse_steps=200,    # grid resolution (tradeoff: accuracy vs speed)
)
```

Returns:
- `epoch_tca`: brahe `Epoch` of closest approach
- `miss_distance`: separation at TCA in meters

**Note on slow overtaking conjunctions:** Very low relative speeds (~5 m/s) can produce multiple near-approach points over 24 hours, making TCA ambiguous. Realistic conjunctions have relative speeds ≥ 50 m/s.

### Helper: `get_states_at_tca()`

Propagates both spacecraft from T=0 to a given epoch and returns their ECI states. Use this after `find_tca` to get the states needed for Pc calculations.

```python
from collision.tca import get_states_at_tca

sc1_tca, sc2_tca = get_states_at_tca(
    scenario["epoch_start"],
    scenario["sc1_eci_t0"],
    scenario["sc2_eci_t0"],
    epoch_tca,
)
# sc1_tca, sc2_tca: ndarray [6] (m, m/s)
```

### Typical usage flow

```python
import brahe
from collision.conjunction import generate_conjunction
from collision.tca import find_tca, get_states_at_tca

brahe.initialize_eop()

# 1. Generate a scenario with known ground truth
scenario = generate_conjunction(r_mag=200.0, v_mag=500.0, conjunction_type="head-on")

# 2. Find TCA from T=0 initial conditions
epoch_tca, miss = find_tca(scenario["epoch_start"], scenario["sc1_eci_t0"], scenario["sc2_eci_t0"])
print(f"TCA miss distance: {miss:.1f} m")

# 3. Get ECI states at TCA for Pc calculation
sc1_tca, sc2_tca = get_states_at_tca(scenario["epoch_start"], scenario["sc1_eci_t0"], scenario["sc2_eci_t0"], epoch_tca)

# 4. Compute Pc
from collision.covariance import generate_covariances
from collision.fowler import fowler_pc

cov1, cov2 = generate_covariances(sc1_tca, sc2_tca)
pc = fowler_pc(sc1_tca, sc2_tca, cov1, cov2, hard_body_radius=10.0)
print(f"Pc = {pc:.3e}")
```

---

## Test Suite

Run with:
```
uv run pytest tests/ -v
```

All 70 tests complete in ~35 seconds. The session-scoped fixtures mean the expensive propagations (scenario generation + TCA search) happen once per `pytest` run, not once per test.

### `conftest.py` — Shared Fixtures

Everything expensive is computed once and reused:

| Fixture | What it creates |
|---------|----------------|
| `eop` | Initializes Earth Orientation Parameters (required by brahe) |
| `crossing_scenario` | 500 m miss, 15 m/s, out-of-plane geometry |
| `head_on_scenario` | 200 m miss, 500 m/s, opposing along-track |
| `overtaking_scenario` | 1000 m miss, 50 m/s, same-direction along-track |
| `near_miss_scenario` | 10 m miss, 15 m/s — stress test for Pc methods |
| `crossing_tca` | Cached `find_tca` + `get_states_at_tca` for crossing |
| `head_on_tca` | Same for head-on |
| `overtaking_tca` | Same for overtaking |
| `near_miss_tca` | Cached `find_tca` for near-miss |
| `high_pc_crossing_scenario` | 200 m miss, 500 m/s crossing — produces operationally-interesting Pc (~3e-5) |
| `high_pc_crossing_tca` | Cached `find_tca` + `get_states_at_tca` for high-Pc crossing |
| `crossing_covs` | `generate_covariances` result for crossing TCA states |
| `head_on_covs` | Same for head-on |
| `overtaking_covs` | Same for overtaking |
| `near_miss_covs` | Same for near-miss (uses scenario TCA states directly) |
| `high_pc_crossing_covs` | Same for high-Pc crossing |

---

### `test_conjunction.py`

#### `TestConjunctionOutputStructure` — does the function return the right shape of data?

| Test | Checks |
|------|--------|
| `test_returns_all_keys` | Dict has exactly the 8 expected keys |
| `test_state_shapes` | All four ECI state arrays are shape `(6,)` |
| `test_scalars_are_float` | `miss_distance` and `rel_speed` are Python floats |
| `test_epoch_ordering` | `epoch_start` is strictly before `epoch_tca` |

#### `TestTCAGeometry` — does the scenario match the requested parameters?

| Test | Checks |
|------|--------|
| `test_crossing_miss_distance` | Miss distance within 1 m of requested 500 m |
| `test_crossing_rel_speed` | Relative speed within 0.1 m/s of requested 15 m/s |
| `test_head_on_miss_distance` | Miss distance within 1 m of requested 200 m |
| `test_head_on_rel_speed` | Relative speed within 1 m/s of requested 500 m/s |
| `test_overtaking_miss_distance` | Miss distance within 2 m of requested 1000 m |
| `test_near_miss_distance` | Miss distance within 0.5 m of requested 10 m |

#### `TestPhysicalSanity` — are the generated states physically plausible?

| Test | Checks |
|------|--------|
| `test_sc1_altitude_at_tca[*]` | SC1 altitude is between 200–2000 km (parametrized over 3 scenarios) |
| `test_sc1_speed_at_tca[*]` | SC1 orbital speed is between 6–9 km/s (parametrized over 3 scenarios) |
| `test_two_spacecraft_are_distinct_at_t0` | SC1 and SC2 are at least 100 m apart at T=0 |

#### `TestConjunctionTypeGeometry` — is the RTN geometry correct for each type?

| Test | Checks |
|------|--------|
| `test_crossing_dominated_by_normal` | N component of relative position is largest at TCA |
| `test_head_on_dominated_by_along_track` | T component (negative) is largest at TCA |
| `test_overtaking_dominated_by_along_track` | T component (positive) is largest at TCA |

#### `TestReproducibility` — does the random seed work correctly?

| Test | Checks |
|------|--------|
| `test_same_seed_same_result` | Same seed → identical SC1 and SC2 TCA states |
| `test_different_seeds_different_sc2` | SC1 is identical (same orbital elements), SC2 differs between seeds |

#### `TestRTNTrajectory` — does `sample_rtn_trajectory` work correctly?

| Test | Checks |
|------|--------|
| `test_output_shape` | Returns array of shape `(n_samples, 7)` |
| `test_time_column_starts_at_zero` | First time sample is t=0 |
| `test_time_column_ends_at_tca_hours` | Last time sample matches window length |
| `test_final_sample_near_tca_miss_distance` | Last sample's RTN separation matches known miss distance (within 10 m) |

---

### `test_tca.py`

#### `TestFindTCAEpoch` — does `find_tca` find the right time?

| Test | Checks |
|------|--------|
| `test_crossing_tca_timing` | Found TCA within 5 seconds of generated TCA |
| `test_head_on_tca_timing` | Same for head-on (500 m/s, sharp miss distance profile) |
| `test_overtaking_tca_timing` | Same for overtaking (50 m/s) |

#### `TestFindTCAMissDistance` — does `find_tca` find the right miss distance?

| Test | Checks |
|------|--------|
| `test_crossing_miss_distance` | Found miss within 5 m of generated 500 m |
| `test_head_on_miss_distance` | Found miss within 5 m of generated 200 m |
| `test_overtaking_miss_distance` | Found miss within 5 m of generated 1000 m |
| `test_near_miss_miss_distance` | Found miss within 5 m of generated 10 m |
| `test_miss_distance_is_positive` | Miss distance is non-negative |

#### `TestGetStatesAtTCA` — does `get_states_at_tca` return valid states?

| Test | Checks |
|------|--------|
| `test_state_shapes` | Both returned arrays are shape `(6,)` |
| `test_sc1_in_leo` | SC1 at found TCA is between 200–2000 km altitude |
| `test_sc1_speed_plausible` | SC1 orbital speed is in LEO range |
| `test_miss_distance_consistent_with_states` | Separation between returned states matches miss distance from `find_tca` (within 1 m) |

#### `TestFindTCAEdgeCases` — robustness checks

| Test | Checks |
|------|--------|
| `test_coarse_steps_parameter` | With only 50 coarse steps (vs default 200), TCA is still found within 30 s and 10 m — verifies graceful degradation |

---

## Design Decisions Worth Knowing

**Why generate TCA at t=24h (window end)?**
It's the most natural place — you're saying "I know TCA is in 24 hours, what are the initial conditions?" Having TCA at the boundary also stress-tests the TCA finder's ability to handle boundary minima, which requires the fine search to extend slightly past the window.

**Why 50 m/s for the overtaking test fixture?**
At very low relative speeds (~5 m/s), two nearly co-orbital satellites can have multiple near-approach points over a 24-hour window. `find_tca` correctly finds *a* minimum — that is the TCA for that scenario, and Pc methods can work with it just fine. The test fixture uses 50 m/s simply so the found TCA matches the one we planted at t=24h, making the test assertion straightforward. For future work: multi-minima scenarios are worth exploring explicitly — the "right" TCA to use for Pc may not always be the global minimum (e.g. if a later closer approach is more operationally relevant).

**Why are TCA tests cached in conftest.py?**
Each `find_tca` call involves 200 numerical orbit propagations. Without caching, 13 TCA tests × 1 call each = 13 expensive computations. With caching, it's 4 computations total (one per scenario), shared across all tests.

**Why do slow-encounter scenarios (crossing, overtaking) produce very low Pc?**
The Fowler method projects the 3D position covariance onto the 2D encounter plane — the plane perpendicular to `v_rel`. When `v_rel` is small (15–50 m/s), the encounter plane is nearly parallel to the orbit, so the large along-track uncertainty (500 m 1σ) projects almost entirely onto that plane, creating a very spread-out 2D Gaussian. A 10 m HBR disk covers a tiny fraction of that distribution → Pc is extremely small. Fast head-on encounters (500 m/s) project the covariance more compactly onto the encounter plane, giving higher Pc for similar miss distances.

**Covariance generation is a synthetic test harness, not a propagated covariance.**
In real conjunction assessment, the covariance at T=0 would be propagated forward to TCA using a state transition matrix (STM), which inflates and rotates the uncertainty — especially along-track. `generate_covariances` instead constructs a covariance directly at TCA from fixed RTN 1-sigma assumptions. This is correct for testing `fowler_pc` in isolation (the algorithm doesn't care how the covariance was obtained), but a full end-to-end pipeline would need proper covariance propagation. See NEXT_STEPS.md for details.

---

## `covariance.py` — RTN Covariance Generator

### What it does

Builds a 6×6 position-velocity covariance in ECI from diagonal uncertainties specified in the RTN frame. This is the standard representation used by conjunction assessment tools — tracking errors are naturally expressed in RTN (largest along-track, smaller radial and cross-track) and then rotated to ECI for Pc computation.

### Main function: `generate_covariances()`

```python
from collision.covariance import generate_covariances

cov1, cov2 = generate_covariances(
    sc1_eci_tca,                      # [6,] ECI state of SC1 at TCA
    sc2_eci_tca,                      # [6,] ECI state of SC2 at TCA
    pos_std_rtn=(100.0, 500.0, 50.0), # 1-sigma (R, T, N) in metres
    vel_std_rtn=(0.1, 0.5, 0.05),     # 1-sigma (R, T, N) in m/s
)
# cov1, cov2: (6, 6) arrays in ECI
```

### Algorithm

1. Build a 6×6 diagonal covariance in RTN from the squared 1-sigma values
2. Fetch the 3×3 rotation matrix from `brahe.rotation_rtn_to_eci(eci_state)`
3. Assemble a 6×6 block-diagonal rotation `T` (same 3×3 block for position and velocity)
4. Rotate: `C_eci = T @ C_rtn @ T.T`

The rotation preserves the trace of the position block (sum of position variances is invariant), which the tests verify.

**Key simplification:** this constructs the covariance *at TCA* from fixed assumptions rather than propagating it from T=0. Sufficient for testing Pc methods; a real pipeline would propagate the initial covariance via STM.

---

## `fowler.py` — Analytic Pc (Fowler 1993)

### What it does

Computes probability of collision at TCA using the analytic 2D-projection method. It reduces the 3D collision problem to a 2D integral by projecting everything onto the encounter plane — the plane perpendicular to the relative velocity vector at TCA.

### Main function: `fowler_pc()`

```python
from collision.fowler import fowler_pc

pc = fowler_pc(
    sc1_eci,          # [6,] ECI state of SC1 at TCA
    sc2_eci,          # [6,] ECI state of SC2 at TCA
    cov1,             # [6,6] ECI covariance of SC1
    cov2,             # [6,6] ECI covariance of SC2
    hard_body_radius, # combined physical radius (m), typically 5–20 m
)
# returns float in [0, 1]
```

### Algorithm

1. Compute `r_rel = r1 − r2`, `v_rel = v1 − v2` at TCA
2. Build encounter-plane basis (orthonormal):
   - `z_hat = v_rel / |v_rel|` (along relative velocity, normal to plane)
   - `x_hat` = component of `r_rel` perpendicular to `z_hat`, normalised (toward miss vector)
   - `y_hat = z_hat × x_hat` (completes right-hand frame)
   - Special case: if `r_rel ∥ v_rel` (zero impact parameter), `x_hat` is chosen arbitrarily perpendicular to `z_hat`
3. Project combined position covariance onto the 2D plane: `C_2d = B @ (C1[:3,:3] + C2[:3,:3]) @ B.T`
4. Project miss vector: `miss_2d = B @ r_rel`
5. Numerically integrate the 2D Gaussian over the hard-body disk using `scipy.integrate.dblquad`

Raises `ValueError` if `|v_rel| == 0` (encounter plane is undefined).

---

## `test_fowler.py`

### `TestCovarianceGeneration` — does `generate_covariances` produce valid covariances?

| Test | Checks |
|------|--------|
| `test_output_shapes` | Both outputs are shape `(6, 6)` |
| `test_symmetric` | Both matrices are symmetric to numerical precision |
| `test_positive_definite` | All eigenvalues are positive |
| `test_position_variance_order_of_magnitude` | ECI diagonal entries are in the expected range for the default RTN sigmas |
| `test_custom_std` | Trace of position block equals sum of input variances (rotation preserves trace) |

### `TestFowlerPcBasicProperties` — do the inputs affect Pc in the right direction?

| Test | Checks |
|------|--------|
| `test_pc_in_unit_interval` | Pc ∈ [0, 1] for the crossing scenario |
| `test_pc_increases_as_miss_decreases` | Closer miss → higher Pc with same covariance |
| `test_pc_increases_as_hbr_increases` | Larger HBR → higher Pc |
| `test_pc_increases_as_covariance_grows` | Wider covariance → higher Pc (up to saturation) |
| `test_pc_returns_float` | Return type is `float` |

### `TestFowlerPcLimits` — does Pc behave correctly at the extremes?

| Test | Checks |
|------|--------|
| `test_zero_miss_small_cov_near_one` | Zero miss + HBR >> sigma → Pc > 0.99 |
| `test_large_miss_tiny_cov_near_zero` | Miss >> HBR, sigma << miss → Pc < 1e-10 |
| `test_symmetric_zero_miss_depends_only_on_hbr_and_sigma` | Different HBR gives different Pc at zero miss |
| `test_large_covariance_geometric_limit` | Larger sigma lowers Pc in the geometric (uniform-Gaussian) limit |

### `TestFowlerPcScenarios` — do all scenario fixtures produce sensible results?

| Test | Checks |
|------|--------|
| `test_crossing_pc_in_range` | Pc ∈ [0, 1] |
| `test_head_on_pc_in_range` | Pc ∈ [0, 1] |
| `test_overtaking_pc_in_range` | Pc ∈ [0, 1] |
| `test_near_miss_pc_higher_than_crossing` | 10 m miss has higher Pc than 500 m miss |
| `test_pc_increases_with_hbr[*]` | HBR=50 m gives higher Pc than HBR=5 m (parametrized over 3 scenarios) |
| `test_slow_crossing_pc_is_very_low` | Slow crossing (500 m, 15 m/s) Pc < 1e-10 — large projected covariance |
| `test_high_pc_crossing_magnitude` | Fast crossing (204 m, 500 m/s) Pc in [1e-7, 1e-2] |
| `test_head_on_pc_magnitude` | Head-on Pc (HBR=10 m) is in [1e-6, 1e-1] — compact projected covariance |
| `test_overtaking_pc_magnitude` | Overtaking Pc (HBR=10 m) is in [1e-7, 1e-1] |
| `test_near_miss_pc_magnitude` | Near-miss Pc (HBR=10 m) is in [1e-5, 1e-1] — miss distance ≈ HBR |

### `TestCovarianceMagnitudeEffects` — does Pc respond correctly to varying covariance size?

| Test | Checks |
|------|--------|
| `test_head_on_pc_monotone_decreasing_with_covariance` | Head-on Pc decreases monotonically tight→default→loose→very_loose |
| `test_high_pc_crossing_pc_non_monotone_with_covariance` | Crossing Pc is non-monotone: near-zero at tight, peaks near loose, falls at very_loose |
| `test_near_miss_pc_monotone_decreasing_with_covariance` | Near-miss Pc decreases monotonically across all levels |
| `test_all_covariance_levels_produce_valid_pc` | Pc ∈ [0, 1] at every covariance level |

### `TestFowlerPcDegenerate` — error handling

| Test | Checks |
|------|--------|
| `test_zero_relative_speed_raises` | `ValueError` raised when `v_rel == 0` (no encounter plane) |
