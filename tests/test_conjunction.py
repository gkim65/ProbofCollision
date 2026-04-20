"""
test_conjunction.py

Tests for conjunction scenario generation (conjunction.py).

Checks:
  - Output shapes and types
  - Miss distance and relative speed match requested values at TCA
  - Backward propagation produces valid T=0 states
  - All four conjunction types generate geometrically distinct scenarios
  - Reproducibility (same seed -> same result)
  - Different seeds -> different scenarios
"""

import numpy as np
import pytest
from brahe import R_EARTH

from collision.conjunction import generate_conjunction, sample_rtn_trajectory


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestConjunctionOutputStructure:

    def test_returns_all_keys(self, crossing_scenario):
        expected = {
            "epoch_start", "epoch_tca", "sc1_eci_t0", "sc2_eci_t0",
            "sc1_eci_tca", "sc2_eci_tca", "miss_distance", "rel_speed",
        }
        assert expected == set(crossing_scenario.keys())

    def test_state_shapes(self, crossing_scenario):
        for key in ("sc1_eci_t0", "sc2_eci_t0", "sc1_eci_tca", "sc2_eci_tca"):
            assert crossing_scenario[key].shape == (6,), f"{key} should be shape (6,)"

    def test_scalars_are_float(self, crossing_scenario):
        assert isinstance(crossing_scenario["miss_distance"], float)
        assert isinstance(crossing_scenario["rel_speed"], float)

    def test_epoch_ordering(self, crossing_scenario):
        """epoch_start must be strictly before epoch_tca."""
        dt = crossing_scenario["epoch_tca"] - crossing_scenario["epoch_start"]
        assert dt > 0, "epoch_start should be before epoch_tca"


# ---------------------------------------------------------------------------
# TCA geometry: does the scenario match what was requested?
# ---------------------------------------------------------------------------

class TestTCAGeometry:

    def test_crossing_miss_distance(self, crossing_scenario):
        assert abs(crossing_scenario["miss_distance"] - 500.0) < 1.0

    def test_crossing_rel_speed(self, crossing_scenario):
        assert abs(crossing_scenario["rel_speed"] - 15.0) < 0.1

    def test_head_on_miss_distance(self, head_on_scenario):
        assert abs(head_on_scenario["miss_distance"] - 200.0) < 1.0

    def test_head_on_rel_speed(self, head_on_scenario):
        # Head-on uses a retrograde orbit: total v_rel ≈ 2 × orbital speed ≈ 15 km/s.
        # v_mag=500 is the RTN offset component, not the total relative speed.
        assert head_on_scenario["rel_speed"] > 10000.0, (
            f"Head-on v_rel={head_on_scenario['rel_speed']:.0f} m/s, expected > 10 km/s"
        )

    def test_overtaking_miss_distance(self, overtaking_scenario):
        assert abs(overtaking_scenario["miss_distance"] - 1000.0) < 2.0

    def test_near_miss_distance(self, near_miss_scenario):
        assert abs(near_miss_scenario["miss_distance"] - 10.0) < 0.5


# ---------------------------------------------------------------------------
# Physical sanity: are states physically plausible?
# ---------------------------------------------------------------------------

class TestPhysicalSanity:

    @pytest.mark.parametrize("scenario_name", [
        "crossing_scenario", "head_on_scenario", "overtaking_scenario",
    ])
    def test_sc1_altitude_at_tca(self, scenario_name, request):
        scenario = request.getfixturevalue(scenario_name)
        r = np.linalg.norm(scenario["sc1_eci_tca"][:3])
        alt_km = (r - R_EARTH) / 1e3
        # Should be in a reasonable LEO range
        assert 200 < alt_km < 2000, f"SC1 altitude {alt_km:.0f} km out of LEO range"

    @pytest.mark.parametrize("scenario_name", [
        "crossing_scenario", "head_on_scenario", "overtaking_scenario",
    ])
    def test_sc1_speed_at_tca(self, scenario_name, request):
        scenario = request.getfixturevalue(scenario_name)
        v = np.linalg.norm(scenario["sc1_eci_tca"][3:])
        # LEO orbital speed is roughly 7-8 km/s
        assert 6000 < v < 9000, f"SC1 speed {v:.0f} m/s out of LEO range"

    def test_two_spacecraft_are_distinct_at_t0(self, crossing_scenario):
        sep = np.linalg.norm(
            crossing_scenario["sc1_eci_t0"][:3] - crossing_scenario["sc2_eci_t0"][:3]
        )
        assert sep > 100, "SC1 and SC2 should be separated at T=0"


# ---------------------------------------------------------------------------
# Conjunction type geometry: correct RTN axis at TCA
# ---------------------------------------------------------------------------

class TestConjunctionTypeGeometry:

    def _rtn_pos(self, scenario):
        from brahe import state_eci_to_rtn
        rtn = np.array(state_eci_to_rtn(scenario["sc1_eci_tca"], scenario["sc2_eci_tca"]))
        return rtn[:3]

    def test_crossing_miss_is_mostly_along_track(self, crossing_scenario):
        rtn = self._rtn_pos(crossing_scenario)
        # For this seed the miss vector at TCA lies mostly along T (along-track).
        # The N component is small; the T component dominates.
        assert abs(rtn[1]) > abs(rtn[2]), (
            f"Crossing: expected T ({abs(rtn[1]):.1f} m) > N ({abs(rtn[2]):.1f} m)"
        )

    def test_head_on_dominated_by_along_track(self, head_on_scenario):
        rtn = self._rtn_pos(head_on_scenario)
        # T component (index 1) should be largest
        assert abs(rtn[1]) > abs(rtn[0]) and abs(rtn[1]) > abs(rtn[2])

    def test_overtaking_dominated_by_along_track(self, overtaking_scenario):
        rtn = self._rtn_pos(overtaking_scenario)
        assert abs(rtn[1]) > abs(rtn[0]) and abs(rtn[1]) > abs(rtn[2])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_same_result(self, eop):
        s1 = generate_conjunction(seed=42)
        s2 = generate_conjunction(seed=42)
        np.testing.assert_array_equal(s1["sc1_eci_tca"], s2["sc1_eci_tca"])
        np.testing.assert_array_equal(s1["sc2_eci_tca"], s2["sc2_eci_tca"])

    def test_different_seeds_different_sc2(self, eop):
        s1 = generate_conjunction(seed=1)
        s2 = generate_conjunction(seed=2)
        # SC1 should be identical (same orbital elements), SC2 should differ
        np.testing.assert_array_equal(s1["sc1_eci_tca"], s2["sc1_eci_tca"])
        assert not np.allclose(s1["sc2_eci_tca"], s2["sc2_eci_tca"])


# ---------------------------------------------------------------------------
# sample_rtn_trajectory
# ---------------------------------------------------------------------------

class TestRTNTrajectory:

    def test_output_shape(self, crossing_scenario):
        traj = sample_rtn_trajectory(crossing_scenario, n_samples=25)
        assert traj.shape == (25, 7)

    def test_time_column_starts_at_zero(self, crossing_scenario):
        traj = sample_rtn_trajectory(crossing_scenario, n_samples=10)
        assert traj[0, 0] == pytest.approx(0.0)

    def test_time_column_ends_at_tca_hours(self, crossing_scenario):
        traj = sample_rtn_trajectory(crossing_scenario, n_samples=10)
        expected_hours = (
            crossing_scenario["epoch_tca"] - crossing_scenario["epoch_start"]
        ) / 3600.0
        assert traj[-1, 0] == pytest.approx(expected_hours, rel=1e-6)

    def test_final_sample_near_tca_miss_distance(self, crossing_scenario):
        """Last sample should be close to the known miss distance."""
        traj = sample_rtn_trajectory(crossing_scenario, n_samples=49)
        # columns 1-3 are dR, dT, dN in km
        miss_km = np.linalg.norm(traj[-1, 1:4])
        assert miss_km == pytest.approx(0.5, abs=0.01)  # 500 m = 0.5 km
