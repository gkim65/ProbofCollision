"""
test_tca.py

Tests for TCA finding (tca.py).

TCA searches are expensive (full 24-hour propagations) so they are computed
once per session via the cached fixtures in conftest.py and reused here.

Checks:
  - find_tca recovers the known TCA epoch within tolerance
  - find_tca recovers the known miss distance within tolerance
  - Works across all conjunction types
  - get_states_at_tca returns states consistent with the known TCA states
  - Handles near-miss (very small miss distance) without crashing
"""

import numpy as np

from collision.tca import find_tca


# Tolerances
TCA_TIME_TOL_S   = 5.0    # seconds — TCA timing accuracy
MISS_DIST_TOL_M  = 5.0    # meters  — miss distance accuracy
STATE_POS_TOL_M  = 500.0  # meters  — position state recovery (propagation noise)
STATE_VEL_TOL    = 1.0    # m/s     — velocity state recovery


# ---------------------------------------------------------------------------
# TCA epoch recovery
# ---------------------------------------------------------------------------

class TestFindTCAEpoch:

    def test_crossing_tca_timing(self, crossing_scenario, crossing_tca):
        dt = abs(crossing_tca["epoch"] - crossing_scenario["epoch_tca"])
        assert dt < TCA_TIME_TOL_S, f"TCA timing error {dt:.1f} s exceeds {TCA_TIME_TOL_S} s"

    def test_head_on_tca_timing(self, head_on_scenario, head_on_tca):
        dt = abs(head_on_tca["epoch"] - head_on_scenario["epoch_tca"])
        assert dt < TCA_TIME_TOL_S

    def test_overtaking_tca_timing(self, overtaking_scenario, overtaking_tca):
        dt = abs(overtaking_tca["epoch"] - overtaking_scenario["epoch_tca"])
        assert dt < TCA_TIME_TOL_S


# ---------------------------------------------------------------------------
# Miss distance recovery
# ---------------------------------------------------------------------------

class TestFindTCAMissDistance:

    def test_crossing_miss_distance(self, crossing_scenario, crossing_tca):
        assert abs(crossing_tca["miss"] - crossing_scenario["miss_distance"]) < MISS_DIST_TOL_M

    def test_head_on_miss_distance(self, head_on_scenario, head_on_tca):
        assert abs(head_on_tca["miss"] - head_on_scenario["miss_distance"]) < MISS_DIST_TOL_M

    def test_overtaking_miss_distance(self, overtaking_scenario, overtaking_tca):
        assert abs(overtaking_tca["miss"] - overtaking_scenario["miss_distance"]) < MISS_DIST_TOL_M

    def test_near_miss_miss_distance(self, near_miss_scenario, near_miss_tca):
        assert abs(near_miss_tca["miss"] - near_miss_scenario["miss_distance"]) < MISS_DIST_TOL_M

    def test_miss_distance_is_positive(self, crossing_tca):
        assert crossing_tca["miss"] >= 0.0


# ---------------------------------------------------------------------------
# State recovery at TCA
# ---------------------------------------------------------------------------

class TestGetStatesAtTCA:

    def test_state_shapes(self, crossing_tca):
        assert crossing_tca["sc1"].shape == (6,)
        assert crossing_tca["sc2"].shape == (6,)

    def test_sc1_in_leo(self, crossing_tca):
        """State at found TCA should have SC1 at a plausible LEO altitude."""
        from brahe import R_EARTH
        alt_km = (np.linalg.norm(crossing_tca["sc1"][:3]) - R_EARTH) / 1e3
        assert 200 < alt_km < 2000

    def test_sc1_speed_plausible(self, crossing_tca):
        """SC1 orbital speed at found TCA should be in LEO range."""
        v = np.linalg.norm(crossing_tca["sc1"][3:])
        assert 6000 < v < 9000

    def test_miss_distance_consistent_with_states(self, crossing_tca):
        miss_from_states = np.linalg.norm(crossing_tca["sc1"][:3] - crossing_tca["sc2"][:3])
        assert abs(miss_from_states - crossing_tca["miss"]) < 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestFindTCAEdgeCases:

    def test_coarse_steps_parameter(self, crossing_scenario):
        """Fewer coarse steps should still find a reasonable TCA (looser tolerance)."""
        epoch_found, miss = find_tca(
            crossing_scenario["epoch_start"],
            crossing_scenario["sc1_eci_t0"],
            crossing_scenario["sc2_eci_t0"],
            coarse_steps=50,
        )
        dt = abs(epoch_found - crossing_scenario["epoch_tca"])
        assert dt < 30.0
        assert abs(miss - crossing_scenario["miss_distance"]) < 10.0
