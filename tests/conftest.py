"""
conftest.py — shared fixtures for the test suite.

EOP is initialized once per session since it hits the filesystem.
Conjunction scenarios and TCA searches are computed once per session and
reused — each involves full 24-hour orbit propagations and is expensive.
"""

import numpy as np
import pytest
import brahe

from collision.conjunction import generate_conjunction
from collision.tca import find_tca, get_states_at_tca


@pytest.fixture(scope="session", autouse=True)
def eop():
    """Initialize Earth Orientation Parameters once for the whole test session."""
    brahe.initialize_eop()


# ---------------------------------------------------------------------------
# Conjunction scenarios
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def crossing_scenario(eop):
    """Standard crossing conjunction: 500 m miss, 15 m/s relative speed."""
    return generate_conjunction(
        tca_hours=24.0, r_mag=500.0, v_mag=15.0,
        conjunction_type="crossing", seed=42,
    )


@pytest.fixture(scope="session")
def head_on_scenario(eop):
    """Head-on conjunction: 200 m miss, 500 m/s relative speed (typical LEO head-on)."""
    return generate_conjunction(
        tca_hours=24.0, r_mag=200.0, v_mag=500.0,
        conjunction_type="head-on", seed=7,
    )


@pytest.fixture(scope="session")
def overtaking_scenario(eop):
    """Overtaking conjunction: 1000 m miss, 50 m/s relative speed.

    Note: very slow overtaking speeds (~5 m/s) produce multiple close-approach
    minima over a 24-hour window, making TCA finding ambiguous. 50 m/s is the
    lower end of realistic overtaking conjunctions and gives an unambiguous TCA.
    """
    return generate_conjunction(
        tca_hours=24.0, r_mag=1000.0, v_mag=50.0,
        conjunction_type="overtaking", seed=99,
    )


@pytest.fixture(scope="session")
def near_miss_scenario(eop):
    """Very close approach: 10 m miss distance — stress-test for Pc methods."""
    return generate_conjunction(
        tca_hours=24.0, r_mag=10.0, v_mag=15.0,
        conjunction_type="crossing", seed=123,
    )


# ---------------------------------------------------------------------------
# Cached TCA results — one find_tca call per scenario for the whole session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def crossing_tca(crossing_scenario):
    epoch, miss = find_tca(
        crossing_scenario["epoch_start"],
        crossing_scenario["sc1_eci_t0"],
        crossing_scenario["sc2_eci_t0"],
    )
    s1, s2 = get_states_at_tca(
        crossing_scenario["epoch_start"],
        crossing_scenario["sc1_eci_t0"],
        crossing_scenario["sc2_eci_t0"],
        epoch,
    )
    return dict(epoch=epoch, miss=miss, sc1=s1, sc2=s2)


@pytest.fixture(scope="session")
def head_on_tca(head_on_scenario):
    epoch, miss = find_tca(
        head_on_scenario["epoch_start"],
        head_on_scenario["sc1_eci_t0"],
        head_on_scenario["sc2_eci_t0"],
    )
    s1, s2 = get_states_at_tca(
        head_on_scenario["epoch_start"],
        head_on_scenario["sc1_eci_t0"],
        head_on_scenario["sc2_eci_t0"],
        epoch,
    )
    return dict(epoch=epoch, miss=miss, sc1=s1, sc2=s2)


@pytest.fixture(scope="session")
def overtaking_tca(overtaking_scenario):
    epoch, miss = find_tca(
        overtaking_scenario["epoch_start"],
        overtaking_scenario["sc1_eci_t0"],
        overtaking_scenario["sc2_eci_t0"],
    )
    s1, s2 = get_states_at_tca(
        overtaking_scenario["epoch_start"],
        overtaking_scenario["sc1_eci_t0"],
        overtaking_scenario["sc2_eci_t0"],
        epoch,
    )
    return dict(epoch=epoch, miss=miss, sc1=s1, sc2=s2)


@pytest.fixture(scope="session")
def near_miss_tca(near_miss_scenario):
    epoch, miss = find_tca(
        near_miss_scenario["epoch_start"],
        near_miss_scenario["sc1_eci_t0"],
        near_miss_scenario["sc2_eci_t0"],
    )
    return dict(epoch=epoch, miss=miss)
