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
from collision.covariance import generate_covariances


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


@pytest.fixture(scope="session")
def high_pc_crossing_scenario(eop):
    """Crossing conjunction at 200 m miss, 500 m/s — gives operationally-interesting Pc.

    The standard crossing fixture (500 m, 15 m/s) produces Pc ~ 1e-14 because the
    slow relative speed makes the encounter-plane projection very wide.  This fixture
    uses a faster relative speed so the projected covariance stays compact and Pc lands
    in the 1e-5 to 1e-4 range with default covariances.
    """
    return generate_conjunction(
        tca_hours=24.0, r_mag=200.0, v_mag=500.0,
        conjunction_type="crossing", seed=42,
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


@pytest.fixture(scope="session")
def high_pc_crossing_tca(high_pc_crossing_scenario):
    epoch, miss = find_tca(
        high_pc_crossing_scenario["epoch_start"],
        high_pc_crossing_scenario["sc1_eci_t0"],
        high_pc_crossing_scenario["sc2_eci_t0"],
    )
    s1, s2 = get_states_at_tca(
        high_pc_crossing_scenario["epoch_start"],
        high_pc_crossing_scenario["sc1_eci_t0"],
        high_pc_crossing_scenario["sc2_eci_t0"],
        epoch,
    )
    return dict(epoch=epoch, miss=miss, sc1=s1, sc2=s2)


# ---------------------------------------------------------------------------
# Covariance fixtures — one generate_covariances call per scenario
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def crossing_covs(crossing_tca):
    """Realistic diagonal RTN covariances for the crossing scenario."""
    return generate_covariances(crossing_tca["sc1"], crossing_tca["sc2"])


@pytest.fixture(scope="session")
def head_on_covs(head_on_tca):
    """Realistic diagonal RTN covariances for the head-on scenario."""
    return generate_covariances(head_on_tca["sc1"], head_on_tca["sc2"])


@pytest.fixture(scope="session")
def overtaking_covs(overtaking_tca):
    """Realistic diagonal RTN covariances for the overtaking scenario."""
    return generate_covariances(overtaking_tca["sc1"], overtaking_tca["sc2"])


@pytest.fixture(scope="session")
def high_pc_crossing_covs(high_pc_crossing_tca):
    """Default RTN covariances for the high-Pc crossing scenario."""
    return generate_covariances(high_pc_crossing_tca["sc1"], high_pc_crossing_tca["sc2"])


@pytest.fixture(scope="session")
def near_miss_covs(near_miss_scenario):
    """Realistic diagonal RTN covariances for the near-miss scenario.

    Uses scenario TCA states directly (no cached get_states_at_tca for near_miss).
    """
    return generate_covariances(
        near_miss_scenario["sc1_eci_tca"],
        near_miss_scenario["sc2_eci_tca"],
    )
