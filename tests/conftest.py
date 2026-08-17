import pathlib
import types

import pytest

from s1reader import s1_reader


@pytest.fixture(scope="session")
def test_paths():
    test_paths = types.SimpleNamespace()

    test_path = pathlib.Path(__file__).parent.resolve()
    test_paths.safe = f"{test_path}/data/S1A_IW_SLC__1SDV_20200511T135117_20200511T135144_032518_03C421_7768.zip"
    test_paths.orbit_dir = f"{test_path}/data/orbits"
    test_paths.orbit_file = (
        "S1A_OPER_AUX_POEORB_OPOD_20210318T120818_V20200510T225942_20200512T005942.EOF"
    )

    return test_paths


@pytest.fixture(scope="session")
def bursts(test_paths):
    i_subswath = 3
    pol = "vv"

    orbit_path = f"{test_paths.orbit_dir}/{test_paths.orbit_file}"
    bursts = s1_reader.load_bursts(test_paths.safe, orbit_path, i_subswath, pol)

    return bursts


@pytest.fixture(scope="session")
def ew_test_paths():
    ew_test_paths = types.SimpleNamespace()

    test_path = pathlib.Path(__file__).parent.resolve()
    ew_test_paths.safe = f"{test_path}/data/S1A_EW_SLC__1SDH_20220330T185405_20220330T185511_042554_051380_3E95.zip"
    ew_test_paths.orbit_dir = f"{test_path}/data/orbits"
    ew_test_paths.orbit_file = (
        "S1A_OPER_AUX_POEORB_OPOD_20220419T081726_V20220329T225942_20220331T005942.EOF"
    )

    return ew_test_paths


@pytest.fixture(scope="session")
def ew3_bursts(ew_test_paths):
    subswath = 3
    pol = "hh"

    orbit_path = f"{ew_test_paths.orbit_dir}/{ew_test_paths.orbit_file}"
    return s1_reader.load_bursts(ew_test_paths.safe, orbit_path, subswath, pol)


@pytest.fixture(scope="session")
def ew_bursts_by_subswath(ew_test_paths):
    """Returns a dict {subswath_num: [bursts]} for all 5 EW subswaths."""
    pol = "hh"
    orbit_path = f"{ew_test_paths.orbit_dir}/{ew_test_paths.orbit_file}"
    return {
        i: s1_reader.load_bursts(ew_test_paths.safe, orbit_path, i, pol)
        for i in range(1, 6)
    }
