import pathlib
import types

import pandas as pd
import pytest
from shapely import wkt

from s1reader import s1_reader


@pytest.fixture(scope="session")
def test_paths():
    data_dir = pathlib.Path(__file__).parent.resolve() / "data"

    test_paths = types.SimpleNamespace()
    test_paths.data_dir = data_dir
    test_paths.safe = data_dir / "S1A_IW_SLC__1SDV_20200511T135117_20200511T135144_032518_03C421_7768.zip"  # noqa
    test_paths.orbit_dir = data_dir / "orbits"
    test_paths.orbit_file = "S1A_OPER_AUX_POEORB_OPOD_20210318T120818_V20200510T225942_20200512T005942.EOF"  # noqa

    return test_paths


@pytest.fixture(scope="session")
def bursts(test_paths):
    i_subswath = 3
    pol = "vv"

    orbit_path = f"{test_paths.orbit_dir}/{test_paths.orbit_file}"
    bursts = s1_reader.load_bursts(test_paths.safe, orbit_path, i_subswath, pol)

    return bursts


@pytest.fixture(scope="session")
def esa_burst_db(test_paths):
    """Load the sample of the ESA burst database."""
    db_path = test_paths.data_dir / "esa_burst_db_sample.csv"
    df = pd.read_csv(db_path)
    df["geometry"] = df.geometry.apply(wkt.loads)
    return df
