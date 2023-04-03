import datetime

import isce3
import numpy as np
from shapely.geometry import Point

from s1reader.s1_orbit import get_orbit_file_from_dir


def test_get_orbit_file(test_paths):
    orbit_file = get_orbit_file_from_dir(test_paths.safe, test_paths.orbit_dir)

    expected_orbit_path = f'{test_paths.orbit_dir}/{test_paths.orbit_file}'
    assert orbit_file == expected_orbit_path


def test_get_orbit_file_multi_mission(tmp_path):
    orbit_a = tmp_path / "S1A_OPER_AUX_POEORB_OPOD_20210314T131617_V20191007T225942_20191009T005942.EOF"
    orbit_a.write_text("")
    orbit_b = tmp_path / "S1B_OPER_AUX_POEORB_OPOD_20210304T232500_V20191007T225942_20191009T005942.EOF"
    orbit_b.write_text("")

    zip_path = tmp_path / "zips"
    zip_path.mkdir()
    zip_a = zip_path / "S1A_IW_SLC__1SDV_20191008T005936_20191008T010003_018377_0229E5_909C.zip"
    zip_a.write_text("")
    zip_b = zip_path / "S1B_IW_SLC__1SDV_20191008T005936_20191008T010003_018377_0229E5_909C.zip"
    zip_b.write_text("")

    # Test S1A zip file
    assert get_orbit_file_from_dir(zip_a, orbit_dir=tmp_path) == str(orbit_a)
    assert get_orbit_file_from_dir(zip_b, orbit_dir=tmp_path) == str(orbit_b)


def test_orbit_datetime(bursts):
    # pad in seconds used in orbit_reader
    pad = datetime.timedelta(seconds=60)
    for burst in bursts:
        # check if orbit times within burst +/- 60 sec pad
        start_minus_pad = isce3.core.DateTime(burst.sensing_start - pad)
        stop_plus_pad = isce3.core.DateTime(burst.sensing_stop + pad)
        ref_epoch = burst.orbit.reference_epoch
        for t in burst.orbit.time:
            t = ref_epoch + isce3.core.TimeDelta(t)
            assert t > start_minus_pad
            assert t < stop_plus_pad

        # check if middle of the burst in radar coordinates
        r0 = burst.starting_range + 0.5 * burst.width * burst.range_pixel_spacing
        t0 = isce3.core.DateTime(burst.sensing_mid) - burst.orbit.reference_epoch

        dem = isce3.geometry.DEMInterpolator(0)
        dop = 0.0
        rdr_grid = burst.as_isce3_radargrid()
        llh = isce3.geometry.rdr2geo(t0.total_seconds(), r0, burst.orbit,
                                     rdr_grid.lookside, dop,
                                     rdr_grid.wavelength, dem)
        pnt = Point(np.degrees(llh[0]), np.degrees(llh[1]))
        assert(burst.border[0].contains(pnt))
