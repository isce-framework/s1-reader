import datetime
import pytest

import isce3

from s1reader.s1_orbit import get_orbit_file_from_dir

def test_get_orbit_file(test_paths):
    orbit_file = get_orbit_file_from_dir(test_paths.safe, test_paths.orbit_dir)

    expected_orbit_path = f'{test_paths.orbit_dir}/{test_paths.orbit_file}'
    assert orbit_file == expected_orbit_path

def test_orbit_datetime(bursts):
    # pad in seconds used in orbit_reader
    pad = datetime.timedelta(seconds=60)
    for burst in bursts:
        start_minus_pad = isce3.core.DateTime(burst.sensing_start - pad)
        stop_plus_pad = isce3.core.DateTime(burst.sensing_stop + pad)
        ref_epoch = burst.orbit.reference_epoch
        for t in burst.orbit.time:
            t = ref_epoch + isce3.core.TimeDelta(t)
            assert t > start_minus_pad
            assert t < stop_plus_pad
