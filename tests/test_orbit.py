import pytest

from sentinel1_reader.sentinel1_orbit_reader import get_orbit_file_from_dir

def test_get_orbit_file(test_paths):
    orbit_file = get_orbit_file_from_dir(test_paths.safe, test_paths.orbit_dir)

    expected_orbit_path = f'{test_paths.orbit_dir}/{test_paths.orbit_file}'
    assert orbit_file == expected_orbit_path
