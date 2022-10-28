import re

from s1reader import load_bursts
from s1reader.s1_orbit import get_orbit_file_from_dir

def test_burst_from_zip(bursts):
    assert len(bursts) == 9


def test_anx_crossing(test_paths):
    zip_path = test_paths.data_dir / "S1A_IW_SLC__1SDV_20221024T184148_20221024T184218_045587_05735F_D6E2.zip"
    orbit_file = get_orbit_file_from_dir(zip_path, test_paths.orbit_dir)
    bursts = load_bursts(zip_path, orbit_file, 2, pol="vv")
    burst_id_pat = r"t(?P<track>\d{3})_(?P<burst_id>\d{6})_iw(?P<subswath_num>[1-3])"

    # t015_032217_iw2
    match = re.match(burst_id_pat, bursts[0].burst_id)
    assert match
    assert False
    # First burst is in track 15
    assert int(match.group("track")) == 15

    # t016_032226_iw2
    # Last burst should be in track 16
    match = re.match(burst_id_pat, bursts[0].burst_id)
    assert match
    assert int(match.group("track")) == 16