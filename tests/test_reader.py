import re
import zipfile
from pathlib import Path

from lxml import etree as ET
from shapely.geometry import MultiPolygon

from s1reader import load_bursts
from s1reader.s1_orbit import get_orbit_file_from_dir
from s1reader.s1_reader import get_start_end_track

BURST_ID_PAT = r"t(?P<track>\d{3})_(?P<burst_id>\d{6})_iw(?P<subswath_num>[1-3])"


def test_all_files(test_paths, esa_burst_db):
    # Hawaii dataset

    # "S1A_IW_SLC__1SDV_20220828T042306_20220828T042335_044748_0557C6_F396.zip"
    files = Path(test_paths.data_dir).glob("S1*.zip")
    for f in files:
        _compare_bursts_to_esa(
            f, test_paths, esa_burst_db
        )


def _compare_bursts_to_esa(zip_name, test_paths, esa_burst_db):
    """Check that the burst IDs and computed geometries match ESA's."""
    zip_path = test_paths.data_dir / zip_name
    orbit_file = get_orbit_file_from_dir(zip_path, test_paths.orbit_dir)
    bursts = load_bursts(zip_path, orbit_file, 2, pol="vv")
    assert len(bursts) == 9

    # Compare with ESA burst IDs that are available in the annotation files for
    # IPF >= 3.40
    esa_burst_ids = _get_esa_burst_ids(zip_path)
    s1_burst_ids = [int(b.burst_id.split("_")[1]) for b in bursts]
    assert esa_burst_ids == s1_burst_ids

    _compare_bursts_geometry_to_esa(bursts, esa_burst_db)


def _compare_bursts_geometry_to_esa(bursts, esa_burst_db):
    # Check that all the geometries match roughly to the ESA burst database
    s1_geometries = [MultiPolygon(b.border) for b in bursts]
    esa_geometries = [esa_burst_db[b.burst_id]["geometry"] for b in bursts]

    for s1_geom, esa_geom in zip(s1_geometries, esa_geometries):
        # Check that the intersection over union is > 0.75
        assert iou(s1_geom, esa_geom) > 0.75


def test_anx_crossing(test_paths, esa_burst_db):
    """Check on a frame that crosses the equator mid frame."""
    zip_path = test_paths.data_dir / "S1A_IW_SLC__1SDV_20221024T184148_20221024T184218_045587_05735F_D6E2.zip"
    orbit_file = get_orbit_file_from_dir(zip_path, test_paths.orbit_dir)
    bursts = load_bursts(zip_path, orbit_file, 2, pol="vv")

    # get the start/end track from manifest.safe file
    tree = _get_safe_et(zip_path, "manifest.safe")
    start_track, end_track = get_start_end_track(tree)

    # t015_032217_iw2
    match = re.match(BURST_ID_PAT, bursts[0].burst_id)
    assert match
    # First burst is in track 15
    assert int(match.group("track")) == start_track

    # t016_032226_iw2
    # Last burst should be in track 16
    match = re.match(BURST_ID_PAT, bursts[-1].burst_id)
    assert match
    assert int(match.group("track")) == end_track

    _compare_bursts_geometry_to_esa(bursts, esa_burst_db)

def _get_safe_et(zip_path, file_pattern):
    with zipfile.ZipFile(zip_path) as zf:
        for fn in zf.namelist():
            if file_pattern in fn:
                return ET.fromstring(zf.read(fn))


def _get_esa_burst_ids(zip_path):
    tree = _get_safe_et(zip_path, "annotation/s1")
    return [int(b.text) for b in tree.findall("swathTiming/burstList/burst/burstId")]


def iou(geom1, geom2):
    """Calculate the intersection over union of two polygons."""
    return geom1.intersection(geom2).area / geom1.union(geom2).area
