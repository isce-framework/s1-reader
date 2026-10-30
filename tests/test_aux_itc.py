"""Tests for the AUX_ITC reader."""

import pathlib
import zipfile

import lxml.etree as ET
import pytest

from s1reader.s1_annotation import AuxItc

AUX_ITC_DIR = pathlib.Path(__file__).parent / "data" / "aux_itc"

# (azimuthCalibration, rangeCalibration) as ESA ships them. S1C/S1D are zero,
# which looks like a placeholder rather than a measurement.
EXPECTED = {
    "S1A_AUX_ITC_V20160627T000000_G20230406T084701": (6.3522e-06, 7.4103e-10),
    "S1B_AUX_ITC_V20160422T000000_G20230406T084600": (-3.5523e-05, -1.2855e-10),
    "S1C_AUX_ITC_V20250519T094846_G20250521T125837": (0.0, 0.0),
    "S1D_AUX_ITC_V20260319T000000_G20260318T105930": (0.0, 0.0),
}

S1A = str(AUX_ITC_DIR / "S1A_AUX_ITC_V20160627T000000_G20230406T084701.SAFE.zip")


def write_itc_zip(path, reference, offsets):
    """Write a minimal AUX_ITC .SAFE.zip.

    `reference` is (azimuthCalibration, rangeCalibration); `offsets` is a
    list of (swath, pol, azimuthOffset, rangeOffset).
    """
    stem = path.name.replace(".zip", "")
    root = ET.Element("auxiliarySetap")
    ref = ET.SubElement(root, "instrumentTimingCalibrationReference")
    if reference is not None:
        ET.SubElement(ref, "rangeCalibration").text = str(reference[1])
        ET.SubElement(ref, "azimuthCalibration").text = str(reference[0])
    else:
        root.remove(ref)
    lst = ET.SubElement(root, "instrumentTimingCalibrationOffsetList")
    for swath, pol, az, rg in offsets:
        entry = ET.SubElement(lst, "instrumentTimingCalibrationOffset")
        ET.SubElement(entry, "swath").text = swath
        ET.SubElement(entry, "polarisation").text = pol
        ET.SubElement(entry, "rangeOffset").text = str(rg)
        ET.SubElement(entry, "azimuthOffset").text = str(az)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{stem}/data/{stem[:3].lower()}-aux-itc.xml", ET.tostring(root))
    return str(path)


@pytest.mark.parametrize("name,expected", sorted(EXPECTED.items()))
def test_load_matches_esa_values(name, expected):
    aux = AuxItc.load_from_zip_file(str(AUX_ITC_DIR / f"{name}.SAFE.zip"), "vv", "iw2")
    assert aux.azimuth_calibration == pytest.approx(expected[0], abs=1e-15)
    assert aux.range_calibration == pytest.approx(expected[1], abs=1e-18)
    assert aux.swath == "IW2"
    assert aux.polarisation == "VV"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_per_swath_offsets_are_zero_in_every_shipped_file(name):
    """All 56 swath/pol offsets are zero in the generations we use.

    The relative timing table treats the reference as an instrument-wide value.
    If ESA ever ships a beam-dependent offset this fails, which is exactly when
    that assumption stops holding.
    """
    stem = f"{name}.SAFE"
    with zipfile.ZipFile(AUX_ITC_DIR / f"{stem}.zip") as z:
        with z.open(f"{stem}/data/{name[:3].lower()}-aux-itc.xml") as f:
            root = ET.parse(f).getroot()
    entries = root.find("instrumentTimingCalibrationOffsetList").findall(
        "instrumentTimingCalibrationOffset"
    )
    assert len(entries) == 56
    assert {"IW1", "IW2", "IW3"} <= {e.findtext("swath") for e in entries}
    assert {float(e.findtext("azimuthOffset")) for e in entries} == {0.0}
    assert {float(e.findtext("rangeOffset")) for e in entries} == {0.0}


def test_totals_add_reference_and_offset(tmp_path):
    path = write_itc_zip(
        tmp_path / "S1C_AUX_ITC_V1_G2.SAFE.zip",
        reference=(1.0e-6, 2.0e-10),
        offsets=[("IW1", "VV", 3.0e-6, 4.0e-10), ("IW2", "VV", 0.0, 0.0)],
    )
    iw1 = AuxItc.load_from_zip_file(path, "vv", "iw1")
    iw2 = AuxItc.load_from_zip_file(path, "vv", "iw2")
    assert iw1.azimuth_offset == pytest.approx(3.0e-6)
    assert iw1.range_offset == pytest.approx(4.0e-10)
    assert iw1.azimuth_time_total == pytest.approx(4.0e-6)
    assert iw1.range_time_total == pytest.approx(6.0e-10)
    assert iw2.azimuth_time_total == pytest.approx(1.0e-6)
    assert iw2.range_time_total == pytest.approx(2.0e-10)


def test_load_returns_independent_instances():
    """Unlike AuxCal, which writes to the class, two loads do not share state."""
    s1b = str(AUX_ITC_DIR / "S1B_AUX_ITC_V20160422T000000_G20230406T084600.SAFE.zip")
    a = AuxItc.load_from_zip_file(S1A, "vv", "iw1")
    b = AuxItc.load_from_zip_file(s1b, "vv", "iw1")
    assert a.azimuth_calibration == pytest.approx(6.3522e-06)
    assert b.azimuth_calibration == pytest.approx(-3.5523e-05)


def test_load_rejects_non_zip(tmp_path):
    p = tmp_path / "S1C_AUX_ITC_V1_G2.SAFE"
    p.mkdir()
    with pytest.raises(ValueError, match="Only AUX_ITC files in .zip format"):
        AuxItc.load_from_zip_file(str(p), "vv", "iw1")


def test_load_rejects_missing_file(tmp_path):
    missing = str(tmp_path / "S1C_AUX_ITC_V1_G2.SAFE.zip")
    with pytest.raises(ValueError, match="Cannot find AUX_ITC .zip file"):
        AuxItc.load_from_zip_file(missing, "vv", "iw1")


def test_load_rejects_renamed_download(tmp_path):
    """A browser copy named `... .SAFE(1).zip` no longer matches its .SAFE
    directory; it must fail loudly rather than be half-read."""
    src = pathlib.Path(S1A)
    renamed = tmp_path / src.name.replace(".SAFE.zip", ".SAFE(1).zip")
    renamed.write_bytes(src.read_bytes())
    with pytest.raises(ValueError, match="Cannot find .*-aux-itc.xml"):
        AuxItc.load_from_zip_file(str(renamed), "vv", "iw1")


def test_load_rejects_unknown_swath():
    with pytest.raises(ValueError, match="No AUX_ITC entry for swath=iw9"):
        AuxItc.load_from_zip_file(S1A, "vv", "iw9")


def test_load_rejects_missing_reference(tmp_path):
    path = write_itc_zip(
        tmp_path / "S1C_AUX_ITC_V1_G2.SAFE.zip",
        reference=None,
        offsets=[("IW1", "VV", 0.0, 0.0)],
    )
    with pytest.raises(ValueError, match="No instrumentTimingCalibrationReference"):
        AuxItc.load_from_zip_file(path, "vv", "iw1")
