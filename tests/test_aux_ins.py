"""Tests for the AUX_INS reader: name lookup, zip parsing, and burst wiring."""

import os
import pathlib
import zipfile

import lxml.etree as ET
import pytest

from s1reader import s1_annotation
from s1reader.s1_annotation import AuxIns
from s1reader.s1_reader import get_aux_ins_name, get_path_aux_ins

AUX_INS_DIR = pathlib.Path(s1_annotation.__file__).parent / "data" / "aux_ins"

#: The values ESA ships, constant across IW1-3 and all polarisations. S1A/S1B
#: are zero; S1C became nonzero in May 2025 and S1D in April 2026.
EXPECTED_BIAS = {
    "S1A_AUX_INS_V20250601T000000_G20251015T084917": 0.0,
    "S1B_AUX_INS_V20190514T090000_G20251015T071234": 0.0,
    "S1C_AUX_INS_V20250224T000000_G20260604T134021": -1.9685e-05,
    "S1D_AUX_INS_V20260319T000000_G20260603T113743": -6.6299e-05,
}


def manifest_tree(safe_zip):
    """Parse manifest.safe out of a SAFE .zip."""
    with zipfile.ZipFile(safe_zip) as z:
        name = next(f for f in z.namelist() if f.endswith("manifest.safe"))
        with z.open(name) as f:
            return ET.parse(f)


# --------------------------------------------------------------------------- #
# get_aux_ins_name
# --------------------------------------------------------------------------- #
def test_get_aux_ins_name_from_test_safe(test_paths):
    """The name comes back without its .SAFE suffix, ready for lookup.

    ESA nests the AUX_INS resource one level deeper than the top-level
    processing block, so this also guards the depth-agnostic search.
    """
    name = get_aux_ins_name(manifest_tree(test_paths.safe))
    assert name == "S1A_AUX_INS_V20171017T080000_G20180313T104658"


def test_get_aux_ins_name_returns_none_without_resource():
    """A manifest naming no AUX_INS yields None rather than raising."""
    xml = b"""<?xml version="1.0"?>
    <xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1"
               xmlns:safe="http://www.esa.int/safe/sentinel-1.0">
      <metadataSection><metadataObject><metadataWrap><xmlData>
        <safe:processing name="SLC Processing">
          <safe:resource name="/path/S1A_AUX_CAL_V1_G2.SAFE" role="AUX_CAL"/>
        </safe:processing>
      </xmlData></metadataWrap></metadataObject></metadataSection>
    </xfdu:XFDU>"""
    assert get_aux_ins_name(ET.fromstring(xml).getroottree()) is None


# --------------------------------------------------------------------------- #
# get_path_aux_ins
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(EXPECTED_BIAS))
def test_get_path_aux_ins_finds_vendored(name):
    path = get_path_aux_ins(str(AUX_INS_DIR), name)
    assert path is not None and os.path.exists(path)


def test_get_path_aux_ins_does_not_substitute_another_generation():
    """An unknown generation must return None, never a neighbouring file.

    ESA revises AUX_INS in place under the same validity start: S1C's
    V20250224 file has azimuthTimeBias 0 in generation G20250324T091341 and
    -19.685 us from G20250519T123409. Falling back to a different generation
    would return a bias the IPF never saw.
    """
    assert (
        get_path_aux_ins(
            str(AUX_INS_DIR), "S1C_AUX_INS_V20250224T000000_G20250324T091341"
        )
        is None
    )


def test_get_path_aux_ins_handles_missing_name():
    assert get_path_aux_ins(str(AUX_INS_DIR), None) is None


# --------------------------------------------------------------------------- #
# AuxIns.load_from_zip_file
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,bias", sorted(EXPECTED_BIAS.items()))
def test_load_bias_matches_esa_values(name, bias):
    aux = AuxIns.load_from_zip_file(str(AUX_INS_DIR / f"{name}.SAFE.zip"), "vv", "iw2")
    assert aux.azimuth_time_bias == pytest.approx(bias, abs=1e-12)


@pytest.mark.parametrize("swath", ["iw1", "iw2", "iw3"])
@pytest.mark.parametrize("pol", ["vv", "vh", "hh", "hv"])
def test_bias_is_constant_across_swath_and_pol(swath, pol):
    """A spacecraft timing bias cannot depend on which beam or channel is used.

    This is the property that makes a single bulk offset legitimate, so it is
    worth pinning rather than assuming. Checked here for the IW channels a CSLC
    run actually asks for; ``test_bias_is_single_valued_for_whole_instrument``
    pins it for the entire file.
    """
    name = "S1D_AUX_INS_V20260319T000000_G20260603T113743"
    aux = AuxIns.load_from_zip_file(str(AUX_INS_DIR / f"{name}.SAFE.zip"), pol, swath)
    assert aux.azimuth_time_bias == pytest.approx(-6.6299e-05, abs=1e-12)
    assert aux.swath == swath.upper()
    assert aux.polarisation == pol.upper()


@pytest.mark.parametrize("name,bias", sorted(EXPECTED_BIAS.items()))
def test_bias_is_single_valued_for_whole_instrument(name, bias):
    """Every swath/pol entry in the file carries the same azimuthTimeBias.

    ESA ships 88 entries (23 swaths x 4 pols) and they are identical, so the
    swath/pol filter cannot return a wrong value and a bulk offset applied to
    all three IW subswaths is exactly right. If ESA ever makes this
    beam-dependent this test fails, which is precisely when the bulk-offset
    assumption stops holding.
    """
    stem = f"{name}.SAFE"
    with zipfile.ZipFile(AUX_INS_DIR / f"{stem}.zip") as z:
        with z.open(f"{stem}/data/{name[:3].lower()}-aux-ins.xml") as f:
            root = ET.parse(f).getroot()
    entries = root.find("internalCalibrationParamsList").findall(
        "internalCalibrationParams"
    )
    values = sorted({float(e.findtext("azimuthTimeBias")) for e in entries})
    swaths = {e.findtext("swath") for e in entries}
    assert len(entries) == 88
    assert {"IW1", "IW2", "IW3"} <= swaths
    assert values == pytest.approx([bias], abs=1e-12)


def test_load_returns_independent_instances():
    """Two loads must not share state.

    ``AuxCal.load_from_zip_file`` assigns to ``cls`` and hands the class back,
    so loading a second channel silently rewrites the first. ``AuxIns`` returns
    instances instead, and this pins that difference.
    """
    s1c = AUX_INS_DIR / "S1C_AUX_INS_V20250224T000000_G20260604T134021.SAFE.zip"
    s1d = AUX_INS_DIR / "S1D_AUX_INS_V20260319T000000_G20260603T113743.SAFE.zip"
    a = AuxIns.load_from_zip_file(str(s1c), "vv", "iw1")
    b = AuxIns.load_from_zip_file(str(s1d), "vv", "iw1")
    assert a.azimuth_time_bias == pytest.approx(-1.9685e-05, abs=1e-12)
    assert b.azimuth_time_bias == pytest.approx(-6.6299e-05, abs=1e-12)
    assert a is not b


def test_swst_bias_is_read_too():
    name = "S1C_AUX_INS_V20250224T000000_G20260604T134021"
    aux = AuxIns.load_from_zip_file(str(AUX_INS_DIR / f"{name}.SAFE.zip"), "vv", "iw1")
    assert aux.swst_bias == pytest.approx(-5.934106e-09, rel=1e-6)


def test_load_rejects_non_zip(tmp_path):
    p = tmp_path / "S1C_AUX_INS_V1_G2.SAFE"
    p.mkdir()
    with pytest.raises(ValueError, match="Only AUX_INS files in .zip format"):
        AuxIns.load_from_zip_file(str(p), "vv", "iw1")


def test_load_rejects_missing_file(tmp_path):
    missing = str(tmp_path / "S1C_AUX_INS_V1_G2.SAFE.zip")
    with pytest.raises(ValueError, match="Cannot find AUX_INS .zip file"):
        AuxIns.load_from_zip_file(missing, "vv", "iw1")


def test_load_rejects_unknown_swath():
    name = "S1C_AUX_INS_V20250224T000000_G20260604T134021"
    with pytest.raises(ValueError, match="No AUX_INS entry for swath=iw9"):
        AuxIns.load_from_zip_file(str(AUX_INS_DIR / f"{name}.SAFE.zip"), "vv", "iw9")


def test_load_rejects_zip_without_aux_ins_xml(tmp_path):
    """A .zip that is not an AUX_INS product fails loudly, not with a KeyError."""
    p = tmp_path / "S1C_AUX_INS_V1_G2.SAFE.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("S1C_AUX_INS_V1_G2.SAFE/data/readme.txt", "not the xml")
    with pytest.raises(ValueError, match="Cannot find .*-aux-ins.xml"):
        AuxIns.load_from_zip_file(str(p), "vv", "iw1")


# --------------------------------------------------------------------------- #
# burst wiring
# --------------------------------------------------------------------------- #
def test_burst_does_not_read_the_bias_by_default(bursts):
    assert bursts[0].azimuth_time_bias is None


def test_burst_records_aux_ins_name(bursts):
    """The AUX_INS name from manifest.safe rides on every burst, whether or not
    the bias is read: a downstream timing correction needs the generation."""
    assert bursts[0].aux_ins_name == "S1A_AUX_INS_V20171017T080000_G20180313T104658"


def test_burst_bias_is_unknown_for_a_generation_not_shipped(test_paths):
    """Not 0.0: real S1C/S1D products name generations with a nonzero bias that
    are not shipped, e.g. S1D_AUX_INS_V20260319T000000_G20260415T150144."""
    from s1reader import s1_reader

    orbit_path = f"{test_paths.orbit_dir}/{test_paths.orbit_file}"
    with pytest.warns(UserWarning, match="azimuth_time_bias is unknown"):
        bursts = s1_reader.load_bursts(
            test_paths.safe, orbit_path, 3, "vv", flag_read_aux_ins=True
        )
    assert bursts[0].azimuth_time_bias is None
    assert bursts[0].aux_ins_name == "S1A_AUX_INS_V20171017T080000_G20180313T104658"


def test_burst_reads_the_bias_of_a_shipped_generation(test_paths, monkeypatch):
    from s1reader import s1_reader

    name = "S1C_AUX_INS_V20250224T000000_G20260604T134021"
    monkeypatch.setattr(s1_reader, "get_aux_ins_name", lambda tree: name)
    orbit_path = f"{test_paths.orbit_dir}/{test_paths.orbit_file}"
    bursts = s1_reader.load_bursts(
        test_paths.safe, orbit_path, 3, "vv", flag_read_aux_ins=True
    )
    assert bursts[0].azimuth_time_bias == pytest.approx(-1.9685e-05, abs=1e-12)
    assert bursts[0].aux_ins_name == name


def test_burst_default_keeps_positional_construction_working(bursts):
    """The AUX_INS fields are last and defaulted, so the frozen dataclass can
    still be built positionally by callers that predate them."""
    from s1reader.s1_burst_slc import Sentinel1BurstSlc
    from dataclasses import fields

    flds = fields(Sentinel1BurstSlc)
    assert [f.name for f in flds[-2:]] == ["azimuth_time_bias", "aux_ins_name"]
    assert flds[-2].default is None
    assert flds[-1].default is None
