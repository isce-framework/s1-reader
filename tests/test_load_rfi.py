from pathlib import Path

from s1reader import load_bursts

data_path = Path(__file__).parent.resolve() / "data"


def test_load_rfi():
    zipname = "S1A_IW_SLC__1SDV_20230108T135249_20230108T135316_046693_0598D3_BA76.zip"
    zip_path = data_path / zipname

    bursts = load_bursts(zip_path, None, 2, pol="vv", flag_apply_eap=False)
    for b in bursts:
        assert b.burst_rfi_info is not None


def test_load_missing_rfi():
    # Check it's skipped for zip files without RFI
    zipname = "S1A_IW_SLC__1SDV_20200511T135117_20200511T135144_032518_03C421_7768.zip"
    zip_path = data_path / zipname
    for b in load_bursts(zip_path, None, 2, pol="vv", flag_apply_eap=False):
        assert b.burst_rfi_info is None
