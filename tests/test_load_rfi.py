import json
from pathlib import Path


from s1reader import load_bursts

data_path = Path(__file__).parent.resolve() / "data"


def test_load_rfi():
    '''
    Check if the RFI information was loaded properly, and
    the information in the object is as expected.
    '''
    zipname = "S1A_IW_SLC__1SDV_20230108T135249_20230108T135316_046693_0598D3_BA76.zip"
    rfi_info_pickle_filename = "rfi_info_list.json"
    zip_path = data_path / zipname
    rfi_info_path = data_path / rfi_info_pickle_filename

    # load the expected RFI info
    with open(rfi_info_path, 'r') as file_in:
        expected_rfi_info_list = json.load(file_in)

    bursts = load_bursts(zip_path, None, 2, pol="vv", flag_apply_eap=False)
    for i_burst, b in enumerate(bursts):
        # Check if the burst RFI info has loaded
        assert b.burst_rfi_info is not None

        burst_rfi_info_dict = b.burst_rfi_info.__dict__
        burst_rfi_info_dict['rfi_burst_report']['azimuthTime'] = \
            burst_rfi_info_dict['rfi_burst_report']['azimuthTime'].isoformat()

        # Check if the information in the burst RFI into is correct
        assert burst_rfi_info_dict == expected_rfi_info_list[i_burst]


def test_load_missing_rfi():
    # Check it's skipped for zip files without RFI
    zipname = "S1A_IW_SLC__1SDV_20200511T135117_20200511T135144_032518_03C421_7768.zip"
    zip_path = data_path / zipname
    for b in load_bursts(zip_path, None, 2, pol="vv", flag_apply_eap=False):
        assert b.burst_rfi_info is None
