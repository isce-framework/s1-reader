import os
from copy import deepcopy
from datetime import datetime, timedelta

from lxml import etree as ET

from s1reader.s1_orbit import get_orbit_file_from_dir

s3_url = "http://sentinel1-slc-seasia-pds.s3-website-ap-southeast-1.amazonaws.com/datasets/slc/v1.1"
data_info = "2021/08/15"
granule_id = "S1A_IW_SLC__1SDV_20210815T100025_20210815T100055_039238_04A1E8_D145"


def download_granule(in_url):
    zip_file = os.path.basename(in_url)
    if not os.path.isfile(zip_file):
        print(f'Start downloading of {in_url}')
        os.system(f'wget {in_url}')


def test_data_download():
    '''
    Test S1-A/B data download. Use seasia AWS s3 bucket
    to download data. Note, it contains only data acquired
    South East Asia, Taiwan, Korea, and Japan.
    '''
    # Get Full data URL
    data_url = f'{s3_url}/{data_info}/{granule_id}/{granule_id}.zip'
    download_granule(data_url)
    print('Done')


def shrink_orbit(safe_file, orbit_dir="orbits", overwrite=True, outname=None):
    """Download/shrink a precise orbit file to just the elements."""
    safe_file = safe_file.rstrip("/")
    str_sensing_start = os.path.basename(safe_file).split("_")[5]
    format_datetime = '%Y%m%dT%H%M%S'
    acq_dt = datetime.strptime(str_sensing_start, format_datetime)
    start_dt = acq_dt - timedelta(minutes=10)
    end_dt = acq_dt + timedelta(minutes=10)

    orbit_file = get_orbit_file_from_dir(safe_file, orbit_dir, auto_download=True)
    tree = ET.parse(orbit_file)
    osv_list_elem = tree.find("Data_Block/List_of_OSVs")

    start_idx, end_idx = None, None
    for idx, osv in enumerate(osv_list_elem):
        t_orbit = datetime.fromisoformat(osv[1].text[4:])

        if start_idx is None and t_orbit > start_dt:
            start_idx = idx

        if end_idx is None and t_orbit > end_dt:
            end_idx = idx
    
    # Now splice in the sublist
    cut_osv_list = deepcopy(osv_list_elem[start_idx:end_idx])
    ET.strip_elements(tree, "OSV")
    osv_list_elem.extend(cut_osv_list)
    osv_list_elem.attrib["count"] = str(end_idx - start_idx)

    if overwrite:
        outname = orbit_file

    with open(outname, "w") as f:
        f.write(ET.tostring(tree, pretty_print=True, encoding="unicode"))
    return outname

if __name__ == "__main__":
    test_data_download()
