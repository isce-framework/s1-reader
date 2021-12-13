import datetime
import os
import xml.etree.ElementTree as ET

import numpy as np

def get_swath_orbit_file(zip_path: str, orbit_dir: str):
    '''
    Get orbit state vector list for a given swath.

    Parameters:
    -----------
    zip_path : string
        Path to Sentinel1 SAFE zip file
    orbit_dir : string
        Path to directory containing orbit files.

    Returns:
    --------
    orbit_path : str
        Path the orbit file.
    '''
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist")

    if not os.path.isdir(orbit_dir):
        raise NotADirectoryError(f"{orbit_dir} not found")

    # date format used in file names
    fmt = "%Y%m%dT%H%M%S"

    # determine start and end times from file name
    file_name_tokens = os.path.basename(zip_path).split('_')
    platform_id = file_name_tokens[0]
    t_swath_start_stop = [datetime.datetime.strptime(t, fmt)
                          for t in file_name_tokens[5:7]]

    # find files with platform_id
    item_valid = lambda item, sat_id: os.path.isfile(item) and sat_id in item
    orbit_files = [item for item in os.listdir(orbit_dir)
                   if item_valid(f'{orbit_dir}/{item}', platform_id)]
    if not orbit_files:
        err_str = f"No orbit files found for {platform_id} in f{orbit_dir}"
        raise RuntimeError(err_str)

    # parse start and end time of files
    for orbit_file in orbit_files:
        # get file name and extract state vector start/end time strings
        t_orbit_start, t_orbit_end = os.path.basename(orbit_file).split('_')[-2:]

        # strip 'V' at start of start time string
        t_orbit_start = datetime.datetime.strptime(t_orbit_start[1:], fmt)

        # string '.EOF' from end of end time string
        t_orbit_end = datetime.datetime.strptime(t_orbit_end[:-4], fmt)

        # check if swath start end times fall within orbit start end times
        dt0 = datetime.timedelta(seconds=0)
        if all([t - t_orbit_start > dt0 for t in t_swath_start_stop]) and \
                all([t_orbit_end - t > dt0 for t in t_swath_start_stop]):
            break

    orbit_path = f'{orbit_dir}/{orbit_file}'

    return orbit_path
