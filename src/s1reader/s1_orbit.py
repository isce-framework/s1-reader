import datetime
import os
import xml.etree.ElementTree as ET

import isce3

# date format used in file names
FMT = "%Y%m%dT%H%M%S"

def get_file_name_tokens(zip_path: str) -> [str, list[datetime.datetime]]:
    '''Extract swath platform ID and start/stop times from SAFE zip file path.

    Parameters:
    -----------
    zip_path: list[str]
        List containing orbit path strings. Orbit files required to adhere to
        naming convention found here:
        https://s1qc.asf.alaska.edu/aux_poeorb/

    Returns:
    --------
    platform_id: ('S1A', 'S1B')
    orbit_path : str
        Path the orbit file.
    t_swath_start_stop: list[datetime.datetime]
        Swath start/stop times
    '''
    file_name_tokens = os.path.basename(zip_path).split('_')

    # extract and check platform ID
    platform_id = file_name_tokens[0]
    if platform_id not in ['S1A', 'S1B']:
        err_str = f'{platform_id} not S1A nor S1B'
        ValueError(err_str)

    # extract start/stop time as a list[datetime.datetime]: [t_start, t_stop]
    t_swath_start_stop = [datetime.datetime.strptime(t, FMT)
                          for t in file_name_tokens[5:7]]

    return platform_id, t_swath_start_stop


# lambda to check if file exisits if desired sat_id in basename
item_valid = lambda item, sat_id: os.path.isfile(item) and sat_id in os.path.basename(item)


def get_orbit_file_from_list(zip_path: str, orbit_file_list: list[str]) -> str:
    '''Get orbit state vector list for a given swath.

    Parameters:
    -----------
    zip_path : string
        Path to Sentinel1 SAFE zip file. File names required to adhere to the
        format described here:
        https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    orbit_file_list: list[str]
        List containing orbit files paths. Orbit files required to adhere to
        naming convention found here:
        https://s1qc.asf.alaska.edu/aux_poeorb/

    Returns:
    --------
    orbit_path : str
        Path the orbit file.
    '''
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist")

    if not orbit_file_list:
        raise ValueError("no orbit paths provided")

    # extract platform id, start and end times from swath file name
    platform_id, t_swath_start_stop = get_file_name_tokens(zip_path)

    for orbit_file in orbit_file_list:
        # check if file validity
        if not  item_valid(orbit_file, platform_id):
            continue

        # get file name and extract state vector start/end time strings
        t_orbit_start, t_orbit_end = os.path.basename(orbit_file).split('_')[-2:]

        # strip 'V' at start of start time string
        t_orbit_start = datetime.datetime.strptime(t_orbit_start[1:], FMT)

        # string '.EOF' from end of end time string
        t_orbit_end = datetime.datetime.strptime(t_orbit_end[:-4], FMT)

        # check if:
        # 1. swath start and stop time > orbit file start time
        # 2. swath start and stop time < orbit file stop time
        if all([t > t_orbit_start for t in t_swath_start_stop]) and \
                all([t < t_orbit_end for t in t_swath_start_stop]):
            return orbit_file

    return ''

def get_orbit_file_from_dir(path: str, orbit_dir: str) -> str:
    '''Get orbit state vector list for a given swath.

    Parameters:
    -----------
    path : string
        Path to Sentinel1 SAFE zip file. Base names required to adhere to the
        format described here:
        https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    orbit_dir : string
        Path to directory containing orbit files. Orbit files required to adhere
        to naming convention found here:
        https://s1qc.asf.alaska.edu/aux_poeorb/

    Returns:
    --------
    orbit_path : str
        Path the orbit file.
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")

    if not os.path.isdir(orbit_dir):
        raise NotADirectoryError(f"{orbit_dir} not found")

    orbit_path = get_orbit_file_from_list(
        path, [f'{orbit_dir}/{item}' for item in os.listdir(orbit_dir)])

    return orbit_path

def burst_orbit_from_file(sensing_start, sensing_stop, osv_list: ET.Element):
    '''Init and return ISCE3 orbit from element in orbit XML file.

    Parameters:
    -----------
    sensing_start : datetime.datetime
        Sensing start of burst; taken from azimuth time
    sensing_stop : datetime.datetime
        Sensing stop of burst
    osv_list : xml.etree.ElementTree.Element
        ElementTree containing orbit state vectors

    Returns:
    --------
    _ : datetime
        Sensing mid as datetime object.
    '''
    fmt = "UTC=%Y-%m-%dT%H:%M:%S.%f"
    orbit_sv = []
    # add start & end padding to ensure sufficient number of orbit points
    pad = datetime.timedelta(seconds=60)
    for osv in osv_list:
        t_orbit = datetime.datetime.strptime(osv[1].text, fmt)

        if t_orbit > sensing_stop + pad:
            break

        if t_orbit > sensing_start - pad:
            pos = [float(osv[i].text) for i in range(4,7)]
            vel = [float(osv[i].text) for i in range(7,10)]
            orbit_sv.append(isce3.core.StateVector(isce3.core.DateTime(t_orbit),
                                                   pos, vel))

    # use list of stateVectors to init and return isce3.core.Orbit
    time_delta = datetime.timedelta(days=2)
    ref_epoch = isce3.core.DateTime(sensing_start - time_delta)

    return isce3.core.Orbit(orbit_sv, ref_epoch)
