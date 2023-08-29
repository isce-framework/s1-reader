from __future__ import annotations

import cgi
import datetime
import glob
import os
import requests
import warnings

from xml.etree import ElementTree

# date format used in file names
FMT = "%Y%m%dT%H%M%S"

# Orbital period of Sentinel-1 in seconds:
# 12 days * 86400.0 seconds/day, divided into 175 orbits
T_ORBIT = (12 * 86400.0) / 175.0

# Temporal margin to apply to the start time of a frame
#  to make sure that the ascending node crossing is
#    included when choosing the orbit file
margin_start_time = datetime.timedelta(seconds=T_ORBIT + 60.0)

# Scihub guest credential
scihub_user = 'gnssguest'
scihub_password = 'gnssguest'


def download_orbit(safe_file: str, orbit_dir: str):
    '''
    Download orbits for S1-A/B SAFE "safe_file"

    Parameters
    ----------
    safe_file: str
        File path to SAFE file for which download the orbits
    orbit_dir: str
        File path to directory where to store downloaded orbits

    Returns:
    --------
    orbit_file : str
        Path to the orbit file.
    '''

    # Create output directory & check internet connection
    os.makedirs(orbit_dir, exist_ok=True)
    check_internet_connection()

    # Parse info from SAFE file name
    mission_id, _, start_time, end_time, _ = parse_safe_filename(safe_file)

    # Apply margin to the start time
    start_time = start_time - margin_start_time

    # Find precise orbit first
    orbit_dict = get_orbit_dict(mission_id, start_time,
                                end_time, 'AUX_POEORB')
    # If orbit dict is empty, find restituted orbits
    if orbit_dict is None:
        orbit_dict = get_orbit_dict(mission_id, start_time,
                                    end_time, 'AUX_RESORB')
    # Download orbit file
    orbit_file = os.path.join(orbit_dir, f"{orbit_dict['orbit_name']}.EOF")
    if not os.path.exists(orbit_file):
        download_orbit_file(orbit_dir, orbit_dict['orbit_url'])

    return orbit_file


def check_internet_connection():
    '''
    Check connection availability
    '''
    url = "http://google.com"
    try:
        requests.get(url, timeout=10)
    except (requests.ConnectionError, requests.Timeout) as exception:
        raise ConnectionError(f'Unable to reach {url}: {exception}')


def parse_safe_filename(safe_filename):
    '''
    Extract info from S1-A/B SAFE filename
    SAFE filename structure: S1A_IW_SLC__1SDV_20150224T114043_20150224T114111_004764_005E86_AD02.SAFE
    Parameters
    -----------
    safe_filename: string
       Path to S1-A/B SAFE file

    Returns
    -------
    List of [mission_id, mode_id, start_datetime,
                end_datetime, abs_orbit_num]
       mission_id: sensor identifier (S1A or S1B)
       mode_id: mode/beam (e.g. IW)
       start_datetime: acquisition start datetime
       stop_datetime: acquisition stop datetime
       abs_orbit_num: absolute orbit number

    Examples
    ---------
    parse_safe_filename('S1A_IW_SLC__1SDV_20150224T114043_20150224T114111_004764_005E86_AD02.SAFE')
    returns
    ['S1A', 'IW', datetime.datetime(2015, 2, 24, 11, 40, 43),\
    datetime.datetime(2015, 2, 24, 11, 41, 11), 4764]
    '''

    safe_name = os.path.basename(safe_filename)
    mission_id = safe_name[:3]
    sensor_mode = safe_name[4:6]
    start_datetime = datetime.datetime.strptime(safe_name[17:32],
                                                FMT)
    end_datetime = datetime.datetime.strptime(safe_name[33:48],
                                              FMT)
    abs_orb_num = int(safe_name[49:55])

    return [mission_id, sensor_mode, start_datetime, end_datetime, abs_orb_num]


def get_file_name_tokens(zip_path: str) -> [str, list[datetime.datetime]]:
    '''Extract swath platform ID and start/stop times from SAFE zip file path.

    Parameters
    ----------
    zip_path: list[str]
        List containing orbit path strings.
        Orbit files required to adhere to naming convention found here:
        https://sentinels.copernicus.eu/documents/247904/351187/Copernicus_Sentinels_POD_Service_File_Format_Specification

    Returns
    -------
    mission_id: ('S1A', 'S1B')
    orbit_path : str
        Path the orbit file.
    t_swath_start_stop: list[datetime.datetime]
        Swath start/stop times
    '''
    mission_id, _, start_time, end_time, _ = parse_safe_filename(zip_path)
    return mission_id, [start_time, end_time]


def get_orbit_dict(mission_id, start_time, end_time, orbit_type):
    '''
    Query Copernicus GNSS API to find latest orbit file
    Parameters
    ----------
    mission_id: str
        Sentinel satellite identifier ('S1A' or 'S1B')
    start_time: datetime object
        Sentinel start acquisition time
    end_time: datetime object
        Sentinel end acquisition time
    orbit_type: str
        Type of orbit to download (AUX_POEORB: precise, AUX_RESORB: restituted)

    Returns
    -------
    orbit_dict: dict
        Python dictionary with [orbit_name, orbit_type, download_url]
    '''
    # Required for orbit download
    scihub_url = 'https://scihub.copernicus.eu/gnss/odata/v1/Products'
    # Namespaces of the XML file returned by the S1 query. Will they change it?
    m_url = '{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}'
    d_url = '{http://schemas.microsoft.com/ado/2007/08/dataservices}'

    # Check if correct orbit_type
    if orbit_type not in ['AUX_POEORB', 'AUX_RESORB']:
        err_msg = f'{orbit_type} not a valid orbit type'
        raise ValueError(err_msg)

    # Add a 30 min margin to start_time and end_time
    pad_30_min = datetime.timedelta(hours=0.5)
    pad_start_time = start_time - pad_30_min
    pad_end_time = end_time + pad_30_min
    new_start_time = pad_start_time.strftime('%Y-%m-%dT%H:%M:%S')
    new_end_time = pad_end_time.strftime('%Y-%m-%dT%H:%M:%S')
    query_string = f"startswith(Name,'{mission_id}') and substringof('{orbit_type}',Name) " \
                   f"and ContentDate/Start lt datetime'{new_start_time}' and ContentDate/End gt datetime'{new_end_time}'"
    query_params = {'$top': 1, '$orderby': 'ContentDate/Start asc',
                    '$filter': query_string}
    query_response = requests.get(url=scihub_url, params=query_params,
                                  auth=(scihub_user, scihub_password))
    # Parse XML tree from query response
    xml_tree = ElementTree.fromstring(query_response.content)
    # Extract w3.org URL
    w3_url = xml_tree.tag.split('feed')[0]

    # Extract orbit's name, id, url
    orbit_id = xml_tree.findtext(
        f'.//{w3_url}entry/{m_url}properties/{d_url}Id')
    orbit_url = f"{scihub_url}('{orbit_id}')/$value"
    orbit_name = xml_tree.findtext(f'./{w3_url}entry/{w3_url}title')

    if orbit_id is not None:
        orbit_dict = {'orbit_name': orbit_name, 'orbit_type': orbit_type,
                      'orbit_url': orbit_url}
    else:
        orbit_dict = None
    return orbit_dict


def download_orbit_file(output_folder, orbit_url):
    '''
    Download S1-A/B orbits
    Parameters
    ----------
    output_folder: str
        Path to directory where to store orbits
    orbit_url: str
        Remote url of orbit file to download
    '''
    print('downloading URL:', orbit_url)
    response = requests.get(url=orbit_url, auth=(scihub_user, scihub_password))

    # Get header and find filename
    header = response.headers['content-disposition']
    header_params = cgi.parse_header(header)[1]
    # construct orbit filename
    orbit_file = os.path.join(output_folder, header_params['filename'])

    # Save orbits
    with open(orbit_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()
    return orbit_file


def get_orbit_file_from_dir(zip_path: str, orbit_dir: str, auto_download: bool = False) -> str:
    '''Get orbit state vector list for a given swath.

    Parameters:
    -----------
    zip_path : string
        Path to Sentinel1 SAFE zip file. Base names required to adhere to the
        format described here:
        https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    orbit_dir : string
        Path to directory containing orbit files. Orbit files required to adhere
        to naming convention found here:
        https://s1qc.asf.alaska.edu/aux_poeorb/
    auto_download : bool
        Automatically download the orbit file if not exist in the orbit_dir.

    Returns:
    --------
    orbit_file : str
        Path to the orbit file.
    '''

    # check the existence of input file path and directory
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist")

    if not os.path.isdir(orbit_dir):
        if not auto_download:
            raise NotADirectoryError(f"{orbit_dir} not found")
        else:
            print(f"{orbit_dir} not found, creating directory.")
            os.makedirs(orbit_dir, exist_ok=True)

    # search for orbit file
    orbit_file_list = glob.glob(os.path.join(orbit_dir, 'S1*.EOF'))

    orbit_file = get_orbit_file_from_list(zip_path, orbit_file_list)

    if orbit_file:
        return orbit_file

    if not auto_download:
        msg = (f'No orbit file was found for {os.path.basename(zip_path)} '
                f'from the directory provided: {orbit_dir}')
        warnings.warn(msg)
        return

    # Attempt auto download
    orbit_file = download_orbit(zip_path, orbit_dir)
    return orbit_file


def get_orbit_file_from_list(zip_path: str, orbit_file_list: list) -> str:
    '''Get orbit file for a given S-1 swath from a list of files

    Parameters:
    -----------
    zip_path : string
        Path to Sentinel1 SAFE zip file. Base names required to adhere to the
        format described here:
        https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    orbit_file_list : list
        List of the orbit files that exists in the system.

    Returns:
    --------
    orbit_file : str
        Path to the orbit file, or an empty string if no orbit file was found.
    '''
    # check the existence of input file path and directory
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist")

    # extract platform id, start and end times from swath file name
    mission_id, t_swath_start_stop = get_file_name_tokens(zip_path)

    # Apply temporal margin to the start time of the frame
    # 1st element: start time, 2nd element: end time
    t_swath_start_stop[0] = t_swath_start_stop[0] - margin_start_time

    # initiate output
    orbit_file_final = ''

    # search for orbit file
    for orbit_file in orbit_file_list:
        # check if file validity
        if not os.path.isfile(orbit_file):
            continue
        if mission_id not in os.path.basename(orbit_file):
            continue

        # get file name and extract state vector start/end time strings
        t_orbit_start, t_orbit_end = os.path.basename(orbit_file).split('_')[-2:]

        # strip 'V' at start of start time string
        t_orbit_start = datetime.datetime.strptime(t_orbit_start[1:], FMT)

        # string '.EOF' from end of end time string
        t_orbit_stop = datetime.datetime.strptime(t_orbit_end[:-4], FMT)

        # check if:
        # 1. swath start and stop time > orbit file start time
        # 2. swath start and stop time < orbit file stop time
        if all([t_orbit_start < t < t_orbit_stop for t in t_swath_start_stop]):
            orbit_file_final = orbit_file
            break

    if not orbit_file_final:
        msg = 'No orbit file was found in the file list provided.'
        warnings.warn(msg)

    return orbit_file_final


def combine_xml_orbit_elements(
    file1: str, file2: str, write_output: bool = True
) -> ElementTree.ElementTree:
    """Combine the orbit elements from two XML files.

    Parameters
    ----------
    file1 : str
        The path to the first .EOF file.
    file2 : str
        The path to the second .EOF file.
    write_output : bool, default = True
        Create a new .EOF file with the combined results.
        Output is named with the start_datetime and stop_datetime changed, with
        the same base as `file1`.

    Returns
    -------
    ET.ElementTree
        Combined XML structure.
    """

    def get_dt(root: ElementTree.ElementTree, tag_name: str) -> datetime:
        time_str = root.find(f".//{tag_name}").text.split("=")[-1]
        return datetime.fromisoformat(time_str)

    # Parse the XML files
    tree1 = ElementTree.parse(file1)
    tree2 = ElementTree.parse(file2)

    root1 = tree1.getroot()
    root2 = tree2.getroot()

    # Extract the Validity_Start and Validity_Stop timestamps from both files
    start_time1 = get_dt(root1, "Validity_Start")
    stop_time1 = get_dt(root1, "Validity_Start")
    start_time2 = get_dt(root2, "Validity_Start")
    stop_time2 = get_dt(root2, "Validity_Start")

    # Determine the new Validity_Start and Validity_Stop values
    new_start_dt = min(start_time1, start_time2)
    new_stop_dt = max(stop_time1, stop_time2)

    # Update the Validity_Start and Validity_Stop timestamps in the first XML
    root1.find(".//Validity_Start").text = "UTC=" + new_start_dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    root1.find(".//Validity_Stop").text = "UTC=" + new_stop_dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    # Combine the <OSV> elements
    list_of_osvs1 = root1.find(".//List_of_OSVs")
    list_of_osvs2 = root2.find(".//List_of_OSVs")

    for osv in list_of_osvs2.findall("OSV"):
        list_of_osvs1.append(osv)

    # Adjust the count attribute in <List_of_OSVs>
    new_count = len(list_of_osvs1.findall("OSV"))
    list_of_osvs1.set("count", str(new_count))

    if write_output:
        outfile = _generate_filename(file1, new_start_dt, new_stop_dt)
        tree1.write(outfile, encoding="UTF-8", xml_declaration=True)
    return tree1


def _generate_filename(file_base: str, new_start: datetime, new_stop: datetime) -> str:
    """Generate a new filename based on the two provided filenames.

    Parameters
    ----------
    file_base : str
        The name of one of the concatenated files
    new_start : datetime
        The new first datetime of the updated orbital elements
    new_stop : datetime
        The new final datetime of the updated orbital elements

    Returns
    -------
    str
        Generated filename.
    """
    product_name = Path(file_base).name
    # >>> 'S1A_OPER_AUX_PREORB_OPOD_20200325T131800_V20200325T121452_20200325T184952'.index('V')
    # 41
    fmt = "%Y%m%dT%H%M%S"
    new_start_stop_str = new_start.strftime(fmt) + "_" + new_stop.strftime(fmt)
    new_product_name = product_name[:42] + new_start_stop_str
    return str(file_base).replace(product_name, new_product_name)
