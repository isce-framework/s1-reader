import cgi
import datetime
import glob
import os
import requests
import warnings

from xml.etree import ElementTree

# date format used in file names
FMT = "%Y%m%dT%H%M%S"

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
    sensor_id, _, start_time, end_time, _ = parse_safe_filename(safe_file)

    # Find precise orbit first
    orbit_dict = get_orbit_dict(sensor_id, start_time,
                                end_time, 'AUX_POEORB')
    # If orbit dict is empty, find restituted orbits
    if orbit_dict is None:
        orbit_dict = get_orbit_dict(sensor_id, start_time,
                                    end_time, 'AUX_RESORB')
    # Download orbit file
    orbit_file = os.path.join(orbit_dir, orbit_dict["orbit_name"] + '.EOF')
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
    List of [sensor_id, mode_id, start_datetime,
                end_datetime, abs_orbit_num]
       sensor_id: sensor identifier (S1A or S1B)
       mode_id: mode/beam (e.g. IW)
       start_datetime: acquisition start datetime
       stop_datetime: acquisition stop datetime
       abs_orbit_num: absolute orbit number

    Examples
    ---------
    parse_safe_filename('S1A_IW_SLC__1SDV_20150224T114043_20150224T114111_004764_005E86_AD02.SAFE')
    returns
    ['A', 'IW', datetime.datetime(2015, 2, 24, 11, 40, 43),\
    datetime.datetime(2015, 2, 24, 11, 41, 11), 4764]
    '''

    safe_name = os.path.basename(safe_filename)
    sensor_id = safe_name[2]
    sensor_mode = safe_name[4:6]
    start_datetime = datetime.datetime.strptime(safe_name[17:32],
                                                FMT)
    end_datetime = datetime.datetime.strptime(safe_name[33:48],
                                              FMT)
    abs_orb_num = int(safe_name[49:55])

    return [sensor_id, sensor_mode, start_datetime, end_datetime, abs_orb_num]


def get_orbit_dict(sensor_id, start_time, end_time, orbit_type):
    '''
    Query Copernicus GNSS API to find latest orbit file
    Parameters
    ----------
    sensor_id: str
        Sentinel satellite identifier ('A' or 'B')
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
    query_string = f"startswith(Name,'S1{sensor_id}') and substringof('{orbit_type}',Name) " \
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


def get_file_name_tokens(zip_path: str) -> [str, list[datetime.datetime]]:
    '''Extract swath platform ID and start/stop times from SAFE zip file path.

    Parameters
    ----------
    zip_path: list[str]
        List containing orbit path strings. Orbit files required to adhere to
        naming convention found here:
        https://s1qc.asf.alaska.edu/aux_poeorb/

    Returns
    -------
    platform_id: ('S1A', 'S1B')
    orbit_path : str
        Path the orbit file.
    t_swath_start_stop: list[datetime.datetime]
        Swath start/stop times
    '''
    platform_id, _, start_time, end_time, _ = parse_safe_filename(zip_path)
    return platform_id,[start_time, end_time]


# lambda to check if file exists if desired sat_id in basename
item_valid = lambda item, sat_id: os.path.isfile(item) and sat_id in os.path.basename(item)



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
        raise NotADirectoryError(f"{orbit_dir} not found")

    # extract platform id, start and end times from swath file name
    platform_id, t_swath_start_stop = get_file_name_tokens(zip_path)

    # initiate output
    orbit_file = ''

    # search for orbit file
    orbit_file_list = glob.glob(os.path.join(orbit_dir, 'S1*.EOF'))

    orbit_file = get_orbit_file_from_list(zip_path, orbit_file_list)

    if not orbit_file:
        orbit_dir = os.path.dirname(orbit_dir)
        if not orbit_dir:
            orbit_dir = os.getcwd()

        if auto_download:
            orbit_file = download_orbit(zip_path, orbit_dir)
        else:
            msg = (f'No orbit file found for {os.path.basename(zip_path)} '
                    'from the orbit_source provided.')
            warnings.warn(msg)

    return orbit_file



def get_orbit_file_from_list(zip_path: str, orbit_file_list: list) -> str:
    '''Get orbit state vector list for a given swath.

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
        Path to the orbit file.
    '''

    # check the existence of input file path and directory
    #if not os.path.exists(zip_path):
    #    raise FileNotFoundError(f"{zip_path} does not exist")

    # extract platform id, start and end times from swath file name
    platform_id, t_swath_start_stop = get_file_name_tokens(zip_path)

    # initiate output
    orbit_file = ''

    # search for orbit file
    #orbit_file_list = glob.glob(os.path.join(orbit_dir, 'S1*.EOF'))
    for orbit_file in orbit_file_list:
        # check if file validity
        if not item_valid(orbit_file, platform_id):
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
            break
        else:
            orbit_file = ''

    if not orbit_file:
        msg = 'No orbit file found from the file list provided.!'
        warnings.warn(msg)

    return orbit_file

