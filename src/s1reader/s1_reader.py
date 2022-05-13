import datetime
import os
import warnings
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import shapely

import isce3
from nisar.workflows.stage_dem import check_dateline
from s1reader.s1_burst_slc import Doppler, Sentinel1BurstSlc

# TODO evaluate if it make sense to combine below into a class
def as_datetime(t_str, fmt = "%Y-%m-%dT%H:%M:%S.%f"):
    '''Parse given time string to datetime.datetime object.

    Parameters:
    ----------
    t_str : string
        Time string to be parsed. (e.g., "2021-12-10T12:00:0.0")
    fmt : string
        Format of string provided. Defaults to az time format found in annotation XML.
        (e.g., "%Y-%m-%dT%H:%M:%S.%f").

    Returns:
    ------
    _ : datetime.datetime
        datetime.datetime object parsed from given time string.
    '''
    return datetime.datetime.strptime(t_str, fmt)

def parse_polynomial_element(elem, poly_name):
    '''Parse azimuth FM (Frequency Modulation) rate element to reference time and poly1d tuples.

    Parameters
    ----------
    elem : Element
        Element containing coefficients.
    poly_name : string
        Name of element containing azimuth time and polynomial coefficients.

    Returns:
    ------
    _ : tuple
        Tuple of time and Poly1d constructed from azimuth time and coefficients.
    '''
    ref_time = as_datetime(elem.find('azimuthTime').text)

    half_c = 0.5 * isce3.core.speed_of_light
    r0 = half_c * float(elem.find('t0').text)
    coeffs = [float(x) for x in elem.find(poly_name).text.split()]
    poly1d = isce3.core.Poly1d(coeffs, r0, half_c)
    return (ref_time, poly1d)

def get_nearest_polynomial(t_mid, time_poly_pair):
    '''Find polynomial closest to given sensing mid and return associated poly1d.

    Parameters
    ----------
    t_mid : datetime.datetime
        Middle of the burst
    time_poly_pair: list(tuple)
        List of tuples of time and associated Poly1d

    Returns:
    ------
    nearest_poly: list
        Polynomial coefficients associated with nearest time.
    '''
    # lambda calculating absolute time difference
    get_abs_dt = lambda t_mid, t_new : np.abs((t_mid - t_new).total_seconds())

    # calculate 1st dt and polynomial
    dt = get_abs_dt(t_mid, time_poly_pair[0][0])
    nearest_poly = time_poly_pair[0][1]

    # loop thru remaining time, polynomial pairs
    for x in time_poly_pair[1:]:
        temp_dt = get_abs_dt(t_mid, x[0])

        # stop looping if dt starts growing
        if temp_dt > dt:
            break

        # set dt and polynomial for next iteration
        dt, nearest_poly = temp_dt, x[1]

    return nearest_poly

def doppler_poly1d_to_lut2d(doppler_poly1d, starting_slant_range,
                            slant_range_res, shape, az_time_interval):
    '''Convert doppler poly1d to LUT2d.

    Parameters
    ----------
    doppler_poly1d : poly1d
        Poly1d object to be convereted.
    starting_slant_range : float
        Starting slant range of the burst.
    slant_range_res : float
        Slant-range pixel spacing of the burst.
    shape : tuple
        Tuples holding number of lines and samples of the burst.
    az_time_interval : float
        Azimth time interval of the burst.

    Returns:
    ------
    _ : LUT2d
        LUT2d calculated from poly1d.
    '''
    (n_lines, n_samples) = shape
    # calculate all slant ranges in grid
    slant_ranges = starting_slant_range + np.arange(n_samples) * slant_range_res

    # no az dependency, but LUT2d required, so ensure all az coords covered
    # offset by -2 days in seconds (referenece epoch)
    offset_ref_epoch = 2 * 24 *3600
    az_times = offset_ref_epoch + np.array([0, n_lines * az_time_interval])

    # calculate frequency for all slant range
    freq_1d = np.array([doppler_poly1d.eval(t) for t in slant_ranges])

    # init LUT2d (vstack freq since no az dependency) and return
    return isce3.core.LUT2d(slant_ranges, az_times,
                            np.vstack((freq_1d, freq_1d)))

def get_burst_orbit(sensing_start, sensing_stop, osv_list: ET.Element):
    '''Init and return ISCE3 orbit.

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

def calculate_centroid(lons, lats):
    '''Calculate burst centroid from boundary longitude/latitude points.

    Parameters:
    -----------
    lons : list
        Burst longitudes (degrees)
    lats : list
        Burst latitudes (degrees)

    Returns:
    --------
    _ : shapely.geometry.Point
        Burst center in degrees longitude and latitude
    '''
    proj = isce3.core.Geocent()

    # convert boundary points to geocentric
    xyz = [proj.forward([np.deg2rad(lon), np.deg2rad(lat), 0])
           for lon, lat in zip(lons, lats)]

    # get mean of corners as centroid
    xyz_centroid = np.mean(np.array(xyz), axis=0)

    # convert back to LLH
    llh_centroid = [np.rad2deg(x) for x in proj.inverse(xyz_centroid)]

    return shapely.geometry.Point(llh_centroid[:2])

def get_burst_centers_and_boundaries(tree):
    '''Parse grid points list and calculate burst center lat and lon

    Parameters:
    -----------
    tree : Element
        Element containing geolocation grid points.

    Returns:
    --------
    center_pts : list
        List of burst centroids ass shapely Points
    boundary_pts : list
        List of burst boundaries as shapely Polygons
    '''
    # find element tree
    grid_pt_list = tree.find('geolocationGrid/geolocationGridPointList')

    # read in all points
    n_grid_pts = int(grid_pt_list.attrib['count'])
    lines = np.empty(n_grid_pts)
    pixels = np.empty(n_grid_pts)
    lats = np.empty(n_grid_pts)
    lons = np.empty(n_grid_pts)
    for i, grid_pt in enumerate(grid_pt_list):
        lines[i] = int(grid_pt[2].text)
        pixels[i] = int(grid_pt[3].text)
        lats[i] = float(grid_pt[4].text)
        lons[i] = float(grid_pt[5].text)

    unique_line_indices = np.unique(lines)
    n_bursts = len(unique_line_indices) - 1
    center_pts = [[]] * n_bursts
    boundary_pts = [[]] * n_bursts

    # zip lines numbers of bursts together and iterate
    for i, (ln0, ln1) in enumerate(zip(unique_line_indices[:-1],
                                       unique_line_indices[1:])):
        # create masks for lines in current burst
        mask0 = lines==ln0
        mask1 = lines==ln1

        # reverse order of 2nd set of points so plots of boundaries
        # are not connected by a diagonal line
        burst_lons = np.concatenate((lons[mask0], lons[mask1][::-1]))
        burst_lats = np.concatenate((lats[mask0], lats[mask1][::-1]))

        center_pts[i] = calculate_centroid(burst_lons, burst_lats)

        poly = shapely.geometry.Polygon(zip(burst_lons, burst_lats))
        boundary_pts[i] = check_dateline(poly)

    return center_pts, boundary_pts

def burst_from_xml(annotation_path: str, orbit_path: str, tiff_path: str,
                   iw2_annotation_path: str, open_method=open):
    '''Parse bursts in Sentinel-1 annotation XML.

    Parameters:
    -----------
    annotation_path : str
        Path to Sentinel-1 annotation XML file of specific subswath and
        polarization.
    orbit_path : str
        Path the orbit file.
    tiff_path : str
        Path to tiff file holding Sentinel-1 SLCs.
    iw2_annotation_path : str
        Path to Sentinel-1 annotation XML file of IW2 subswath.
    open_method : function
        Function used to open annotation file.

    Returns:
    --------
    bursts : list
        List of Sentinel1BurstSlc objects found in annotation XML.
    '''
    _, tail = os.path.split(annotation_path)
    platform_id, subswath_id, _, pol = [x.upper() for x in tail.split('-')[:4]]

    # For IW mode, one burst has a duration of ~2.75 seconds and a burst
    # overlap of approximately ~0.4 seconds.
    # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/product-types-processing-levels/level-1
    # Additional precision calculated from averaging the differences between
    # burst sensing starts in prototyping test data
    burst_interval = 2.758277

    # Nearly all metadata loaded here is common to all bursts in annotation XML
    with open_method(annotation_path, 'r') as f:
        tree = ET.parse(f)

        product_info_element = tree.find('generalAnnotation/productInformation')
        azimuth_steer_rate = np.radians(float(product_info_element.find('azimuthSteeringRate').text))
        radar_freq = float(product_info_element.find('radarFrequency').text)
        range_sampling_rate = float(product_info_element.find('rangeSamplingRate').text)
        orbit_direction = product_info_element.find('pass').text

        image_info_element = tree.find('imageAnnotation/imageInformation')
        azimuth_time_interval = float(image_info_element.find('azimuthTimeInterval').text)
        slant_range_time = float(image_info_element.find('slantRangeTime').text)
        ascending_node_time = as_datetime(image_info_element.find('ascendingNodeTime').text)

        downlink_element =  tree.find('generalAnnotation/downlinkInformationList/downlinkInformation')
        prf_raw_data = float(downlink_element.find('prf').text)
        rank = int(downlink_element.find('downlinkValues/rank').text)
        range_chirp_ramp_rate = float(downlink_element.find('downlinkValues/txPulseRampRate').text)

        n_lines = int(tree.find('swathTiming/linesPerBurst').text)
        n_samples = int(tree.find('swathTiming/samplesPerBurst').text)

        az_rate_list_element = tree.find('generalAnnotation/azimuthFmRateList')
        poly_name = 'azimuthFmRatePolynomial'
        az_fm_rate_list = [parse_polynomial_element(x, poly_name) for x in az_rate_list_element]

        doppler_list_element = tree.find('dopplerCentroid/dcEstimateList')
        poly_name = 'dataDcPolynomial'
        doppler_list = [parse_polynomial_element(x, poly_name) for x in doppler_list_element]

        rng_processing_element = tree.find('imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing')
        rng_processing_bandwidth = float(rng_processing_element.find('processingBandwidth').text)
        range_window_type = str(rng_processing_element.find('windowType').text)
        range_window_coeff = float(rng_processing_element.find('windowCoefficient').text)

        orbit_number = int(tree.find('adsHeader/absoluteOrbitNumber').text)
        orbit_number_offset = 73 if  platform_id == 'S1A' else 202
        track_number = (orbit_number - orbit_number_offset) % 175 + 1

        center_pts, boundary_pts = get_burst_centers_and_boundaries(tree)

    wavelength = isce3.core.speed_of_light / radar_freq
    starting_range = slant_range_time * isce3.core.speed_of_light / 2
    range_pxl_spacing = isce3.core.speed_of_light / (2 * range_sampling_rate)

    # calculate the range at mid swath (mid of SM swath, mid of IW2 or mid of EW3)
    with open_method(iw2_annotation_path, 'r') as iw2_f:
        iw2_tree = ET.parse(iw2_f)
        iw2_slant_range_time = float(iw2_tree.find('imageAnnotation/imageInformation/slantRangeTime').text)
        iw2_n_samples = int(iw2_tree.find('swathTiming/samplesPerBurst').text)
        iw2_starting_range = iw2_slant_range_time * isce3.core.speed_of_light / 2
        iw2_mid_range = iw2_starting_range + 0.5 * iw2_n_samples * range_pxl_spacing

    # find orbit state vectors in 'Data_Block/List_of_OSVs'
    orbit_tree = ET.parse(orbit_path)
    osv_list = orbit_tree.find('Data_Block/List_of_OSVs')

    # load individual burst
    burst_list_elements = tree.find('swathTiming/burstList')
    n_bursts = int(burst_list_elements.attrib['count'])
    bursts = [[]] * n_bursts
    sensing_starts = [[]] * n_bursts
    sensing_times = [[]] * n_bursts

    for i, burst_list_element in enumerate(burst_list_elements):
        # get burst timing
        sensing_start = as_datetime(burst_list_element.find('azimuthTime').text)
        sensing_starts[i] = sensing_start
        sensing_times[i] = as_datetime(burst_list_element.find('sensingTime').text)
        dt = sensing_times[i] - ascending_node_time
        id_burst = int((dt.seconds + dt.microseconds / 1e6) // burst_interval)

        # choose nearest azimuth FM rate
        d_seconds = 0.5 * (n_lines - 1) * azimuth_time_interval
        sensing_mid = sensing_start + datetime.timedelta(seconds=d_seconds)
        az_fm_rate = get_nearest_polynomial(sensing_mid, az_fm_rate_list)

        # choose nearest doppler
        poly1d = get_nearest_polynomial(sensing_mid, doppler_list)
        lut2d = doppler_poly1d_to_lut2d(poly1d, starting_range,
                                        range_pxl_spacing, (n_lines, n_samples),
                                        azimuth_time_interval)
        doppler = Doppler(poly1d, lut2d)

        # get orbit from state vector list/element tree
        sensing_duration = datetime.timedelta(
            seconds=n_samples * azimuth_time_interval)
        orbit = get_burst_orbit(sensing_start, sensing_start + sensing_duration,
                                osv_list)

        # determine burst offset and dimensions
        # TODO move to own method
        first_valid_samples = [int(val) for val in burst_list_element.find('firstValidSample').text.split()]
        last_valid_samples = [int(val) for val in burst_list_element.find('lastValidSample').text.split()]

        first_valid_line = [x > 0 for x in first_valid_samples].index(True)
        n_valid_lines = [x > 0 for x in first_valid_samples].count(True)
        last_line = first_valid_line + n_valid_lines - 1

        first_valid_sample = max(first_valid_samples[first_valid_line],
                                 first_valid_samples[last_line])
        last_sample = min(last_valid_samples[first_valid_line],
                          last_valid_samples[last_line])


        burst_id = f't{track_number}_{subswath_id.lower()}_b{id_burst}'

        bursts[i] = Sentinel1BurstSlc(sensing_start, radar_freq, wavelength,
                                      azimuth_steer_rate, azimuth_time_interval,
                                      slant_range_time, starting_range, iw2_mid_range,
                                      range_sampling_rate, range_pxl_spacing,
                                      (n_lines, n_samples), az_fm_rate, doppler,
                                      rng_processing_bandwidth, pol, burst_id,
                                      platform_id, center_pts[i],
                                      boundary_pts[i], orbit, orbit_direction,
                                      tiff_path, i, first_valid_sample,
                                      last_sample, first_valid_line, last_line,
                                      range_window_type, range_window_coeff,
                                      rank, prf_raw_data, range_chirp_ramp_rate)
    return bursts

def _is_zip_annotation_xml(path: str, id_str: str) -> bool:
    ''' Check if path is annotation XMl and not calibration or rfi related

    path : str
        Path from SAFE zip to be checked
    id_str : str
        Subswath and polarization to be found. e.g. iw1_slc_vv

    Returns:
    --------
    _ : bool
        Whether or not given path is desired annotation XML
    '''
    # break path into tokens by '/'
    tokens = path.split('/')

    # check if 2nd to last path token, directory where file resides, is "annotation"
    # check if last path token, file name, contains ID string
    if tokens[-2] == 'annotation' and id_str in tokens[-1]:
        return True
    return False

def load_bursts(path: str, orbit_path: str, swath_num: int, pol: str = 'vv'):
    '''Find bursts in a Sentinel-1 zip file or a SAFE structured directory.

    Parameters:
    -----------
    path : str
        Path to Sentinel-1 zip file or SAFE directory
    orbit_path : str
        Path the orbit file.
    swath_num : int
        Integer of subswath of desired burst. {1, 2, 3}
    pol : str
        Polarization of desired burst. {hh, vv, hv, vh}

    Returns:
    --------
    bursts : list
        List of Sentinel1BurstSlc objects found in annotation XML.
    '''

    if swath_num < 1 or swath_num > 3:
        raise ValueError("swath_num not <1 or >3")

    # lower case polarity to be consistent with file naming convention
    pol = pol.lower()
    pols = ['vv', 'vh', 'hh', 'hv']
    if pol not in pols:
        raise ValueError(f"polarization not in {pols}")

    id_str = f'iw{swath_num}-slc-{pol}'

    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found')
    elif os.path.isdir(path):
        return _burst_from_safe_dir(path, id_str, orbit_path)
    elif os.path.isfile(path):
        return _burst_from_zip(path, id_str, orbit_path)
    else:
        raise ValueError(f'{path} is unsupported')

def _burst_from_zip(zip_path: str, id_str: str, orbit_path: str):
    '''Find bursts in a Sentinel-1 zip file.

    Parameters:
    -----------
    path : str
        Path to zip file.
    id_str: str
        Identifcation of desired burst. Format: iw[swath_num]-slc-[pol]
    orbit_path : str
        Path the orbit file.

    Returns:
    --------
    bursts : list
        List of Sentinel1BurstSlc objects found in annotation XML.
    '''
    with zipfile.ZipFile(zip_path, 'r') as z_file:
        z_file_list = z_file.namelist()

        # find annotation file - subswath of interest
        f_annotation = [f for f in z_file_list if _is_zip_annotation_xml(f, id_str)]
        if not f_annotation:
            raise ValueError(f"burst {id_str} not in SAFE: {zip_path}")
        f_annotation = f_annotation[0]

        # find annotation file - IW2
        iw2_id_str = f'iw2-{id_str[4:]}'
        iw2_f_annotation = [f for f in z_file_list if _is_zip_annotation_xml(f, iw2_id_str)]
        if not iw2_f_annotation:
            raise ValueError(f"burst {iw2_id_str} not in SAFE: {zip_path}")
        iw2_f_annotation = iw2_f_annotation[0]

        # find tiff file
        f_tiff = [f for f in z_file_list
                  if 'measurement' in f and id_str in f and 'tiff' in f]
        f_tiff = f'/vsizip/{zip_path}/{f_tiff[0]}' if f_tiff else ''

        bursts = burst_from_xml(f_annotation, orbit_path, f_tiff, iw2_f_annotation, z_file.open)
        return bursts

def _burst_from_safe_dir(safe_dir_path: str, id_str: str, orbit_path: str):
    '''Find bursts in a Sentinel-1 SAFE structured directory.

    Parameters:
    -----------
    path : str
        Path to SAFE directory.
    id_str: str
        Identifcation of desired burst. Format: iw[swath_num]-slc-[pol]
    orbit_path : str
        Path the orbit file.

    Returns:
    --------
    bursts : list
        List of Sentinel1BurstSlc objects found in annotation XML.
    '''

    # find annotation file - subswath of interest
    annotation_list = os.listdir(f'{safe_dir_path}/annotation')
    f_annotation = [f for f in annotation_list if id_str in f]
    if not f_annotation:
        raise ValueError(f"burst {id_str} not in SAFE: {safe_dir_path}")
    f_annotation = f'{safe_dir_path}/annotation/{f_annotation[0]}'

    # find annotation file - IW2
    iw2_id_str = f'iw2-{id_str[4:]}'
    iw2_f_annotation = [f for f in annotation_list if iw2_id_str in f]
    if not iw2_f_annotation:
        raise ValueError(f"burst {iw2_id_str} not in SAFE: {safe_dir_path}")
    iw2_f_annotation = f'{safe_dir_path}/annotation/{iw2_f_annotation[0]}'

    # find tiff file if measurement directory found
    if os.path.isdir(f'{safe_dir_path}/measurement'):
        measurement_list = os.listdir(f'{safe_dir_path}/measurement')
        f_tiff = [f for f in measurement_list
                  if 'measurement' in f and id_str in f and 'tiff' in f]
        f_tiff = f'{safe_dir_path}/measurement/{f_tiff[0]}' if f_tiff else ''
    else:
        warning_str = f'measurement directory not found in {safe_dir_path}'
        warnings.warn(warning_str)
        f_tiff = ''

    bursts = burst_from_xml(f_annotation, orbit_path, f_tiff, iw2_f_annotation)
    return bursts
