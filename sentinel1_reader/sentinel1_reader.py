import datetime
import os
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

import isce3
from sentinel1_burst_slc import Doppler, Sentinel1BurstSlc

# TODO evaluate if it make sense to combine below into a class
def as_datetime(t_str, fmt = "%Y-%m-%dT%H:%M:%S.%f"):
    '''
    Parse given time string to datetime.datetime object.

    Parameters:
    ----------
    t_str : string
        Time string to be parsed.
    fmt : string
        Format of string provided. Defaults to az time format found in annotation XML.

    Returns:
    ------
    _ : datetime.datetime
        datetime.dateime object parsed from given time string.
    '''
    return datetime.datetime.strptime(t_str, fmt)

def parse_polynomial_element(elem, poly_name):
    '''
    Parse az fm rate element to reference time and poly1d.

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
    '''
    Find polynomial closest to given sensing mid and return associated poly1d.

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
    for i, x in enumerate(time_poly_pair[1:]):
        temp_dt = get_abs_dt(t_mid, x[0])

        # stop looping if dt starts growing
        if temp_dt > dt:
            break

        # set dt and polynomial for next iteration
        dt, nearest_poly = temp_dt, x[1]

    return nearest_poly

def doppler_poly1d_to_lut2d(doppler_poly1d, starting_slant_range,
                            slant_range_res, shape, az_time_interval):
    '''
    Convert doppler poly1d to LUT2d.

    Parameters
    ----------
    doppler_poly1d : poly1d
        Poly1d object to be convereted.
    starting_slant_range : float
        Starting slant range of the burst.
    slant_range_res : float
        Slant range resolution of the burst.
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
    # TODO fix magic number of 2 days minus sensing_start
    az_times = 2*24*3600 + np.array([0, n_lines * az_time_interval])

    # calculate frequency for all slant range
    freq_1d = np.array([doppler_poly1d.eval(t) for t in slant_ranges])

    # init LUT2d (vstack freq since no az dependency) and return
    return isce3.core.LUT2d(slant_ranges, az_times,
                            np.vstack((freq_1d, freq_1d)))

def get_burst_centers(tree):
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

    ln_unique = np.unique(lines)
    center_lats = np.empty(len(ln_unique)-1)
    center_lons = np.empty(len(ln_unique)-1)
    center_pts = []
    for i, (ln0, ln1) in enumerate(zip(ln_unique[:-1], ln_unique[1:])):
        mask0 = lines==ln0
        mask1 = lines==ln1
        center_lat = np.mean(np.concatenate((lats[mask0], lats[mask1])))
        center_lon = np.mean(np.concatenate((lons[mask0], lons[mask1])))
        center_pts.append((center_lat, center_lon))

    return center_pts

def xml2bursts(annotation_path: str, tiff_path: str, open_method=open):
    '''
    Parse bursts in Sentinel 1 annotation XML.

    Parameters
    ----------
    annotation_path : str
        Path to Sentinel 1 annotation XML file.
    tiff_path : str
        Path to tiff file holding Sentinel 1 SLCs.
    open_method : function
        Function used to open annotation file.

    Returns:
    ------
    bursts : list
        List of Sentinel1BurstSlc objects found in annotation XML.
    '''
    _, tail = os.path.split(annotation_path)
    platform_id, subswath_id, _, pol = [x.upper() for x in tail.split('-')[:4]]
    burst_interval = 2.758277

    # load metadata common rdr2geo and geo2rdr
    # nearly all metadata loaded here is common to all bursts in annotation XML
    with open_method(annotation_path, 'r') as f:
        tree = ET.parse(f)

        product_info_element = tree.find('generalAnnotation/productInformation')
        azimuth_steer_rate = np.radians(float(product_info_element.find('azimuthSteeringRate').text))
        radar_freq = float(product_info_element.find('radarFrequency').text)
        range_sampling_rate = float(product_info_element.find('rangeSamplingRate').text)

        image_info_element = tree.find('imageAnnotation/imageInformation')
        azimuth_time_interval = float(image_info_element.find('azimuthTimeInterval').text)
        slant_range_time = float(image_info_element.find('slantRangeTime').text)
        ascending_node_time = as_datetime(image_info_element.find('ascendingNodeTime').text)

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

        orbit_number = int(tree.find('adsHeader/absoluteOrbitNumber').text)
        orbit_number_offset = 73 if  platform_id == 'S1A' else 202
        track_number = (orbit_number - orbit_number_offset) % 175 + 1

        center_pts = get_burst_centers(tree)

    wavelength = isce3.core.speed_of_light / radar_freq
    starting_range = slant_range_time * isce3.core.speed_of_light / 2
    range_pxl_spacing = isce3.core.speed_of_light / (2 * range_sampling_rate)

    # load individual burst
    burst_list_elements = tree.find('swathTiming/burstList')
    n_bursts = int(burst_list_elements.attrib['count'])
    bursts = [[]] * n_bursts
    sensing_starts = [[]] * n_bursts
    sensing_times = [[]] * n_bursts
    for i, burst_list_element in enumerate(burst_list_elements):
        # TODO make below @classmethod in sentinel1_burst_slc?
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
        lut1d = isce3.core.avg_lut2d_to_lut1d(lut2d)
        doppler = Doppler(poly1d, lut2d)

        # determine burst offset and dimensions
        # TODO move to own method
        first_valid_samples = [int(val) for val in burst_list_element.find('firstValidSample').text.split()]
        last_valid_samples = [int(val) for val in burst_list_element.find('lastValidSample').text.split()]

        first_valid_line = [x > 0 for x in first_valid_samples].index(True)
        n_valid_lines = [x > 0 for x in first_valid_samples].count(True)
        last_line = first_valid_line + n_valid_lines - 1

        first_valid_sample = max(first_valid_samples[first_valid_line], first_valid_samples[last_line])
        last_sample = min(last_valid_samples[first_valid_line], last_valid_samples[last_line])
        n_valid_samples = last_sample - first_valid_sample

        burst_id = f't{track_number}_{subswath_id.lower()}_{id_burst}'
        print(burst_id, f'{center_pts[i][0]:.3f}, 'f'{center_pts[i][1]:.3f}')
        bursts[i] = Sentinel1BurstSlc(sensing_start, radar_freq, wavelength,
                                      azimuth_steer_rate, azimuth_time_interval,
                                      slant_range_time, starting_range,
                                      range_sampling_rate, range_pxl_spacing,
                                      (n_lines, n_samples), az_fm_rate, doppler,
                                      rng_processing_bandwidth, pol, burst_id,
                                      platform_id, tiff_path, i, first_valid_sample,
                                      last_sample, first_valid_line, last_line)

    return bursts

def zip2bursts(zip_path: str, n_subswath: int, pol: str):
    '''
    Find bursts in a Sentinel 1 zip file

    zip_path : str
        Path the zip file.
    n_subswath : int
        Integer of subswath of desired burst. {1, 2, 3}
    pol : str
        Polarization of desired burst. {HH, HV}

    Returns:
    ------
    bursts : list
        List of Sentinel1BurstSlc objects found in annotation XML.
    '''
    id_str = f'iw{n_subswath}-slc-{pol}'
    with zipfile.ZipFile(zip_path, 'r') as z_file:
        # find annotation file
        f_annotation = [f for f in z_file.namelist() if 'calibration' not in f and id_str in f and 'annotation' in f][0]

        # find tiff file
        f_tiff = [f for f in z_file.namelist() if 'measurement' in f and id_str in f and 'tiff' in f][0]
        f_tiff = f'/vsizip/{zip_path}/{f_tiff}'

        bursts = xml2bursts(f_annotation, f_tiff, z_file.open)
        return bursts
