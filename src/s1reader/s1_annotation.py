'''
A module to load annotation files for Sentinel-1 IW SLC SAFE data
To be used for the class "Sentinel1BurstSlc"
'''

from dataclasses import dataclass

import datetime
import os
import lxml.etree as ET
import zipfile

import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from packaging import version

# Minimum IPF version from which the S1 product's Noise Annotation
# Data Set (NADS) includes azimuth noise vector annotation
min_ipf_version_az_noise_vector = version.parse('2.90')

@dataclass
class AnnotationBase:
    '''
    A virtual base class of the inheriting annotation class i.e. Product, Calibration, and Noise.
    Not intended for standalone use.
    '''
    xml_et: ET

    @classmethod
    def _parse_scalar(cls, path_field: str, str_type: str):
        '''A class method that parse the scalar value in AnnotationBase.xml_et

        Parameters
        ----------
        path_field : str
            Field in the xml_et to parse
        str_type : str
            Specify how the texts in the field will be parsed.
            accepted values:
                {'datetime', 'scalar_int', 'scalar_float', 'vector_int', 'vector_float', 'str'}

        Returns
        -------
        val_out: {datetime.datetime, int, float, np.array, str}
            Parsed data in the annotation
            Datatype of vel_out follows str_type.
            val_out becomes np.array when str_type is vector*

        '''

        elem_field = cls.xml_et.find(path_field)
        if str_type == 'datetime':
            val_out = datetime.datetime.strptime(elem_field.text, '%Y-%m-%dT%H:%M:%S.%f')

        elif str_type == 'scalar_int':
            val_out = int(elem_field.text)

        elif str_type == 'scalar_float':
            val_out = float(elem_field.text)

        elif str_type == 'vector_int':
            val_out = np.array([int(strin) for strin in elem_field.text.split()])

        elif str_type == 'vector_float':
            val_out = np.array([float(strin) for strin in elem_field.text.split()])

        elif str_type == 'str':
            val_out = elem_field.text

        else:
            raise ValueError(f'Unsupported type the element: "{str_type}"')

        return val_out

    @classmethod
    def _parse_vectorlist(cls, name_vector_list: str, name_vector: str, str_type: str):
        '''A class method that parse the list of the values from xml_et in the class

        Parameters
        ----------
        name_vector_list : str
            List Field in the xml_et to parse
        name_vector : str
            Name of the field in each elements of the VectorList
            (e.g. 'noiseLut' in 'noiseVectorList')
        str_type : str
            Specify how the texts in the field will be parsed
            accepted values:
                {'datetime', 'scalar_int', 'scalar_float', 'vector_int', 'vector_float', 'str'}

        Returns
        -------
        val_out: list
            Parsed data in the annotation

        '''

        element_to_parse = cls.xml_et.find(name_vector_list)
        num_element = len(element_to_parse)

        list_out = [None]*num_element

        if str_type == 'datetime':
            for i,elem in enumerate(element_to_parse):
                str_elem = elem.find(name_vector).text
                list_out[i] = datetime.datetime.strptime(str_elem, '%Y-%m-%dT%H:%M:%S.%f')
            list_out = np.array(list_out)

        elif str_type == 'scalar_int':
            for i,elem in enumerate(element_to_parse):
                str_elem = elem.find(name_vector).text
                list_out[i] = int(str_elem)

        elif str_type == 'scalar_float':
            for i,elem in enumerate(element_to_parse):
                str_elem = elem.find(name_vector).text
                list_out[i] = float(str_elem)

        elif str_type == 'vector_int':
            for i,elem in enumerate(element_to_parse):
                str_elem = elem.find(name_vector).text
                list_out[i] = np.array([int(strin) for strin in str_elem.split()])

        elif str_type == 'vector_float':
            for i,elem in enumerate(element_to_parse):
                str_elem = elem.find(name_vector).text
                list_out[i] = np.array([float(strin) for strin in str_elem.split()])

        elif str_type == 'str':
            list_out = element_to_parse[0].find(name_vector).text

        else:
            raise ValueError(f'Cannot recognize the type of the element: {str_type}')

        return list_out


@dataclass
class CalibrationAnnotation(AnnotationBase):
    '''Reader for Calibration Annotation Data Set (CADS)'''

    basename_annotation: str
    list_azimuth_time: np.ndarray
    list_line: list
    list_pixel: None
    list_sigma_nought: list
    list_beta_nought : list
    list_gamma: list
    list_dn: list

    @classmethod
    def from_et(cls, et_in: ET, path_annotation: str):
        '''
        Extracts the list of calibration informaton from etree from
        the Calibration Annotation Data Set (CADS).
        Parameters:
        -----------
        et_in: ET
            ElementTree From CADS .xml file

        Returns:
        --------
        cls: CalibrationAnnotation
            Instance of CalibrationAnnotation initialized by the input parameter
        '''

        cls.xml_et = et_in
        cls.basename_annotation = \
            os.path.basename(path_annotation)

        cls.list_azimuth_time = \
            cls._parse_vectorlist('calibrationVectorList',
                                  'azimuthTime',
                                  'datetime')
        cls.list_line = \
            cls._parse_vectorlist('calibrationVectorList',
                                  'line',
                                  'scalar_int')
        cls.list_pixel = \
            cls._parse_vectorlist('calibrationVectorList',
                                  'pixel',
                                  'vector_int')
        cls.list_sigma_nought = \
            cls._parse_vectorlist('calibrationVectorList',
                                'sigmaNought',
                                'vector_float')
        cls.list_beta_nought = \
            cls._parse_vectorlist('calibrationVectorList',
                                  'betaNought',
                                  'vector_float')
        cls.list_gamma = \
            cls._parse_vectorlist('calibrationVectorList',
                                  'gamma',
                                  'vector_float')
        cls.list_dn = \
            cls._parse_vectorlist('calibrationVectorList',
                                  'dn',
                                  'vector_float')

        return cls


@dataclass
class NoiseAnnotation(AnnotationBase):
    '''
    Reader for Noise Annotation Data Set (NADS) for IW SLC
    Based on ESA documentation: "Thermal Denoising of Products Generated by the S-1 IPF"
    '''

    basename_annotation: str
    rg_list_azimuth_time: np.ndarray
    rg_list_line: list
    rg_list_pixel: list
    rg_list_noise_range_lut: list
    az_first_azimuth_line: int
    az_first_range_sample: int
    az_last_azimuth_line: int
    az_last_range_sample: int
    az_line: np.ndarray
    az_noise_azimuth_lut: np.ndarray

    @classmethod
    def from_et(cls,et_in: ET, ipf_version: version.Version, path_annotation: str):
        '''
        Extracts list of noise information from etree

        Parameter
        ----------
        et_in : xml.etree.ElementTree
            Parsed NADS annotation .xml

        Return
        -------
        cls: NoiseAnnotation
            Parsed NADS from et_in
        '''

        cls.xml_et = et_in
        cls.basename_annotation = os.path.basename(path_annotation)

        if ipf_version < min_ipf_version_az_noise_vector:  # legacy SAFE data
            cls.rg_list_azimuth_time = \
                cls._parse_vectorlist('noiseVectorList',
                                      'azimuthTime',
                                      'datetime')
            cls.rg_list_line = \
                cls._parse_vectorlist('noiseVectorList',
                                      'line',
                                      'scalar_int')
            cls.rg_list_pixel = \
                cls._parse_vectorlist('noiseVectorList',
                                      'pixel',
                                      'vector_int')
            cls.rg_list_noise_range_lut = \
                cls._parse_vectorlist('noiseVectorList',
                                      'noiseLut',
                                      'vector_float')

            cls.az_first_azimuth_line = None
            cls.az_first_range_sample = None
            cls.az_last_azimuth_line = None
            cls.az_last_range_sample = None
            cls.az_line = None
            cls.az_noise_azimuth_lut = None

        else:
            cls.rg_list_azimuth_time = \
                cls._parse_vectorlist('noiseRangeVectorList',
                                      'azimuthTime',
                                      'datetime')
            cls.rg_list_line = \
                cls._parse_vectorlist('noiseRangeVectorList',
                                      'line',
                                      'scalar_int')
            cls.rg_list_pixel = \
                cls._parse_vectorlist('noiseRangeVectorList',
                                      'pixel',
                                      'vector_int')
            cls.rg_list_noise_range_lut = \
                cls._parse_vectorlist('noiseRangeVectorList',
                                      'noiseRangeLut',
                                      'vector_float')
            cls.az_first_azimuth_line = \
                cls._parse_vectorlist('noiseAzimuthVectorList',
                                      'firstAzimuthLine',
                                      'scalar_int')[0]
            cls.az_first_range_sample = \
                cls._parse_vectorlist('noiseAzimuthVectorList',
                                      'firstRangeSample',
                                      'scalar_int')[0]
            cls.az_last_azimuth_line = \
                cls._parse_vectorlist('noiseAzimuthVectorList',
                                      'lastAzimuthLine',
                                      'scalar_int')[0]
            cls.az_last_range_sample = \
                cls._parse_vectorlist('noiseAzimuthVectorList',
                                      'lastRangeSample',
                                      'scalar_int')[0]
            cls.az_line = \
                cls._parse_vectorlist('noiseAzimuthVectorList',
                                      'line',
                                      'vector_int')[0]
            cls.az_noise_azimuth_lut = \
                cls._parse_vectorlist('noiseAzimuthVectorList',
                                      'noiseAzimuthLut',
                                      'vector_float')[0]

        return cls


@dataclass
class ProductAnnotation(AnnotationBase):
    '''
    Reader for L1 Product annotation for IW SLC
    For Elevation Antenna Pattern (EAP) correction
    '''

    image_information_slant_range_time: float

    # Attributes to be used when determining what AUX_CAL to load
    instrument_cfg_id: int

    # elevation_angle:
    antenna_pattern_azimuth_time: list
    antenna_pattern_slant_range_time: list
    antenna_pattern_elevation_angle: list
    antenna_pattern_elevation_pattern: list

    ascending_node_time: datetime.datetime
    number_of_samples: int
    range_sampling_rate: float

    slant_range_time: float

    # FM rate parameters for
    # azimuth FM rate mismatch mitigation
    vec_aztime_fm_rate: np.ndarray
    lut_coeff_fm_rate: np.ndarray
    vec_tau0_fm_rate:np.ndarray

    # Doppler centroid (DC) parameters for
    # azimuth FM rate mismatch mitigation
    vec_aztime_dc: np.ndarray
    lut_coeff_dc: np.ndarray
    vec_tau0_dc: np.ndarray


    @classmethod
    def from_et(cls, et_in: ET):
        '''
        Extracts list of product information from etree from
        L1 annotation data set (LADS) Parameter

        ----------
        et_in : xml.etree.ElementTree
            Parsed LADS annotation .xml

        Return
        -------
        cls: ProductAnnotation
            Parsed LADS from et_in
        '''

        cls.xml_et = et_in

        cls.antenna_pattern_azimuth_time = \
            cls._parse_vectorlist('antennaPattern/antennaPatternList',
                                  'azimuthTime',
                                  'datetime')
        cls.antenna_pattern_slant_range_time = \
            cls._parse_vectorlist('antennaPattern/antennaPatternList',
                                  'slantRangeTime',
                                  'vector_float')
        cls.antenna_pattern_elevation_angle = \
            cls._parse_vectorlist('antennaPattern/antennaPatternList',
                                  'elevationAngle',
                                  'vector_float')
        cls.antenna_pattern_elevation_pattern = \
            cls._parse_vectorlist('antennaPattern/antennaPatternList',
                                  'elevationPattern',
                                  'vector_float')

        cls.image_information_slant_range_time = \
            cls._parse_scalar('imageAnnotation/imageInformation/slantRangeTime',
                              'scalar_float')
        cls.ascending_node_time = \
            cls._parse_scalar('imageAnnotation/imageInformation/ascendingNodeTime',
                              'datetime')
        cls.number_of_samples = \
            cls._parse_scalar('imageAnnotation/imageInformation/numberOfSamples',
                              'scalar_int')
        cls.number_of_samples = \
            cls._parse_scalar('imageAnnotation/imageInformation/numberOfSamples',
                              'scalar_int')
        cls.range_sampling_rate = \
            cls._parse_scalar('generalAnnotation/productInformation/rangeSamplingRate',
                              'scalar_float')
        cls.slant_range_time =  \
            cls._parse_scalar('imageAnnotation/imageInformation/slantRangeTime',
                              'scalar_float')

        cls.inst_config_id = \
            cls._parse_scalar('generalAnnotation/downlinkInformationList/downlinkInformation/'
                              'downlinkValues/instrumentConfigId',
                              'scalar_int')

        # Extra from ET FM rate parameters for mismatch mitigation
        cls.vec_aztime_fm_rate = \
            cls._parse_vectorlist('generalAnnotation/azimuthFmRateList',
                                  'azimuthTime',
                                  'datetime')

        if cls.xml_et.find('generalAnnotation/azimuthFmRateList')[0].find('c0') is None:
            cls.lut_coeff_fm_rate = \
                np.array(cls._parse_vectorlist('generalAnnotation/azimuthFmRateList',
                                          'azimuthFmRatePolynomial',
                                          'vector_float'))

        else:
            # Old annotation format
            vec_c0 = cls._parse_vectorlist('generalAnnotation/azimuthFmRateList',
                                           'c0',
                                           'scalar_float')
            vec_c1 = cls._parse_vectorlist('generalAnnotation/azimuthFmRateList',
                                           'c1',
                                           'scalar_float')
            vec_c2 = cls._parse_vectorlist('generalAnnotation/azimuthFmRateList',
                                           'c2',
                                           'scalar_float')
            cls.lut_coeff_fm_rate = np.array([vec_c0, vec_c1, vec_c2]).transpose()

        cls.vec_tau0_fm_rate = np.array(cls._parse_vectorlist('generalAnnotation/azimuthFmRateList',
                                                     't0',
                                                     'scalar_float'))

        # Extract doppler centroid parameters for mismatch mitigation
        cls.vec_aztime_dc = cls._parse_vectorlist('dopplerCentroid/dcEstimateList',
                                                    'azimuthTime',
                                                    'datetime')

        cls.lut_coeff_dc = np.array(cls._parse_vectorlist('dopplerCentroid/dcEstimateList',
                                                 'dataDcPolynomial',
                                                 'vector_float'))

        cls.vec_tau0_dc = np.array(cls._parse_vectorlist('dopplerCentroid/dcEstimateList',
                                                't0',
                                                'scalar_float'))


        return cls


@dataclass
class AuxCal(AnnotationBase):
    '''AUX_CAL information for EAP correction'''

    beam_nominal_near_range: float
    beam_nominal_far_range: float

    elevation_angle_increment: float
    elevation_antenna_pattern: np.ndarray

    azimuth_angle_increment: float
    azimuth_antenna_pattern: np.ndarray
    azimuth_antenna_element_pattern_increment: float
    azimuth_antenna_element_pattern: float
    absolute_calibration_constant: float
    noise_calibration_factor: float

    @classmethod
    def load_from_zip_file(cls, path_aux_cal_zip: str, pol: str, str_swath: str):
        '''
        A class method that extracts list of information AUX_CAL from the input ET.

        Parameters
        ---------
        path_aux_cal_zip : str
            Path to the AUX_CAL .zip file
        pol: str {'vv','vh','hh','hv'}
            Polarization of interest
        str_swath: {'iw1','iw2','iw3'}
            IW subswath of interest

        Returns
        -------
        cls: AuxCal class populated by et_in in the parameter

        '''

        if not path_aux_cal_zip.endswith('.zip'):
            raise ValueError('Only AUX_CAL files in .zip format are accepted.')

        if os.path.exists(path_aux_cal_zip):
            str_safe_aux_cal = os.path.basename(path_aux_cal_zip).replace('.zip','')
            # detect the platform from path_aux_cal_zip
            str_platform = str_safe_aux_cal.split('_')[0]
        else:
            raise ValueError(f'Cannot find AUX_CAL .zip file: {path_aux_cal_zip}')

        with zipfile.ZipFile(path_aux_cal_zip, 'r') as zipfile_aux_cal:
            filepath_xml = f'{str_safe_aux_cal}/data/{str_platform.lower()}-aux-cal.xml'
            # check if the input file has the aux_cal .xml file to load
            list_files_in_zip = [zf.filename for zf in zipfile_aux_cal.filelist]

            if filepath_xml not in list_files_in_zip:
                raise ValueError(f'Cannot find {filepath_xml} in '
                                 f'zip file {path_aux_cal_zip}.\n'
                                  'Make sure if the legit AUX_CAL .zip file is provided.')

            with zipfile_aux_cal.open(filepath_xml,'r') as f_aux_cal:
                et_in = ET.parse(f_aux_cal)

        calibration_params_list = et_in.find('calibrationParamsList')
        for calibration_params in calibration_params_list:
            swath_xml = calibration_params.find('swath').text
            polarisation_xml = calibration_params.find('polarisation').text
            if polarisation_xml == pol.upper() and swath_xml==str_swath.upper():
                cls.beam_nominal_near_range = \
                    float(calibration_params.
                        find('elevationAntennaPattern/beamNominalNearRange').text)
                cls.beam_nominal_far_range = \
                    float(calibration_params.
                        find('elevationAntennaPattern/beamNominalFarRange').text)
                cls.elevation_angle_increment = \
                    float(calibration_params.
                        find('elevationAntennaPattern/elevationAngleIncrement').text)

                n_val = \
                    int(calibration_params.
                        find('elevationAntennaPattern/values').attrib['count'])
                arr_eap_val = \
                    np.array([float(token_val) for \
                             token_val in calibration_params.
                            find('elevationAntennaPattern/values').text.split()])

                if n_val == len(arr_eap_val):
                    # Provided in real numbers: In case of AUX_CAL for old IPFs.
                    cls.elevation_antenna_pattern = arr_eap_val
                elif n_val*2 == len(arr_eap_val):
                    # Provided in complex numbers: In case of recent IPFs e.g. 3.10
                    cls.elevation_antenna_pattern = arr_eap_val[0::2] + arr_eap_val[1::2] * 1.0j
                else:
                    raise ValueError('The number of values does not match. '
                                    f'n_val={n_val}, '
                                    f'#len(elevationAntennaPattern/values)={len(arr_eap_val)}')

                cls.azimuth_angle_increment = \
                    float(calibration_params.
                        find('azimuthAntennaPattern/azimuthAngleIncrement').text)
                cls.azimuth_antenna_pattern = \
                    np.array([float(token_val) for \
                              token_val in calibration_params.
                              find('azimuthAntennaPattern/values').text.split()])

                cls.azimuth_antenna_element_pattern_increment = \
                    float(calibration_params.
                        find('azimuthAntennaElementPattern/azimuthAngleIncrement').text)
                cls.azimuth_antenna_element_pattern = \
                    np.array([float(token_val) for \
                              token_val in calibration_params.
                              find('azimuthAntennaElementPattern/values').text.split()])

                cls.absolute_calibration_constant = \
                    float(calibration_params.find('absoluteCalibrationConstant').text)
                cls.noise_calibration_factor = \
                    float(calibration_params.find('noiseCalibrationFactor').text)

        return cls

def closest_block_to_azimuth_time(vector_azimuth_time: np.ndarray,
                                  azimuth_time_burst: datetime.datetime) -> int:
    '''
    Find the id of the closest data block in annotation.
    To be used when populating BurstNoise, BurstCalibration, and BurstEAP.

    Parameters
    ----------
    vector_azimuth_time : np.ndarray
        numpy array azimuth time whose data type is datetime.datetime
    azimuth_time_burst: datetime.datetime
        Azimuth time of the burst

    Returns
    -------
    _: int
        Index of vector_azimuth_time that is the closest to azimuth_burst_time

    '''

    return np.argmin(np.abs(vector_azimuth_time-azimuth_time_burst))


@dataclass
class BurstNoise:
    '''Noise correction information for Sentinel-1 burst'''
    basename_nads: str
    range_azimith_time: datetime.datetime
    range_line: float
    range_pixel: np.ndarray
    range_lut: np.ndarray

    azimuth_first_azimuth_line: int
    azimuth_first_range_sample: int
    azimuth_last_azimuth_line: int
    azimuth_last_range_sample: int
    azimuth_line: np.ndarray
    azimuth_lut: np.ndarray

    line_from: int
    line_to: int

    @classmethod
    def from_noise_annotation(cls, noise_annotation: NoiseAnnotation,
                              azimuth_time: datetime.datetime,
                              line_from: int,
                              line_to: int,
                              ipf_version: version.Version):
        '''
        Extracts the noise correction information for individual burst from NoiseAnnotation

        Parameters
        ----------
        noise_annotation: NoiseAnnotation
            Subswath-wide noise annotation information
        azimuth_time : datetime.datetime
            Azimuth time of the burst
        line_from: int
            First line of the burst in the subswath
        line_to: int
            Last line of the burst in the subswath
        ipf_version: float
            IPF version of the SAFE data

        Returns
        -------
        cls: BurstNoise
            Instance of BurstNoise initialized by the input parameters

        '''

        basename_nads = noise_annotation.basename_annotation
        id_closest = closest_block_to_azimuth_time(noise_annotation.rg_list_azimuth_time,
                                                   azimuth_time)

        range_azimith_time = noise_annotation.rg_list_azimuth_time[id_closest]
        range_line = noise_annotation.rg_list_line[id_closest]
        range_pixel = noise_annotation.rg_list_pixel[id_closest]
        range_lut = noise_annotation.rg_list_noise_range_lut[id_closest]

        azimuth_first_azimuth_line = noise_annotation.az_first_azimuth_line
        azimuth_first_range_sample = noise_annotation.az_first_range_sample
        azimuth_last_azimuth_line = noise_annotation.az_last_azimuth_line
        azimuth_last_range_sample = noise_annotation.az_last_range_sample

        if ipf_version >= min_ipf_version_az_noise_vector:
            # Azimuth noise LUT exists - crop to the extent of the burst
            id_top = np.argmin(np.abs(noise_annotation.az_line-line_from))
            id_bottom = np.argmin(np.abs(noise_annotation.az_line-line_to))

            # put some margin when possible
            if id_top > 0:
                id_top -= 1
            if id_bottom < len(noise_annotation.az_line)-1:
                id_bottom += 1
            azimuth_line = noise_annotation.az_line[id_top:id_bottom + 1]
            azimuth_lut = noise_annotation.az_noise_azimuth_lut[id_top:id_bottom + 1]

        else:
            azimuth_line = None
            azimuth_lut = None

        return cls(basename_nads, range_azimith_time, range_line, range_pixel, range_lut,
                   azimuth_first_azimuth_line, azimuth_first_range_sample,
                   azimuth_last_azimuth_line, azimuth_last_range_sample,
                   azimuth_line, azimuth_lut,
                   line_from, line_to)


    def compute_thermal_noise_lut(self, shape_lut):
        '''
        Calculate thermal noise LUT whose shape is `shape_lut`

        Parameter:
        ----------
        shape_lut: tuple or list
            Shape of the output LUT

        Returns
        -------
        arr_lut_total: np.ndarray
            2d array containing thermal noise correction look up table values
        '''

        nrows, ncols = shape_lut

        # Interpolate the range noise vector
        rg_lut_interp_obj = InterpolatedUnivariateSpline(self.range_pixel,
                                                         self.range_lut,
                                                         k=1)
        if self.azimuth_last_range_sample is not None:
            vec_rg = np.arange(self.azimuth_last_range_sample + 1)
        else:
            vec_rg = np.arange(ncols)
        rg_lut_interpolated = rg_lut_interp_obj(vec_rg)

        # Interpolate the azimuth noise vector
        if (self.azimuth_line is None) or (self.azimuth_lut is None):
            az_lut_interpolated = np.ones(nrows)
        else:  # IPF >= 2.90
            az_lut_interp_obj = InterpolatedUnivariateSpline(self.azimuth_line,
                                                             self.azimuth_lut,
                                                             k=1)
            vec_az = np.arange(self.line_from, self.line_to + 1)
            az_lut_interpolated = az_lut_interp_obj(vec_az)

        arr_lut_total = np.matmul(az_lut_interpolated[..., np.newaxis],
                                  rg_lut_interpolated[np.newaxis, ...])

        return arr_lut_total


@dataclass
class BurstCalibration:
    '''Calibration information for Sentinel-1 IW SLC burst
    '''
    basename_cads: str
    azimuth_time: datetime.datetime = None
    line: float = None
    pixel: np.ndarray = None
    sigma_naught: np.ndarray = None
    beta_naught: np.ndarray = None
    gamma: np.ndarray = None
    dn: np.ndarray = None

    @classmethod
    def from_calibration_annotation(cls, calibration_annotation: CalibrationAnnotation,
                                    azimuth_time: datetime.datetime):
        '''
        A class method that extracts the calibration info for the burst

        Parameters
        ----------
        calibration_annotation: CalibrationAnnotation
            A subswath-wide calibraion information from CADS file
        azimuth_time: datetime.datetime
            Azimuth time of the burst

        Returns
        -------
        cls: BurstCalibration
            Radiometric correction information for the burst
        '''

        basename_cads = calibration_annotation.basename_annotation
        id_closest = closest_block_to_azimuth_time(calibration_annotation.list_azimuth_time,
                                                   azimuth_time)

        azimuth_time = calibration_annotation.list_azimuth_time[id_closest]
        line = calibration_annotation.list_line[id_closest]
        pixel = calibration_annotation.list_pixel[id_closest]
        sigma_naught = calibration_annotation.list_sigma_nought[id_closest]
        gamma = calibration_annotation.list_gamma[id_closest]
        dn = calibration_annotation.list_dn[id_closest]

        # Check if all values in the list of beta_naught LUTs are the same
        matrix_beta_naught = np.array(calibration_annotation.list_beta_nought)
        if matrix_beta_naught.min() == matrix_beta_naught.max():
            beta_naught = np.min(matrix_beta_naught)
        else:
            # TODO Switch to LUT-based method when there is significant changes in the array
            beta_naught = np.mean(matrix_beta_naught)

        return cls(basename_cads, azimuth_time, line, pixel,
                   sigma_naught, beta_naught, gamma, dn)


@dataclass
class BurstEAP:
    '''EAP correction information for Sentinel-1 IW SLC burst
    '''
    # from LADS
    freq_sampling: float  # range sampling rate
    eta_start: datetime.datetime
    tau_0: float  # imageInformation/slantRangeTime
    tau_sub: np.ndarray  # antennaPattern/slantRangeTime
    theta_sub: np.ndarray  # antennaPattern/elevationAngle
    azimuth_time: datetime.datetime
    ascending_node_time: datetime.datetime

    # from AUX_CAL
    gain_eap: np.ndarray  # elevationAntennaPattern
    delta_theta:float  # elavationAngleIncrement

    @classmethod
    def from_product_annotation_and_aux_cal(cls, product_annotation: ProductAnnotation,
                                            aux_cal: AuxCal, azimuth_time: datetime.datetime):
        '''
        A class method that extracts the EAP correction info for the IW SLC burst

        Parameters
        ----------
        product_annotation: ProductAnnotation
            A swath-wide product annotation class

        aux_cal: AuxCal
            AUX_CAL information that corresponds to the sensing time

        azimuth_time: datetime.datetime
            Azimuth time of the burst

        Returns
        -------
        cls: BurstEAP
            A burst-wide information for EAP correction

        '''
        id_closest = closest_block_to_azimuth_time(product_annotation.antenna_pattern_azimuth_time,
                                                   azimuth_time)
        freq_sampling = product_annotation.range_sampling_rate
        eta_start = azimuth_time
        tau_0 = product_annotation.slant_range_time
        tau_sub = product_annotation.antenna_pattern_slant_range_time[id_closest]
        theta_sub = product_annotation.antenna_pattern_elevation_angle[id_closest]
        gain_eap = aux_cal.elevation_antenna_pattern
        delta_theta = aux_cal.elevation_angle_increment

        ascending_node_time = product_annotation.ascending_node_time

        return cls(freq_sampling, eta_start, tau_0, tau_sub, theta_sub,
                   azimuth_time, ascending_node_time,
                   gain_eap, delta_theta)


    def compute_eap_compensation_lut(self, num_sample):
        '''
        Returns LUT for EAP compensation whose size is `num_sample`.
        Based on ESA docuemnt :
        "Impact of the Elevation Antenna Pattern Phase Compensation
         on the Interferometric Phase Preservation"

        Document URL:
        https://sentinel.esa.int/documents/247904/1653440/Sentinel-1-IPF_EAP_Phase_correction

        Parameter:
        num_sample: int
            Size of the output LUT

        Return:
        -------
            gain_eap_interpolatd: EAP phase for the burst to be compensated

        '''

        n_elt = len(self.gain_eap)

        theta_am = (np.arange(n_elt) - (n_elt - 1) / 2) * self.delta_theta

        delta_anx = self.eta_start - self.ascending_node_time
        theta_offnadir = self._anx2roll(delta_anx.seconds + delta_anx.microseconds * 1.0e-6)

        theta_eap = theta_am + theta_offnadir

        tau = self.tau_0 + np.arange(num_sample) / self.freq_sampling

        theta = np.interp(tau, self.tau_sub, self.theta_sub)

        interpolator_gain = interp1d(theta_eap, self.gain_eap)
        gain_eap_interpolated = interpolator_gain(theta)

        return gain_eap_interpolated


    def _anx2roll(self, delta_anx):
        '''
        Returns the Platform nominal roll as function of elapsed time from
        ascending node crossing time (ANX). (Implemented from S1A documentation.)

        Code copied from ISCE2.

        The units in this function is based on the reference documentation in the URL below:
        https://sentinel.esa.int/documents/247904/1653440/Sentinel-1-IPF_EAP_Phase_correction

        Parameters
        ----------
        delta_anx: float
            elapsed time from ascending node crossing time

        Returns
        -------
        nominal_roll: float
            Estimated nominal roll (degrees)
        '''

        # Estimate altitude based on time elapsed since ANX
        altitude = self._anx2height(delta_anx)

        # Reference altitude (km)
        href = 711.700

        # Reference boresight at reference altitude (degrees)
        boresight_ref = 29.450

        # Partial derivative of roll vs altitude (degrees/m)
        alpha_roll = 0.0566

        # Estimate nominal roll i.e. theta off nadir (degrees)
        nominal_roll = boresight_ref - alpha_roll * (altitude/1000.0 - href)

        return nominal_roll

    @classmethod
    def _anx2height(cls, delta_anx):
        '''
        Returns the platform nominal height as function of elapse time from
        ascending node crossing time (ANX).
        Implementation from S1A documention.

        Code copied from ISCE2.


        Parameters:
        -----------
        delta_anx: float
            elapsed time from ANX time


        Returns:
        --------
        h_t: float
            nominal height of the platform

        '''

        # Average height (m)
        h_0 = 707714.8  #;m

        # Perturbation amplitudes (m)
        h = np.array([8351.5, 8947.0, 23.32, 11.74])

        # Perturbation phases (radians)
        phi = np.array([3.1495, -1.5655 , -3.1297, 4.7222])

        # Orbital time period (seconds)
        t_orb = (12*24*60*60) / 175.

        # Angular velocity (rad/sec)
        worb = 2*np.pi / t_orb

        # Evaluation of series
        h_t = h_0
        for i, h_i in enumerate(h):
            h_t += h_i * np.sin((i+1) * worb * delta_anx + phi[i])

        return h_t


@dataclass
class BurstExtendedCoeffs:
    '''
    Segments of FM rate / Doppler centroid polynomial coefficients.
    For (linear) interpolation of FM rate / Doppler Centroid along azimuth.
    To be used for calculating azimuth FM rate mismatch mitigation
    '''

    # FM rate
    vec_aztime_fm_rate: np.ndarray
    lut_coeff_fm_rate: np.ndarray
    vec_tau0_fm_rate:np.ndarray

    # Doppler centroid
    vec_aztime_dc: np.ndarray
    lut_coeff_dc: np.ndarray
    vec_tau0_dc: np.ndarray

    @classmethod
    def from_product_annotation_and_burst(cls,
                                          product_annotation: ProductAnnotation,
                                          sensing_start: datetime.datetime,
                                          sensing_end: datetime.datetime):
        '''
        Clip the series of coefficients from `product_annotation` that covers
        the burst whose sensing start / end time is provided in the arguments.

        Parameters:
        product_annotation: ProductAnnotation
            Data class from which the coeffieients will be extracted
        sensing_start: datetime.datetime
            Azimuth start time of the burst
        sensing_end: datetime.datetime
            Azimuth end time of the burst
        '''

        # Scan the azimuth time of fm rate
        id_t0_fm_rate, id_t1_fm_rate = cls._find_t0_t1(product_annotation.vec_aztime_fm_rate,
                                                       sensing_start,
                                                       sensing_end)

        # Scan the azimuth time of doppler centroid
        id_t0_dc, id_t1_dc = cls._find_t0_t1(product_annotation.vec_aztime_dc,
                                             sensing_start,
                                             sensing_end)

        vec_aztime_fm_rate_burst = (product_annotation
                                    .vec_aztime_fm_rate[id_t0_fm_rate: id_t1_fm_rate+1])
        lut_coeff_fm_rate_burst = (product_annotation
                                   .lut_coeff_fm_rate[id_t0_fm_rate:id_t1_fm_rate + 1, :])
        vec_tau0_fm_rate_burst = (product_annotation
                                  .vec_tau0_fm_rate[id_t0_fm_rate: id_t1_fm_rate + 1])

        vec_aztime_dc_burst = product_annotation.vec_aztime_dc[id_t0_dc: id_t1_dc + 1]
        lut_coeff_dc_burst = product_annotation.lut_coeff_dc[id_t0_dc:id_t1_dc + 1, :]
        vec_tau0_dc_burst = product_annotation.vec_tau0_dc[id_t0_dc: id_t1_dc + 1]

        return cls(vec_aztime_fm_rate_burst, lut_coeff_fm_rate_burst, vec_tau0_fm_rate_burst,
                   vec_aztime_dc_burst, lut_coeff_dc_burst, vec_tau0_dc_burst)


    @classmethod
    def _find_t0_t1(cls, vec_azimuth_time: np.ndarray,
                   datetime_start: datetime.datetime,
                   datetime_end: datetime.datetime):
        '''
        Scan `vec_azimuth_time` end find indices of the vector
        that covers the period defined with
        `datetime_start` and `datetime_end`

        Parameters:
        -----------
        vec_azimuth_time: np.ndarray
            numpy vector of azimuth time
        datetime_start: datetime.datetime
            Startime of the period
        datetime_end: datetime.datetime
            end time of the period

        Returns:
        --------
        (id_t0, id_t1): tuple(int)
            Indices of the azimuth times in `vec_azimuth_time` that
            covers the period
        '''

        # initial values
        id_t0_so_far = 0
        dt_t0_so_far = float('-inf')
        id_t1_so_far = len(vec_azimuth_time) - 1
        dt_t1_so_far = float('inf')

        # NOTE: dt is defined as: [azimuth time] - [start/end time]
        for id_vec, datetime_vec in enumerate(vec_azimuth_time):

            dt_t0 = (datetime_vec - datetime_start).total_seconds()
            if (dt_t0 < 0) and (dt_t0_so_far < dt_t0):
                id_t0_so_far = id_vec
                dt_t0_so_far = dt_t0
                continue

            dt_t1 = (datetime_vec - datetime_end).total_seconds()
            if (dt_t1 > 0) and (dt_t1_so_far > dt_t1):
                id_t1_so_far = id_vec
                dt_t1_so_far = dt_t1
                continue

        return (id_t0_so_far, id_t1_so_far)
