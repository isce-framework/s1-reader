'''
A module to load annotation files for Sentinel-1 IW SLC SAFE data
To be used for the class "Sentinel1BurstSlc"
'''

from dataclasses import dataclass
import datetime
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

@dataclass
class AnnotationBase:
    '''
    A base class of the inheriting annotation class i.e. Product, Calibration, and Noise.
    '''
    # A parent class for annotation reader for Calibrarion, Noise, and Product
    xml_et: ET
    #kind:str

    @classmethod
    def _parse_scalar(cls,path_field:str, str_type:str):
        '''Parse the scalar value in the annotation'''
        elem_field=cls.xml_et.find(path_field)
        if str_type=='datetime':
            val_out=datetime.datetime.strptime(elem_field.text,'%Y-%m-%dT%H:%M:%S.%f')

        elif str_type=='scalar_int':
            val_out=int(elem_field.text)

        elif str_type=='scalar_float':
            val_out=float(elem_field.text)

        elif str_type=='vector_int':
            val_out=np.array([int(strin) for strin in elem_field.text.split()])

        elif str_type=='vector_float':
            val_out=np.array([float(strin) for strin in elem_field.text.split()])

        elif str_type=='str':
            val_out=elem_field.text

        else:
            raise ValueError(f'Cannot recognize the type of the element: "{str_type}"')

        return val_out

    @classmethod
    def _parse_vectorlist(cls, name_vector_list:str, name_vector:str, str_type:str):
        '''Parse the list of the values in the annotation'''
        #NOTE: str type: ['datetime','scalar_integer','scalar_float','vector_integer','vector_float','str']

        element_to_parse=cls.xml_et.find(name_vector_list)
        num_element=len(element_to_parse)

        list_out=[None]*num_element

        if str_type=='datetime':
            for i,elem in enumerate(element_to_parse):
                str_elem=elem.find(name_vector).text
                list_out[i]=datetime.datetime.strptime(str_elem,'%Y-%m-%dT%H:%M:%S.%f')
            list_out=np.array(list_out)

        elif str_type=='scalar_int':
            for i,elem in enumerate(element_to_parse):
                str_elem=elem.find(name_vector).text
                list_out[i]=int(str_elem)

        elif str_type=='scalar_float':
            for i,elem in enumerate(element_to_parse):
                str_elem=elem.find(name_vector).text
                list_out[i]=float(str_elem)

        elif str_type=='vector_int':
            for i,elem in enumerate(element_to_parse):
                str_elem=elem.find(name_vector).text
                list_out[i]=np.array([int(strin) for strin in str_elem.split()])

        elif str_type=='vector_float':
            for i,elem in enumerate(element_to_parse):
                str_elem=elem.find(name_vector).text
                list_out[i]=np.array([float(strin) for strin in str_elem.split()])

        elif str_type=='str':
            list_out=element_to_parse[0].find(name_vector).text

        else:
            raise ValueError(f'Cannot recognize the type of the element: {str_type}')

        return list_out


@dataclass
class CalibrationAnnotation(AnnotationBase):
    '''Reader for Calibration Annotation Data Set (CADS)'''
    list_azimuth_time: np.ndarray
    list_line:list
    list_pixel:None
    list_sigma_nought:list
    list_beta_nought:list
    list_gamma:list
    list_dn:list

    @classmethod
    def from_et(cls,et_in=None):
        '''Extracts list of calibration from etree'''
        cls.xml_et=et_in
        cls.list_azimuth_time=cls._parse_vectorlist('calibrationVectorList','azimuthTime','datetime')
        cls.list_line=cls._parse_vectorlist('calibrationVectorList','line','scalar_int')
        cls.list_pixel=cls._parse_vectorlist('calibrationVectorList','pixel','vector_int')
        cls.list_sigma_nought=cls._parse_vectorlist('calibrationVectorList','sigmaNought','vector_float')
        cls.list_beta_nought=cls._parse_vectorlist('calibrationVectorList','betaNought','vector_float')
        cls.list_gamma=cls._parse_vectorlist('calibrationVectorList','gamma','vector_float')
        cls.list_dn=cls._parse_vectorlist('calibrationVectorList','dn','vector_float')

        return cls


@dataclass
class NoiseAnnotation(AnnotationBase):
    '''Reader for Noise Annotation Data Set (NADS)'''
    # TODO Schema of the NADS is slightly different before/after IPF version 2.90. Needs to be adaptive in accordance with the version.
    #in ISCE2 code: if float(self.IPFversion) < 2.90:
    # REF: .../isce2/components/isceobj/Sensor/GRD/Sentinel1.py

    rg_list_azimuth_time:np.ndarray
    rg_list_line:list
    rg_list_pixel:list
    rg_list_noise_range_lut:list
    az_first_azimuth_line:int
    az_first_range_sample:int
    az_last_azimuth_line:int
    az_last_range_sample:int
    az_line:np.ndarray
    az_noise_azimuth_lut:np.ndarray

    @classmethod
    def from_et(cls,et_in:ET,et_in_lads:ET=None,ipf_version=3.10):
        '''Extracts list of noise information from etree'''
        if et_in is not None:
            cls.xml_et=et_in

        if ipf_version<2.90: #legacy SAFE data
            cls.rg_list_azimuth_time=cls._parse_vectorlist('noiseVectorList','azimuthTime','datetime')
            cls.rg_list_line=cls._parse_vectorlist('noiseVectorList','line','scalar_int')
            cls.rg_list_pixel=cls._parse_vectorlist('noiseVectorList','pixel','vector_int')
            cls.rg_list_noise_range_lut=cls._parse_vectorlist('noiseVectorList','noiseLut','vector_float')
            cls.az_first_azimuth_line=0
            cls.az_first_range_sample=0
            cls.az_last_azimuth_line=None
            cls.az_last_range_sample=int(et_in_lads.find('imageAnnotation/imageInformation/numberOfSamples').text)-1
            cls.az_line=None
            cls.az_noise_azimuth_lut=None

        else:
            cls.rg_list_azimuth_time=cls._parse_vectorlist('noiseRangeVectorList','azimuthTime','datetime')
            cls.rg_list_line=cls._parse_vectorlist('noiseRangeVectorList','line','scalar_int')
            cls.rg_list_pixel=cls._parse_vectorlist('noiseRangeVectorList','pixel','vector_int')
            cls.rg_list_noise_range_lut=cls._parse_vectorlist('noiseRangeVectorList','noiseRangeLut','vector_float')
            cls.az_first_azimuth_line=cls._parse_vectorlist('noiseAzimuthVectorList','firstAzimuthLine','scalar_int')[0]
            cls.az_first_range_sample=cls._parse_vectorlist('noiseAzimuthVectorList','firstRangeSample','scalar_int')[0]
            cls.az_last_azimuth_line=cls._parse_vectorlist('noiseAzimuthVectorList','lastAzimuthLine','scalar_int')[0]
            cls.az_last_range_sample=cls._parse_vectorlist('noiseAzimuthVectorList','lastRangeSample','scalar_int')[0]
            cls.az_line=cls._parse_vectorlist('noiseAzimuthVectorList','line','vector_int')[0]
            cls.az_noise_azimuth_lut=cls._parse_vectorlist('noiseAzimuthVectorList','noiseAzimuthLut','vector_float')[0]

        return cls


@dataclass
class ProductAnnotation(AnnotationBase):
    '''For L1 SLC product annotation file. For EAP correction.'''
    image_information_slant_range_time:float
    #elevation_angle:
    antenna_pattern_azimuth_time:list
    antenna_pattern_slant_range_time:list
    antenna_pattern_elevation_angle:list
    antenna_pattern_elevation_pattern:list

    ascending_node_time:datetime
    number_of_samples:int
    range_sampling_rate:float

    @classmethod
    def from_et(cls,et_in):
        '''Extracts list of noise information from etree'''
        if et_in is not None:
            cls.xml_et=et_in
        cls.image_information_slant_range_time=cls._parse_scalar('imageAnnotation/imageInformation/slantRangeTime','scalar_float')
        cls.antenna_pattern_azimuth_time=cls._parse_vectorlist('antennaPattern/antennaPatternList','azimuthTime','datetime')
        cls.antenna_pattern_slant_range_time=cls._parse_vectorlist('antennaPattern/antennaPatternList','slantRangeTime','vector_float')
        cls.antenna_pattern_elevation_angle=cls._parse_vectorlist('antennaPattern/antennaPatternList','elevationAngle','vector_float')
        cls.antenna_pattern_elevation_pattern=cls._parse_vectorlist('antennaPattern/antennaPatternList','elevationPattern','vector_float')
        cls.ascending_node_time=cls._parse_scalar('imageAnnotation/imageInformation/ascendingNodeTime','datetime')
        cls.number_of_samples=cls._parse_scalar('imageAnnotation/imageInformation/numberOfSamples','scalar_int')
        cls.number_of_samples=cls._parse_scalar('imageAnnotation/imageInformation/numberOfSamples','scalar_int')
        cls.range_sampling_rate=cls._parse_scalar('generalAnnotation/productInformation/rangeSamplingRate','scalar_float')

        return cls


@dataclass
class AuxCal(AnnotationBase):
    '''AUX_CAL'''
    beam_nominal_near_range:float
    beam_nominal_far_range:float
    elevation_angle_increment:float
    elevation_antenna_pattern:np.ndarray
    azimuth_angle_increment:float
    azimuth_antenna_pattern:np.ndarray
    azimuth_antenna_element_pattern_increment:float
    azimuth_antenna_element_pattern:float
    absolute_calibration_constant:float
    noise_calibration_factor:float

    @classmethod
    def from_et(cls,et_in:ET, pol:str, str_swath:str):
        '''Extracts list of information AUX_CAL from its etree'''
        calibration_params_list=et_in.find('calibrationParamsList')
        for calibration_params in calibration_params_list:
            swath_xml=calibration_params.find('swath').text
            polarisation_xml=calibration_params.find('polarisation').text
            if polarisation_xml==pol.upper() and swath_xml==str_swath.upper():
                print(f'Found a calibration parameters for swath {str_swath} and polarization {pol}.')
                cls.beam_nominal_near_range=float(calibration_params.find('elevationAntennaPattern/beamNominalNearRange').text)
                cls.beam_nominal_far_range=float(calibration_params.find('elevationAntennaPattern/beamNominalFarRange').text)
                cls.elevation_angle_increment=float(calibration_params.find('elevationAntennaPattern/elevationAngleIncrement').text)

                n_val=int(calibration_params.find('elevationAntennaPattern/values').attrib['count'])
                arr_eap_val=np.array([float(token_val) for token_val in calibration_params.find('elevationAntennaPattern/values').text.split()])
                if n_val==len(arr_eap_val): #Provided in real numbers: In case of AUX_CAL for old IPFs.
                    cls.azimuth_antenna_element_pattern=arr_eap_val
                elif n_val*2==len(arr_eap_val): #Provided in complex numbers: In case of recent IPFs e.g. 3.10
                    cls.azimuth_antenna_element_pattern=arr_eap_val[0::2]+arr_eap_val[1::2]*1.0j
                else:
                    raise ValueError(f'The number of values does not match. n_val={n_val}, #values in elevationAntennaPattern/values={len(arr_eap_val)}')

                cls.azimuth_angle_increment=float(calibration_params.find('azimuthAntennaPattern/azimuthAngleIncrement').text)
                cls.azimuth_antenna_pattern=np.array([float(token_val) for token_val in calibration_params.find('azimuthAntennaPattern/values').text.split()])
                cls.absolute_calibration_constant=float(calibration_params.find('absoluteCalibrationConstant').text)
                cls.noise_calibration_factor=float(calibration_params.find('noiseCalibrationFactor').text)

        return cls



def is_eap_correction_necesasry(ipf_version:float) -> int :
    '''Examines if what level of EAP correction is necessary, based on the IPF version
    0: No EAP correction necessary (i.e. correction already applied)
    1: Phase-only correction is necessary
    2: Phase and Magniture correction is necessary'''
    ipf_ver_int=int(ipf_version*100)
    #Based on ESA technical document
    if ipf_ver_int>=243:
        return 0 # No EAP correction necessary (i.e. correction already applied)
    elif ipf_ver_int>=236:
        return 1 # Phase-only correction is necessary
    else:
        return 2 # Phase and Magniture correction is necessary



def closest_block_to_azimuth_time(vector_azimuth_time:np.ndarray, azmuth_time_burst:datetime.datetime) -> int:
    '''Find the id of the closest data block in annotation.'''

    return np.argmin(np.abs(vector_azimuth_time-azmuth_time_burst))


@dataclass
class BurstNoise: #For thermal noise correction
    '''Noise correction information for Sentinel-1 burst'''
    range_azimith_time: datetime.datetime = None
    range_line: float = None
    range_pixel: np.ndarray = None
    range_lut: np.ndarray = None
    azimuth_first_azimuth_line: int = None
    azimuth_first_range_sample: int = None
    azimuth_last_azimuth_line: int = None
    azimuth_last_range_sample: int = None
    azimuth_line: np.ndarray = None
    azimuth_lut: np.ndarray = None
    line_from:int=None
    line_to:int=None


    def from_noise_annotation(self, noise_annotation:NoiseAnnotation, azimuth_time:datetime, line_from:int, line_to:int, ipf_version:float=3.10):
        '''Extracts the noise correction info for the burst'''
        threshold_ipf_version=2.90 #IPF version that stared to provide azimuth noise vector
        id_closest=closest_block_to_azimuth_time(noise_annotation.rg_list_azimuth_time,azimuth_time)
        self.range_azimith_time=noise_annotation.rg_list_azimuth_time[id_closest]
        self.range_line=noise_annotation.rg_list_line[id_closest]
        self.range_pixel=noise_annotation.rg_list_pixel[id_closest]
        self.range_lut=noise_annotation.rg_list_noise_range_lut[id_closest]

        self.azimuth_first_azimuth_line=noise_annotation.az_first_azimuth_line
        self.azimuth_first_range_sample=noise_annotation.az_first_range_sample
        self.azimuth_last_azimuth_line=noise_annotation.az_last_azimuth_line
        self.azimuth_last_range_sample=noise_annotation.az_last_range_sample

        self.line_from=line_from
        self.line_to=line_to

        if ipf_version>=threshold_ipf_version:
            #Azinuth noise LUT exists - crop to the extent of the burst
            id_top=np.argmin(np.abs(noise_annotation.az_line-line_from))
            id_bottom=np.argmin(np.abs(noise_annotation.az_line-line_to))
            #put some margin when possible
            if id_top>0:
                id_top-=1
            if id_bottom<len(noise_annotation.az_line)-1:
                id_bottom+=1
            self.azimuth_line=noise_annotation.az_line[id_top:id_bottom]
            self.azimuth_lut=noise_annotation.az_noise_azimuth_lut[id_top:id_bottom]
        



        #return cls


    def export_lut(self):
        '''Gives out the LUT table whose size is the same as the burst SLC'''
        ncols=self.azimuth_last_range_sample-self.azimuth_first_range_sample+1
        nrows=self.line_to-self.line_from+1

        #interpolator for range noise vector
        intp_rg_lut=InterpolatedUnivariateSpline(self.range_pixel,self.range_lut,k=1)
        grid_rg=np.arange(self.azimuth_last_range_sample+1)
        rg_lut_interp=intp_rg_lut(grid_rg).reshape((1,ncols))

        #interpolator for azimuth noise vector - take IPF version into consideration
        if (self.azimuth_line is None) or (self.azimuth_lut is None): # IPF <2.90
            az_lut_interp=np.ones(nrows).reshape((nrows,1))

        else: #IPF >= 2.90
            intp_az_lut=InterpolatedUnivariateSpline(self.azimuth_line,self.azimuth_lut,k=1)
            grid_az=np.arange(self.line_from,self.line_to+1)
            az_lut_interp=intp_az_lut(grid_az).reshape((nrows,1))
        
        arr_lut_total=np.matmul(az_lut_interp,rg_lut_interp)
        
        return arr_lut_total


@dataclass
class BurstCalibration:
    '''Calibration information for Sentinel-1 burst
    '''
    azimith_time: datetime.datetime = None
    line: float = None
    pixel: np.ndarray = None
    sigma_naught: np.ndarray = None
    beta_naught: np.ndarray = None
    gamma: np.ndarray = None
    dn: np.ndarray = None

    def from_calibraiton_annotation(self, calibration_annotation:CalibrationAnnotation, azimuth_time:datetime):
        '''Extracts the calibration info for the burst'''
        id_closest=closest_block_to_azimuth_time(calibration_annotation.list_azimuth_time, azimuth_time)
        self.azimuth_time=calibration_annotation.list_azimuth_time[id_closest]
        self.line=calibration_annotation.list_line[id_closest]
        self.pixel=calibration_annotation.list_pixel[id_closest]
        self.sigma_naught=calibration_annotation.list_sigma_nought[id_closest]

        matrix_beta_naught=np.array(calibration_annotation.list_beta_nought)
        if matrix_beta_naught.min()==matrix_beta_naught.max(): #NOTE It might not be a good idea to attempt '==' operation on the floating point data.
            self.beta_naught=np.min(matrix_beta_naught)
        else:
            #TODO Switch to LUT-based method when there is significant changes in the array
            self.beta_naught=np.mean(matrix_beta_naught)


@dataclass
class BurstEAP:
    '''EAP correction information for Sentinel-1 burst
       Currently Under development
    '''
    #from LADS
    Ns:int #number of samples
    fs:float #range sampling rate
    eta_start: datetime
    tau_0: float #imageInformation/slantRangeTime
    tau_sub: np.ndarray #antennaPattern/slantRangeTime
    theta_sub: np.ndarray #antennaPattern/elevationAngle
    azimuth_time: datetime
    ascending_node_time: datetime

    #from AUX_CAL
    G_eap: np.ndarray #elevationAntennaPattern
    delta_theta:float #elavationAngleIncrement

