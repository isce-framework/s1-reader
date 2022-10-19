from dataclasses import dataclass
import datetime
import xml.etree.ElementTree as ET

import numpy as np
from packaging import version
from scipy.interpolate import InterpolatedUnivariateSpline

from s1reader.utils.utils import as_datetime, min_ipf_version_az_noise_vector

@dataclass
class BurstNoise: #For thermal noise correction
    '''Noise correction information for Sentinel-1 burst'''
    range_azimith_time: datetime.datetime
    range_line: float # TODO is this type correct?
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
class BurstNoiseLoader:
    ''' document me plz
    '''
    azimuth_first_azimuth_line: int
    azimuth_first_range_sample: int
    azimuth_last_azimuth_line: int
    azimuth_last_range_sample: int
    azimuth_line: np.ndarray
    azimuth_lut: np.ndarray

    noise_rg_vec_list: ET
    noise_rg_key: str

    @classmethod
    def from_file(cls, noise_path: str, ipf_version: version.Version,
                  open_method=open):
        ''' document me plz
        '''
        # TODO comments to explain WTF is going on
        with open_method(noise_path, 'r') as f_noise:
            noise_tree = ET.parse(f_noise)

        #legacy SAFE data
        if ipf_version < min_ipf_version_az_noise_vector:
            az_first_azimuth_line = None
            az_first_range_sample = None
            az_last_azimuth_line = None
            az_last_range_sample = None
            az_line = None
            az_lut = None

            noise_rg_vec_list = noise_tree.find('noiseVectorList')
            noise_rg_key = 'noiseLut'
        else:
            az_noise_tree = noise_tree.find('noiseAzimuthVector')
            az_first_azimuth_line = int(az_noise_tree.find('firstAzimuthLine').text)
            az_first_range_sample = int(az_noise_tree.find('firstRangeSample').text)
            az_last_azimuth_line = int(az_noise_tree.find('lastAzimuthLine').text)
            az_last_range_sample = int(az_noise_tree.find('lastRangeSample').text)
            az_line = np.array([int(x)
                                for x in az_noise_tree.find('line').text.split()])
            az_lut = np.array([float(x)
                               for x in az_noise_tree.find('noiseAzimuthLut').text.split()])

            noise_rg_vec_list = noise_tree.find('noiseRangeVectorList')
            noise_rg_key = 'noiseRangeLut'

        return cls(az_first_azimuth_line, az_first_range_sample,
                   az_last_azimuth_line, az_last_range_sample,
                   az_line, az_lut, noise_rg_vec_list, noise_rg_key)

    def get_nearest_noise(self, burst_az_time, line_from, line_to):
        ''' document me plz
        '''
        # find closest az time
        nearest_noise_rg_vec = None
        nearest_rg_az_time = None
        min_dt = 365 * 24 * 2600
        for noise_rg_vec in self.noise_rg_vec_list:
            rg_az_time = as_datetime(noise_rg_vec.find('azimuthTime').text)
            dt = abs(rg_az_time - burst_az_time)
            if dt < min_dt:
                nearest_noise_rg_vec = noise_rg_vec
                nearest_rg_az_time = rg_az_time
                min_dt = dt
                continue
            if dt > min_dt:
                break

        rg_line = int(nearest_noise_rg_vec.find('line').text)
        rg_pixels = np.array([int(x)
                              for x in nearest_noise_rg_vec.find('pixel').text.split()])

        rg_lut = np.array([float(x)
                           for x in nearest_noise_rg_vec.find(self.noise_rg_key).text.split()])

        return BurstNoise(nearest_rg_az_time, rg_line, rg_pixels, rg_lut,
                          self.azimuth_first_azimuth_line,
                          self.azimuth_first_range_sample,
                          self.azimuth_last_azimuth_line,
                          self.azimuth_last_range_sample,
                          self.azimuth_line, self.azimuth_lut,
                          line_from, line_to)
