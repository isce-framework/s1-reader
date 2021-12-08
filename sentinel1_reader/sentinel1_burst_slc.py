from dataclasses import dataclass
import datetime
import xml.etree.ElementTree as ET
import os

from osgeo import gdal

import isce3

@dataclass(frozen=True)
class Doppler:
    poly1d: isce3.core.Poly1d
    lut2d: isce3.core.LUT2d

@dataclass(frozen=True)
class Sentinel1BurstSlc:
    '''
    Raw values extracted from SAFE XML.
    '''
    sensing_start: datetime.datetime# *
    radar_center_frequency: float
    wavelength: float
    azimuth_steer_rate: float
    azimuth_time_interval: float
    slant_range_time: float
    starting_range: float
    range_sampling_rate: float
    range_pixel_spacing: float
    shape: tuple()
    azimuth_fm_rate: isce3.core.Poly1d
    doppler: Doppler
    range_bandwidth: float
    polarization: str # {VV, VH, HH}
    burst_id: str # t{track_number}_iw{1,2,3}_{burst_index}
    platform_id: str # S1{A,B}
    center: tuple # {center lon, center lat} in degrees
    border: list # list of lon, lat coordinate tuples (in degrees) representing burst border
    # VRT params
    tiff_path: str
    i_burst: int
    first_valid_sample: int
    last_valid_sample: int
    first_valid_line: int
    last_valid_line: int

    def as_isce3_radargrid(self):
        '''
        Init and return isce3.product.RadarGridParameters.

        Returns:
        --------
        _ : RadarGridParameters
            RadarGridParameters constructed from class members.
        '''

        prf = 1 / self.azimuth_time_interval

        length, width = self.shape

        time_delta = datetime.timedelta(days=2)
        ref_epoch = isce3.core.DateTime(self.sensing_start - time_delta)
        # sensing start with respect to reference epoch
        sensing_start = time_delta.total_seconds()

        # init radar grid
        return isce3.product.RadarGridParameters(sensing_start,
                                                 self.wavelength,
                                                 prf,
                                                 self.starting_range,
                                                 self.range_pixel_spacing,
                                                 isce3.core.LookSide.Right,
                                                 length,
                                                 width,
                                                 ref_epoch)

    def to_vrt_file(self, out_path):
        '''
        Write burst to VRT file.

        Parameters:
        -----------
        out_path : string
            Path of output VRT file.
        '''
        line_offset = self.i_burst * self.shape[0]

        inwidth = self.last_valid_sample - self.first_valid_sample
        inlength = self.last_valid_line - self.first_valid_line
        outlength, outwidth = self.shape
        yoffset = line_offset + self.first_valid_line
        localyoffset = self.first_valid_line
        xoffset = self.first_valid_sample
        gdal_obj = gdal.Open(self.tiff_path, gdal.GA_ReadOnly)
        fullwidth = gdal_obj.RasterXSize
        fulllength = gdal_obj.RasterYSize

        # TODO maybe cleaner to write with ElementTree
        tmpl = f'''<VRTDataset rasterXSize="{outwidth}" rasterYSize="{outlength}">
    <VRTRasterBand dataType="CFloat32" band="1">
        <NoDataValue>0.0</NoDataValue>
        <SimpleSource>
            <SourceFilename relativeToVRT="1">{self.tiff_path}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{fullwidth}" RasterYSize="{fulllength}" DataType="CInt16"/>
            <SrcRect xOff="{xoffset}" yOff="{yoffset}" xSize="{inwidth}" ySize="{inlength}"/>
            <DstRect xOff="{xoffset}" yOff="{localyoffset}" xSize="{inwidth}" ySize="{inlength}"/>
        </SimpleSource>
    </VRTRasterBand>
</VRTDataset>'''

        with open(out_path, 'w') as fid:
            fid.write(tmpl)

    def get_sensing_mid(self):
        '''
        Returns sensing mid as datetime object.

        Returns:
        --------
        _ : datetime
            Sensing mid as datetime object.
        '''
        d_seconds = 0.5 * (self.shape[0] - 1) * self.azimuth_time_interval
        return self.sensing_start + datetime.timedelta(seconds=d_seconds)

    def get_isce3_orbit(self, orbit_dir: str):
        '''
        Init and return ISCE3 orbit.

        Parameters:
        -----------
        orbit_dir : string
            Path to directory containing orbit files.

        Returns:
        --------
        _ : datetime
            Sensing mid as datetime object.
        '''
        if not os.path.isdir(orbit_dir):
            raise NotADirectoryError

        # determine start and end time from metadata
        pulse_length = (self.shape[0] - 1) * self.azimuth_time_interval
        t_pulse_end = self.sensing_start + datetime.timedelta(seconds=pulse_length)

        # find files with self.platform_id
        item_valid = lambda item, sat_id: os.path.isfile(item) and sat_id in item
        orbit_files = [item for item in os.listdir(orbit_dir)
                       if item_valid(f'{orbit_dir}/{item}', self.platform_id)]
        if not orbit_files:
            err_str = f"No orbit files found for {self.platform_id} in f{orbit_dir}"
            raise RuntimeError(err_str)

        fmt = "%Y%m%dT%H%M%S"
        # parse start and end time of files
        for orbit_file in orbit_files:
            _, tail = os.path.split(orbit_file)
            t_orbit_start, t_orbit_end = tail.split('_')[-2:]
            t_orbit_start = datetime.datetime.strptime(t_orbit_start[1:], fmt)
            t_orbit_end = datetime.datetime.strptime(t_orbit_end[:-4], fmt)
            if t_orbit_start < self.sensing_start and t_orbit_end > t_pulse_end:
                break

        # find 'Data_Block/List_of_OSVs'
        tree = ET.parse(f'{orbit_dir}/{orbit_file}')
        osv_list = tree.find('Data_Block/List_of_OSVs')
        # TODO turn into generator?
        # loop thru elements
        # while OSV/UTC < burst_end
        #   UTC, pos, vel to list of isce3.core.stateVectors
        fmt = "UTC=%Y-%m-%dT%H:%M:%S.%f"
        orbit_sv = []
        # add start & end padding to ensure sufficient number of orbit points
        pad = datetime.timedelta(seconds=60)
        for osv in osv_list:
            t_orbit = datetime.datetime.strptime(osv[1].text, fmt)
            pos = [float(osv[i].text) for i in range(4,7)]
            vel = [float(osv[i].text) for i in range(7,10)]
            if t_orbit > self.sensing_start - pad:
                orbit_sv.append(isce3.core.StateVector(isce3.core.DateTime(t_orbit),
                                                       pos, vel))
            if t_orbit > t_pulse_end + pad:
                break

        # use list of stateVectors to init and return isce3.core.Orbit
        time_delta = datetime.timedelta(days=2)
        ref_epoch = isce3.core.DateTime(self.sensing_start - time_delta)
        return isce3.core.Orbit(orbit_sv, ref_epoch)
