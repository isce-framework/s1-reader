from dataclasses import dataclass
import datetime

from osgeo import gdal

import isce3

@dataclass(frozen=True)
class Doppler:
    poly1d: isce3.core.Poly1d
    lut2d: isce3.core.LUT2d

@dataclass(frozen=True)
class Sentinel1BurstSlc:
    '''Raw values extracted from SAFE XML.
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
    polarization: str # {VV, VH, HH, HV}
    burst_id: str # t{track_number}_iw{1,2,3}_{burst_index}
    platform_id: str # S1{A,B}
    center: tuple # {center lon, center lat} in degrees
    border: list # list of lon, lat coordinate tuples (in degrees) representing burst border
    orbit: isce3.core.Orbit
    # VRT params
    tiff_path: str
    i_burst: int
    first_valid_sample: int
    last_valid_sample: int
    first_valid_line: int
    last_valid_line: int

    def as_isce3_radargrid(self):
        '''Init and return isce3.product.RadarGridParameters.

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
        '''Write burst to VRT file.

        Parameters:
        -----------
        out_path : string
            Path of output VRT file.
        '''
        line_offset = self.i_burst * self.shape[0]

        inwidth = self.last_valid_sample - self.first_valid_sample + 1
        inlength = self.last_valid_line - self.first_valid_line + 1
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
        '''Returns sensing mid as datetime object.

        Returns:
        --------
        _ : datetime
            Sensing mid as datetime object.
        '''
        d_seconds = 0.5 * (self.shape[0] - 1) * self.azimuth_time_interval
        return self.sensing_start + datetime.timedelta(seconds=d_seconds)
