import os
from dataclasses import dataclass
import datetime
import tempfile
import warnings

import isce3
import numpy as np
from osgeo import gdal


# Other functionalities
def compute_az_carrier(burst, orbit, offset, position):
    '''
    Estimate azimuth carrier and store in numpy arrary

    Parameters
    ----------
    burst: Sentinel1BurstSlc
       Sentinel1 burst object
    orbit: isce3.core.Orbit
       Sentinel1 orbit ephemerides
    offset: float
       Offset between reference and secondary burst
    position: tuple
       Tuple of locations along y and x directions

    Returns
    -------
    carr: np.ndarray
       Azimuth carrier
    '''

    # Get burst sensing mid relative to orbit reference epoch
    fmt = "%Y-%m-%dT%H:%M:%S.%f"
    orbit_ref_epoch = datetime.datetime.strptime(orbit.reference_epoch.__str__()[:-3], fmt)

    t_mid = burst.sensing_mid - orbit_ref_epoch
    _, v = orbit.interpolate(t_mid.total_seconds())
    vs = np.linalg.norm(v)
    ks = 2 * vs * burst.azimuth_steer_rate / burst.wavelength

    y, x = position

    n_lines, _ = burst.shape
    eta = (y - (n_lines // 2) + offset) * burst.azimuth_time_interval
    rng = burst.starting_range + x * burst.range_pixel_spacing

    f_etac = np.array(
        burst.doppler.poly1d.eval(rng.flatten().tolist())).reshape(rng.shape)
    ka = np.array(burst.azimuth_fm_rate.eval(rng.flatten().tolist())).reshape(
        rng.shape)

    eta_ref = (burst.doppler.poly1d.eval(
        burst.starting_range) / burst.azimuth_fm_rate.eval(
        burst.starting_range)) - (f_etac / ka)
    kt = ks / (1.0 - ks / ka)

    carr = np.pi * kt * ((eta - eta_ref) ** 2)

    return carr


def polyfit(xin, yin, zin, azimuth_order, range_order,
            sig=None, snr=None, cond=1.0e-12,
            max_order=True):
    """
    Fit 2-D polynomial
    Parameters:
    xin: np.ndarray
       Array locations along x direction
    yin: np.ndarray
       Array locations along y direction
    zin: np.ndarray
       Array locations along z direction
    azimuth_order: int
       Azimuth polynomial order
    range_order: int
       Slant range polynomial order
    sig: -
       ---------------------------
    snr: float
       Signal to noise ratio
    cond: float
       ---------------------------
    max_order: bool
       ---------------------------

    Returns:
    poly: isce3.core.Poly2D
       class represents a polynomial function of range
       'x' and azimuth 'y'
    """
    x = np.array(xin)
    xmin = np.min(x)
    xnorm = np.max(x) - xmin
    if xnorm == 0:
        xnorm = 1.0
    x = (x - xmin) / xnorm

    y = np.array(yin)
    ymin = np.min(y)
    ynorm = np.max(y) - ymin
    if ynorm == 0:
        ynorm = 1.0
    y = (y - ymin) / ynorm

    z = np.array(zin)
    big_order = max(azimuth_order, range_order)

    arr_list = []
    for ii in range(azimuth_order + 1):
        yfact = np.power(y, ii)
        for jj in range(range_order + 1):
            xfact = np.power(x, jj) * yfact
            if max_order:
                if ((ii + jj) <= big_order):
                    arr_list.append(xfact.reshape((x.size, 1)))
            else:
                arr_list.append(xfact.reshape((x.size, 1)))

    A = np.hstack(arr_list)
    if sig is not None and snr is not None:
        raise Exception('Only one of sig / snr can be provided')
    if sig is not None:
        snr = 1.0 + 1.0 / sig
    if snr is not None:
        A = A / snr[:, None]
        z = z / snr

    return_val = True
    val, res, _, eigs = np.linalg.lstsq(A, z, rcond=cond)
    if len(res) > 0:
        print('Chi squared: %f' % (np.sqrt(res / (1.0 * len(z)))))
    else:
        print('No chi squared value....')
        print('Try reducing rank of polynomial.')
        return_val = False

    coeffs = []
    count = 0
    for ii in range(azimuth_order + 1):
        row = []
        for jj in range(range_order + 1):
            if max_order:
                if (ii + jj) <= big_order:
                    row.append(val[count])
                    count = count + 1
                else:
                    row.append(0.0)
            else:
                row.append(val[count])
                count = count + 1
        coeffs.append(row)
    poly = isce3.core.Poly2d(coeffs, xmin, ymin, xnorm, ynorm)
    return poly


@dataclass(frozen=True)
class Doppler:
    poly1d: isce3.core.Poly1d
    lut2d: isce3.core.LUT2d

@dataclass(frozen=True)
class Sentinel1BurstSlc:
    '''Raw values extracted from SAFE XML.
    '''
    sensing_start: datetime.datetime
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
    burst_id: str # t{track_number}_iw{1,2,3}_b{burst_index}
    platform_id: str # S1{A,B}
    center: tuple # {center lon, center lat} in degrees
    border: list # list of lon, lat coordinate tuples (in degrees) representing burst border
    orbit: isce3.core.Orbit
    orbit_direction: str
    # VRT params
    tiff_path: str  # path to measurement tiff in SAFE/zip
    i_burst: int
    first_valid_sample: int
    last_valid_sample: int
    first_valid_line: int
    last_valid_line: int
    # window parameters
    range_window_type: str
    range_window_coefficient: float


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

    def slc_to_file(self, out_path: str, fmt: str = 'ENVI'):
        '''Write burst to GTiff file.

        Parameters:
        -----------
        out_path : string
            Path of output GTiff file.
        '''
        if not self.tiff_path:
            warn_str = f'Unable write SLC to file. Burst does not contain image data; only metadata.'
            warnings.warn(warn_str)
            return

        # get output directory of out_path
        dst_dir, _ = os.path.split(out_path)

        # create VRT; make temporary if output not VRT
        if fmt != 'VRT':
            temp_vrt = tempfile.NamedTemporaryFile(dir=dst_dir)
            vrt_fname = temp_vrt.name
        else:
            vrt_fname = out_path
        self.slc_to_vrt_file(vrt_fname)

        if fmt == 'VRT':
            return

        # open temporary VRT and translate to GTiff
        src_ds = gdal.Open(vrt_fname)
        gdal.Translate(out_path, src_ds, format=fmt)

        # clean up
        src_ds = None


    def slc_to_vrt_file(self, out_path):
        '''Write burst to VRT file.

        Parameters:
        -----------
        out_path : string
            Path of output VRT file.
        '''
        if not self.tiff_path:
            warn_str = f'Unable write SLC to file. Burst does not contain image data; only metadata.'
            warnings.warn(warn_str)
            return

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

    def get_az_carrier_poly(self, offset=0.0, xstep=500, ystep=50,
                            az_order=5, rg_order=3, index_as_coord=False):
        """
        Estimate burst azimuth carrier polymonials
        Parameters
        ----------
        offset: float
            Offset between reference and secondary bursts
        xstep: int
            Spacing along x direction
        ystep: int
            Spacing along y direction
        az_order: int
            Azimuth polynomial order
        rg_order: int
            Slant range polynomial order
        index_as_coord: bool
            If true, polyfit with az/range indices. Else, polyfit with az/range.

        Returns
        -------
        poly: isce3.core.Poly2D
           class represents a polynomial function of range
           'x' and azimuth 'y'
        """

        rdr_grid = self.as_isce3_radargrid()

        lines, samples = self.shape
        x = np.arange(0, samples, xstep, dtype=int)
        y = np.arange(0, lines, ystep, dtype=int)
        x_mesh, y_mesh = np.meshgrid(x, y)

        # Estimate azimuth carrier
        az_carrier = compute_az_carrier(self, self.orbit,
                                        offset=offset,
                                        position=(y_mesh, x_mesh))

        # Fit azimuth carrier polynomial with x/y or range/azimuth
        if index_as_coord:
            az_carrier_poly = polyfit(x_mesh.flatten()+1, y_mesh.flatten()+1,
                                      az_carrier.flatten(), az_order,
                                      rg_order)
        else:
            # Convert x/y to range/azimuth
            rg = self.starting_range + (x + 1) * self.range_pixel_spacing
            az = rdr_grid.sensing_start + (y + 1) * self.azimuth_time_interval
            rg_mesh, az_mesh = np.meshgrid(rg, az)

            # Estimate azimuth carrier polynomials
            az_carrier_poly = polyfit(rg_mesh.flatten(), az_mesh.flatten(),
                                  az_carrier.flatten(), az_order,
                                  rg_order)

        return az_carrier_poly

    def as_dict(self):
        """
        Return SLC class attributes as dict

        Returns
        -------
        self_as_dict: dict
           Dict representation as a dict
        """
        self_as_dict = {}
        for key, val in self.__dict__.items():
            if key == 'sensing_start':
                val = str(val)
            elif key == 'center':
                val = val.coords[0]
            elif isinstance(val, np.float64):
                val = float(val)
            elif key == 'azimuth_fm_rate':
                temp = {}
                temp['order'] = val.order
                temp['mean'] = val.mean
                temp['std'] = val.std
                temp['coeffs'] = val.coeffs
                val = temp
            elif key == 'border':
                val = self.border[0].wkt
            elif key == 'doppler':
                temp = {}

                temp['poly1d'] = {}
                temp['poly1d']['order'] = val.poly1d.order
                temp['poly1d']['mean'] = val.poly1d.mean
                temp['poly1d']['std'] = val.poly1d.std
                temp['poly1d']['coeffs'] = val.poly1d.coeffs

                temp['lut2d'] = {}
                temp['lut2d']['x_start'] = val.lut2d.x_start
                temp['lut2d']['x_spacing'] = val.lut2d.x_spacing
                temp['lut2d']['y_start'] = val.lut2d.y_start
                temp['lut2d']['y_spacing'] = val.lut2d.y_spacing
                temp['lut2d']['length'] = val.lut2d.length
                temp['lut2d']['width'] = val.lut2d.width
                temp['lut2d']['data'] = val.lut2d.data.flatten().tolist()

                val = temp
            elif key in ['orbit']:
                temp = {}
                temp['ref_epoch'] = str(val.reference_epoch)
                temp['time'] = {}
                temp['time']['first'] = val.time.first
                temp['time']['spacing'] = val.time.spacing
                temp['time']['last'] = val.time.last
                temp['time']['size'] = val.time.size
                temp['position_x'] = val.position[:,0].tolist()
                temp['position_y'] = val.position[:,1].tolist()
                temp['position_z'] = val.position[:,2].tolist()
                temp['velocity_x'] = val.velocity[:,0].tolist()
                temp['velocity_y'] = val.velocity[:,1].tolist()
                temp['velocity_z'] = val.velocity[:,2].tolist()
                val = temp
            self_as_dict[key] = val
        return self_as_dict

    @property
    def sensing_mid(self):
        '''Returns sensing mid as datetime.datetime object.

        Returns:
        --------
        _ : datetime.datetime
            Sensing mid as datetime.datetime object.
        '''
        d_seconds = 0.5 * self.length * self.azimuth_time_interval
        return self.sensing_start + datetime.timedelta(seconds=d_seconds)

    @property
    def sensing_stop(self):
        '''Returns sensing end as datetime.datetime object.

        Returns:
        --------
        _ : datetime.datetime
            Sensing end as datetime.datetime object.
        '''
        d_seconds = (self.length - 1) * self.azimuth_time_interval
        return self.sensing_start + datetime.timedelta(seconds=d_seconds)

    @property
    def length(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]
