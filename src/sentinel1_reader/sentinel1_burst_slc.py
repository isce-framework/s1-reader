from dataclasses import dataclass
from datetime import datetime
import datetime
import numpy as np

from osgeo import gdal

import isce3


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
    orbit_ref_epoch = datetime.strptime(orbit.reference_epoch.__str__()[:-3],
                                        fmt)

    t_mid = burst.get_sensing_mid() - orbit_ref_epoch
    _, v = orbit.interpolate(t_mid.total_seconds())
    vs = np.linalg.norm(v)
    ks = 2 * vs * burst.azimuth_steer_rate / burst.wavelength

    y, x = position

    n_lines, _ = burst.shape
    eta = (y - (n_lines // 2) + offset) * burst.azimuth_time_interval
    rng = burst.starting_range + x * burst.range_pxl_spacing

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

    val, res, rank, eigs = np.linalg.lstsq(A, z, rcond=cond)
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

    def get_az_carrier_poly(self, offset=0.0, xstep=500, ystep=50,
                            az_order=5, rg_order=3):
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

        Returns
        -------
        poly: isce3.core.Poly2D
           class represents a polynomial function of range
           'x' and azimuth 'y'
        """

        lines, samples = self.shape
        x = np.arange(0, samples, xstep, dtype=int)
        y = np.arange(0, lines, ystep, dtype=int)
        xx, yy = np.meshgrid(x, y)

        # Estimate azimuth carrier
        az_carrier = compute_az_carrier(self, self.orbit,
                                        offset=offset,
                                        position=(yy, xx))
        # Estimate azimuth carrier polynomials
        az_carrier_poly = polyfit(xx.flatten() + 1, yy.flatten() + 1,
                                  az_carrier.flatten(), az_order,
                                  rg_order)
        return az_carrier_poly
