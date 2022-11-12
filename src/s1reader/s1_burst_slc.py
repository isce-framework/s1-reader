import os
from dataclasses import dataclass
import datetime
import tempfile
import warnings
from packaging import version

import isce3
import numpy as np
from osgeo import gdal

from scipy.interpolate import InterpolatedUnivariateSpline
from s1reader import s1_annotation


# Other functionalities
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

@dataclass
class AzimuthCarrierComponents:
    kt: np.ndarray
    eta: float
    eta_ref: float

    @property
    def antenna_steering_doppler(self):
        return self.kt * (self.eta - self.eta_ref)

    @property
    def carrier(self):
        return np.pi * self.kt * ((self.eta - self.eta_ref) ** 2)

@dataclass(frozen=True)
class Doppler:
    poly1d: isce3.core.Poly1d
    lut2d: isce3.core.LUT2d

@dataclass(frozen=True)
class Sentinel1BurstSlc:
    '''Raw values extracted from SAFE XML.
    '''
    #ipf_version:float
    ipf_version: version.Version
    sensing_start: datetime.datetime
    radar_center_frequency: float
    wavelength: float
    azimuth_steer_rate: float
    azimuth_time_interval: float
    slant_range_time: float
    starting_range: float
    iw2_mid_range: float
    range_sampling_rate: float
    range_pixel_spacing: float
    shape: tuple()
    azimuth_fm_rate: isce3.core.Poly1d
    doppler: Doppler
    range_bandwidth: float
    polarization: str # {VV, VH, HH, HV}
    burst_id: str # t{track_number}_{burst_index}_iw{1,2,3}
    platform_id: str # S1{A,B}
    safe_filename: str # SAFE file name
    center: tuple # {center lon, center lat} in degrees
    border: list # list of lon, lat coordinate tuples (in degrees) representing burst border
    orbit: isce3.core.Orbit
    orbit_direction: str
    abs_orbit_number: int  # Absolute orbit number
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
    rank: int # The number of PRI between transmitted pulse and return echo.
    prf_raw_data: float  # Pulse repetition frequency (PRF) of the raw data [Hz]
    range_chirp_rate: float # Range chirp rate [Hz]

    # Correction information
    burst_calibration: s1_annotation.BurstCalibration  # Radiometric correction
    burst_noise: s1_annotation.BurstNoise  # Thermal noise correction
    burst_eap: s1_annotation.BurstEAP  # EAP correction

    # Extended FM rate / Doppler centroid polynomial coefficients
    # for azimith FM rate mismatch mitigation
    extended_coeffs_fm_dc: s1_annotation.BurstExtendedCoeffs

    def __str__(self):
        return f"Sentinel1BurstSlc: {self.burst_id} at {self.sensing_start}"

    def __repr__(self):
        return f"{self.__class__.__name__}(burst_id={self.burst_id})"

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
        az_carr_comp = self.az_carrier_components(
                                        offset=offset,
                                        position=(y_mesh, x_mesh))

        # Fit azimuth carrier polynomial with x/y or range/azimuth
        if index_as_coord:
            az_carrier_poly = polyfit(x_mesh.flatten()+1, y_mesh.flatten()+1,
                                      az_carr_comp.carrier.flatten(), az_order,
                                      rg_order)
        else:
            # Convert x/y to range/azimuth
            rg = self.starting_range + (x + 1) * self.range_pixel_spacing
            az = rdr_grid.sensing_start + (y + 1) * self.azimuth_time_interval
            rg_mesh, az_mesh = np.meshgrid(rg, az)

            # Estimate azimuth carrier polynomials
            az_carrier_poly = polyfit(rg_mesh.flatten(), az_mesh.flatten(),
                                  az_carr_comp.carrier.flatten(), az_order,
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
            elif key == 'orbit':
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

    def bistatic_delay(self, xstep=1, ystep=1):
        '''Computes the bistatic delay correction in azimuth direction
        due to the movement of the platform between pulse transmission and echo reception
        as described in equation (21) in Gisinger et al. (2021, TGRS).

        References
        -------
        Gisinger, C., Schubert, A., Breit, H., Garthwaite, M., Balss, U., Willberg, M., et al.
          (2021). In-Depth Verification of Sentinel-1 and TerraSAR-X Geolocation Accuracy Using
          the Australian Corner Reflector Array. IEEE Trans. Geosci. Remote Sens., 59(2), 1154-
          1181. doi:10.1109/TGRS.2019.2961248
        ETAD-DLR-DD-0008, Algorithm Technical Baseline Document. Available: https://sentinels.
          copernicus.eu/documents/247904/4629150/Sentinel-1-ETAD-Algorithm-Technical-Baseline-
          Document.pdf

        Parameters
        -------
        xstep : int
           spacing along x direction (range direction) in units of pixels

        ystep : int
           spacing along y direction (azimuth direction) in units of pixels

        Returns
        -------
           LUT2D object of bistatic delay correction in seconds as a function
           of the range and zimuth indices. This correction needs to be added
           to the SLC tagged azimuth time to get the corrected azimuth times.
        '''

        pri = 1.0 / self.prf_raw_data
        tau0 = self.rank * pri
        tau_mid = self.iw2_mid_range * 2.0 / isce3.core.speed_of_light

        nx = np.ceil(self.width / xstep).astype(int)
        ny = np.ceil(self.length / ystep).astype(int)
        x = np.arange(0, nx*xstep, xstep, dtype=int)
        y = np.arange(0, ny*ystep, ystep, dtype=int)

        slant_range = self.starting_range + x * self.range_pixel_spacing
        tau = slant_range * 2.0 / isce3.core.speed_of_light

        # the first term (tau_mid/2) is the bulk bistatic delay which was
        # removed from the orginial azimuth time by the ESA IPF. Based on
        # Gisinger et al. (2021) and ETAD ATBD, ESA IPF has used the mid of
        # the second subswath to compute the bulk bistatic delay. However
        # currently we have not been able to verify this from ESA documents.
        # This implementation follows the Gisinger et al. (2021) for now, we
        # can revise when we hear back from ESA folks.
        bistatic_correction_vec = tau_mid / 2 + tau / 2 - tau0
        bistatic_correction = np.tile(bistatic_correction_vec.reshape(1,-1), (ny,1))

        return isce3.core.LUT2d(x, y, bistatic_correction)

    def geometrical_and_steering_doppler(self, xstep=500, ystep=50):
        """
        Compute total Doppler which is the sum of two components:
        (1) the geometrical Doppler induced by the relative movement
        of the sensor and target
        (2) the TOPS specicifc Doppler caused by the electric steering
        of the beam along the azimuth direction resulting in Doppler varying
        with azimuth time.
        Parameters
        ----------
        xstep: int
            Spacing along x direction [pixels]
        ystep: int
            Spacing along y direction [pixels]

        Returns
        -------
        x : int
           The index of samples in range direction as an 1D array
        y : int
           The index of samples in azimuth direction as an 1D array
        total_doppler : float
           Total Doppler which is the sum of the geometrical Doppler and
           beam steering induced Doppler [Hz] as a 2D array
        """

        x = np.arange(0, self.width, xstep, dtype=int)
        y = np.arange(0, self.length, ystep, dtype=int)
        x_mesh, y_mesh = np.meshgrid(x, y)
        az_carr_comp = self.az_carrier_components(
                                        offset=0.0,
                                        position=(y_mesh, x_mesh))

        slant_range = self.starting_range + x * self.range_pixel_spacing
        geometrical_doppler = self.doppler.poly1d.eval(slant_range)

        total_doppler = az_carr_comp.antenna_steering_doppler + geometrical_doppler

        return x, y, total_doppler

    def doppler_induced_range_shift(self, xstep=500, ystep=50):
        """
        Computes the range delay caused by the Doppler shift as described
        by Gisinger et al 2021

        Parameters
        ----------
        xstep: int
            Spacing along x direction [pixels]
        ystep: int
            Spacing along y direction [pixels]

        Returns
        -------
        isce3.core.LUT2d:
           LUT2D object of range delay correction [seconds] as a function
           of the x and y indices.

        """

        x, y, doppler_shift = self.geometrical_and_steering_doppler(
                                                    xstep=xstep, ystep=ystep)
        tau_corr = doppler_shift / self.range_chirp_rate

        return isce3.core.LUT2d(x, y, tau_corr)

    def az_carrier_components(self, offset, position):
        '''
        Estimate azimuth carrier and store in numpy arrary. Also return
        contributing components.

        Parameters
        ----------
        offset: float
           Offset between reference and secondary burst
        position: tuple
           Tuple of locations along y and x directions

        Returns
        -------
        eta: float
            zero-Doppler azimuth time centered in the middle of the burst
        eta_ref: float
            refernce time
        kt: np.ndarray
            Doppler centroid rate in the focused TOPS SLC data [Hz/s]
        carr: np.ndarray
           Azimuth carrier

        Reference
        ---------
        https://sentinels.copernicus.eu/documents/247904/0/Sentinel-1-TOPS-SLC_Deramping/b041f20f-e820-46b7-a3ed-af36b8eb7fa0
        '''
        # Get self.sensing mid relative to orbit reference epoch
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        orbit_ref_epoch = datetime.datetime.strptime(self.orbit.reference_epoch.__str__()[:-3], fmt)

        t_mid = self.sensing_mid - orbit_ref_epoch
        _, v = self.orbit.interpolate(t_mid.total_seconds())
        vs = np.linalg.norm(v)
        ks = 2 * vs * self.azimuth_steer_rate / self.wavelength

        y, x = position

        n_lines, _ = self.shape
        eta = (y - (n_lines // 2) + offset) * self.azimuth_time_interval
        rng = self.starting_range + x * self.range_pixel_spacing

        f_etac = np.array(
            self.doppler.poly1d.eval(rng.flatten().tolist())).reshape(rng.shape)
        ka = np.array(
            self.azimuth_fm_rate.eval(rng.flatten().tolist())).reshape(rng.shape)

        eta_ref = (self.doppler.poly1d.eval(
            self.starting_range) / self.azimuth_fm_rate.eval(
            self.starting_range)) - (f_etac / ka)
        kt = ks / (1.0 - ks / ka)


        return AzimuthCarrierComponents(kt, eta, eta_ref)




    def _evaluate_polynomial_array(self, arr_polynomial, grid_tau, vec_tau0):
        '''
        Evaluate the polynomials on the correction grid
        To be used for azimuth FM mismatch rate 

        '''
        nrow = grid_tau.shape[0]
        ncol = grid_tau.shape[1]

        term_tau = grid_tau - vec_tau0 * np.ones(ncol)[np.newaxis, ...]

        eval_out = np.zeros((nrow, ncol))
        eval_out += arr_polynomial[:,0][...,np.newaxis] \
                *np.ones(ncol)[np.newaxis, ...]
        eval_out += (arr_polynomial[:,1][...,np.newaxis] \
                * np.ones(ncol)[np.newaxis, ...]) * term_tau
        eval_out += (arr_polynomial[:,2][...,np.newaxis] \
                * np.ones(ncol)[np.newaxis, ...]) * term_tau**2

        return eval_out


    def _latlon_to_ecef(self, lat, lon, alt, ellipsoid, unit_degree=True):
        '''
        Docstring here
        '''

        if unit_degree:
            rad_lat = lat * (np.pi / 180.0)
            rad_lon = lon * (np.pi / 180.0)
        else:
            rad_lat = lat
            rad_lon = lon


        #a = ellipsoid_in.a
        #finv = 298.257223563
        #f = 1 / finv
        #e2 = 1 - (1 - f) * (1 - f)
        v = ellipsoid.a / np.sqrt(1 - ellipsoid.e2 * np.sin(rad_lat) * np.sin(rad_lat))

        x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
        y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
        z = (v * (1 - ellipsoid.e2) + alt) * np.sin(rad_lat)

        return np.array([x, y, z])


    def az_fm_rate_mismatch_mitigation(self, path_dem: str, path_scratch: str,
                                       custom_radargrid_correction: isce3.product.RadarGridParameters=None):
        '''
        Calculate azimuth FM rate mismatch mitigation
        Based on ETAD-DLR-DD-0008, Algorithm Technical Baseline Document.
        Available: https://sentinels.copernicus.eu/documents/247904/4629150/
        Sentinel-1-ETAD-Algorithm-Technical-Baseline-Document.pdf

        Parameters:
        -----------
        path_dem: str
        '''
        
        os.makedirs(path_scratch, exist_ok=True)
        
        if custom_radargrid_correction is None:
            radargrid_correction = self.as_isce3_radargrid()
        else:
            radargrid_correction = custom_radargrid_correction

        # Generate vectors for t and tau for correction grid definition
        # either from radar grid or from the burst
        width_grid = radargrid_correction.width
        length_grid = radargrid_correction.length
        intv_t = 1/radargrid_correction.prf
        intv_tau = radargrid_correction.range_pixel_spacing * 2.0 / isce3.core.speed_of_light
        delta_sec_from_ref_epoch = (radargrid_correction.ref_epoch - self.orbit.reference_epoch).total_seconds() + radargrid_correction.sensing_start
        vec_t = np.arange(length_grid) * intv_t + delta_sec_from_ref_epoch
        vec_t_staggered = (np.arange(length_grid+1)
                           * intv_t
                           + delta_sec_from_ref_epoch
                           - intv_t / 2)

        vec_tau = np.arange(width_grid)*intv_tau

        vec_position_intp = np.zeros((length_grid,3))
        vec_vel_intp = np.zeros((length_grid,3))
        vec_vel_intp_staggered = [None] * (length_grid+1)
        for i_azimiuth, t_azimuth in enumerate(vec_t):
            vec_position_intp[i_azimiuth, :], vec_vel_intp[i_azimiuth, :] = \
                self.orbit.interpolate(t_azimuth)

        # Calculation on staggered grid - For acceleration calculation
        for i_azimiuth, t_azimuth in enumerate(vec_t_staggered):
            _, vec_vel_intp_staggered[i_azimiuth] = self.orbit.interpolate(t_azimuth)

        vec_acceleration_intp = np.diff(vec_vel_intp_staggered, axis=0) / intv_t

        # convert azimuth time to seconds from the reference epoch of burst_in.orbit
        vec_aztime_coeff_fm_rate_sec = np.zeros(self.extended_coeffs_fm_dc.vec_aztime_coeff_fm_rate.shape)
        for i, datetime_vec in enumerate(self.extended_coeffs_fm_dc.vec_aztime_coeff_fm_rate):
            vec_aztime_coeff_fm_rate_sec[i] = (isce3.core.DateTime(datetime_vec)
                                         - self.orbit.reference_epoch).total_seconds()

        vec_aztime_coeff_dc_sec = np.zeros(self.extended_coeffs_fm_dc.vec_aztime_coeff_dc.shape)
        for i, datetime_vec in enumerate(self.extended_coeffs_fm_dc.vec_aztime_coeff_dc):
            vec_aztime_coeff_dc_sec[i] = (isce3.core.DateTime(datetime_vec)
                                    - self.orbit.reference_epoch).total_seconds()

        # calculate splined interpolation of the coeffs. and tau_0s
        interpolator_tau0_ka = InterpolatedUnivariateSpline(vec_aztime_coeff_fm_rate_sec,
                                                            self.extended_coeffs_fm_dc.vec_tau0_fm_rate,
                                                            k=1)
        tau0_ka_interp = interpolator_tau0_ka(vec_t)[..., np.newaxis]

        interpolator_tau0_fdc_interp = InterpolatedUnivariateSpline(vec_aztime_coeff_dc_sec,
                                                                    self.extended_coeffs_fm_dc.vec_tau0_dc,
                                                                    k=1)
        tau0_fdc_interp = interpolator_tau0_fdc_interp(vec_t)[..., np.newaxis]

        grid_tau, grid_t = np.meshgrid(vec_tau, vec_t)

        # add range time origin to vec_tau
        grid_tau += tau0_ka_interp * np.ones(vec_tau.shape)[np.newaxis, ...]
        

        interpolator_b0 = InterpolatedUnivariateSpline(vec_aztime_coeff_fm_rate_sec,
                                                       self.extended_coeffs_fm_dc.lut_coeff_fm_rate[:,0],
                                                       k=1)
        b0_interp = interpolator_b0(vec_t)

        interpolator_b1 = InterpolatedUnivariateSpline(vec_aztime_coeff_fm_rate_sec,
                                                       self.extended_coeffs_fm_dc.lut_coeff_fm_rate[:,1],
                                                       k=1)
        b1_interp = interpolator_b1(vec_t)

        interpolator_b2 = InterpolatedUnivariateSpline(vec_aztime_coeff_fm_rate_sec,
                                                       self.extended_coeffs_fm_dc.lut_coeff_fm_rate[:,2],
                                                       k=1)
        b2_interp = interpolator_b2(vec_t)

        arr_coeff_b = np.array([b0_interp, b1_interp, b2_interp]).transpose()
        
        interpolator_a0 = InterpolatedUnivariateSpline(vec_aztime_coeff_dc_sec,
                                                       self.extended_coeffs_fm_dc.lut_coeff_dc[:,0],
                                                       k=1)
        a0_interp = interpolator_a0(vec_t)

        interpolator_a1 = InterpolatedUnivariateSpline(vec_aztime_coeff_dc_sec,
                                                       self.extended_coeffs_fm_dc.lut_coeff_dc[:,1],
                                                       k=1)
        a1_interp = interpolator_a1(vec_t)

        interpolator_a2 = InterpolatedUnivariateSpline(vec_aztime_coeff_dc_sec,
                                                       self.extended_coeffs_fm_dc.lut_coeff_dc[:,2],
                                                       k=1)
        a2_interp = interpolator_a2(vec_t)

        arr_coeff_a = np.array([a0_interp, a1_interp, a2_interp]).transpose()




        # Run topo on scratch directory
        dem_raster = isce3.io.Raster(path_dem)
        epsg = dem_raster.get_epsg()
        proj = isce3.core.make_projection(epsg)
        ellipsoid = proj.ellipsoid
        grid_doppler = isce3.core.LUT2d()

        Rdr2Geo = isce3.geometry.Rdr2Geo

        rdr2geo_obj = Rdr2Geo(
                radargrid_correction,
                self.orbit,
                ellipsoid,
                grid_doppler)

        str_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        filename_llh = [f'{path_scratch}/lat_{str_datetime}.rdr',
                        f'{path_scratch}/lon_{str_datetime}.rdr',
                        f'{path_scratch}/hgt_{str_datetime}.rdr']

        lat_raster = isce3.io.Raster(filename_llh[0],
                                     radargrid_correction.width,
                                     radargrid_correction.length,
                                     1, gdal.GDT_Float64, 'ENVI')

        lon_raster = isce3.io.Raster(filename_llh[1],
                                     radargrid_correction.width,
                                     radargrid_correction.length,
                                     1, gdal.GDT_Float64, 'ENVI')
        
        hgt_raster = isce3.io.Raster(filename_llh[2],
                                     radargrid_correction.width,
                                     radargrid_correction.length,
                                     1, gdal.GDT_Float64, 'ENVI')


        rdr2geo_obj.topo(dem_raster, lon_raster, lat_raster, hgt_raster)

        lat_raster.close_dataset()
        lon_raster.close_dataset()
        hgt_raster.close_dataset()



        # Do the bunch of computation
        kappa_steer_vec = (2 * (np.linalg.norm(vec_vel_intp, axis=1))
                         / isce3.core.speed_of_light *  self.radar_center_frequency
                         * self.azimuth_steer_rate)

        kappa_steer_grid = kappa_steer_vec[...,np.newaxis] * \
                           np.ones(grid_tau.shape[1])[np.newaxis,...]

        t_burst = (grid_t[0, 0] + grid_t[-1,0]) / 2.0
        index_mid_burst_t = int(grid_t.shape[0]/2 +0.5)
        tau_burst = (grid_tau[index_mid_burst_t,0] + grid_tau[index_mid_burst_t,-1]) / 2.0
        tau0_fdc_burst =  tau0_fdc_interp[index_mid_burst_t]
        tau0_fm_rate_burst =  tau0_ka_interp[index_mid_burst_t]

        a0_burst, a1_burst, a2_burst = arr_coeff_a[index_mid_burst_t,:]
        b0_burst, b1_burst, b2_burst = arr_coeff_b[index_mid_burst_t,:]

        kappa_annotation_burst = (b0_burst
                                + b1_burst*(tau_burst-tau0_fm_rate_burst)
                                + b2_burst*(tau_burst-tau0_fm_rate_burst)**2)

        kappa_annotation_grid = self._evaluate_polynomial_array(arr_coeff_b,
                                                          grid_tau,
                                                          tau0_ka_interp)

        grid_kappa_t = (kappa_annotation_grid * kappa_steer_grid) / (kappa_annotation_grid - kappa_steer_grid)
        freq_dcg_burst = a0_burst + a1_burst*(tau_burst-tau0_fdc_burst) + a2_burst*(tau_burst-tau0_fdc_burst)**2
        grid_freq_dcg = self._evaluate_polynomial_array(arr_coeff_a,
                                                  grid_tau,
                                                  tau0_fdc_interp)

        grid_freq_dc = grid_freq_dcg + grid_kappa_t * ( (grid_t - t_burst) + ( freq_dcg_burst / kappa_annotation_burst - grid_freq_dcg / kappa_annotation_grid))

        #print('Calculating XYZ in ECEF')
        raster_lat = gdal.Open(filename_llh[0], gdal.GA_ReadOnly)
        raster_lon = gdal.Open(filename_llh[1], gdal.GA_ReadOnly)
        raster_hgt = gdal.Open(filename_llh[2], gdal.GA_ReadOnly)

        lat_map = raster_lat.ReadAsArray()
        lon_map = raster_lon.ReadAsArray()
        hgt_map = raster_hgt.ReadAsArray()

        x_ecef, y_ecef, z_ecef = self._latlon_to_ecef(lat_map, lon_map, hgt_map, ellipsoid)

        #print('Calculating True FM rate')

        x_s = vec_position_intp[:,0][..., np.newaxis] * np.ones(grid_tau.shape[1])[np.newaxis, ...]
        y_s = vec_position_intp[:,1][..., np.newaxis] * np.ones(grid_tau.shape[1])[np.newaxis, ...]
        z_s = vec_position_intp[:,2][..., np.newaxis] * np.ones(grid_tau.shape[1])[np.newaxis, ...]

        vx_s = vec_vel_intp[:,0][..., np.newaxis] * np.ones(grid_tau.shape[1])[np.newaxis, ...]
        vy_s = vec_vel_intp[:,1][..., np.newaxis] * np.ones(grid_tau.shape[1])[np.newaxis, ...]
        vz_s = vec_vel_intp[:,2][..., np.newaxis] * np.ones(grid_tau.shape[1])[np.newaxis, ...]

        ax_s = vec_acceleration_intp[:,0][..., np.newaxis] * np.ones(grid_tau.shape[1])[np.newaxis, ...]
        ay_s = vec_acceleration_intp[:,1][..., np.newaxis] * np.ones(grid_tau.shape[1])[np.newaxis, ...]
        az_s = vec_acceleration_intp[:,2][..., np.newaxis] * np.ones(grid_tau.shape[1])[np.newaxis, ...]


        mag_xs_xg = np.sqrt((x_s - x_ecef)**2 + (y_s - y_ecef)**2 + (z_s - z_ecef)**2)

        dotp_dxsg_acc = (x_s - x_ecef)*ax_s + (y_s - y_ecef)*ay_s + (z_s - z_ecef)*az_s

        kappa_annotation_true = ( -(2 * self.radar_center_frequency)
                                 / (isce3.core.speed_of_light * mag_xs_xg)
                                 * (dotp_dxsg_acc + (vx_s**2 + vy_s**2 + vz_s**2)))

        delta_t_freq_mm = grid_freq_dc * (-1/kappa_annotation_grid + 1/kappa_annotation_true)
        
        return delta_t_freq_mm










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
    def burst_duration(self):
        '''Returns burst sensing duration as float in seconds.

        Returns:
        --------
        _ : float
            Burst sensing duration as float in seconds.
        '''
        return self.azimuth_time_interval * self.length

    @property
    def length(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    @property
    def swath_name(self):
        '''Swath name in iw1, iw2, iw3.'''
        return self.burst_id.split('_')[2]

    @property
    def thermal_noise_lut(self):
        '''
        Returns the LUT for thermal noise correction for the burst
        '''
        if self.burst_noise is None:
            raise ValueError('burst_noise is not defined for this burst.')

        return self.burst_noise.compute_thermal_noise_lut(self.shape)

    @property
    def eap_compensation_lut(self):
        '''Returns LUT for EAP compensation.

        Returns:
        -------
            _: Interpolated EAP gain for the burst's lines

        '''
        if self.burst_eap is None:
            raise ValueError('burst_eap is not defined for this burst.'
                            f' IPF version = {self.ipf_version}')

        return self.burst_eap.compute_eap_compensation_lut(self.width)
    def bbox(self):
        '''Returns the (west, south, east, north) bounding box of the burst.'''
        # Uses https://shapely.readthedocs.io/en/stable/manual.html#object.bounds
        # Returns a tuple of 4 floats representing (west, south, east, north) in degrees.
        return self.border[0].bounds
