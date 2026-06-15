from datetime import timedelta

import isce3
import numpy as np


def test_ew3_burst(ew3_bursts):
    """Detailed per-burst assertions for EW3 (reference subswath), all 21 bursts."""
    first_valid_lines = [
        11, 10, 10, 10, 10, 11, 11, 10, 11, 10,
        10, 10, 10, 10, 11, 10, 10, 11, 10, 10, 11,
    ]
    last_valid_lines = [
        1158, 1159, 1159, 1159, 1159, 1160, 1160, 1158, 1159, 1159,
        1159, 1159, 1159, 1159, 1159, 1159, 1158, 1159, 1159, 1158, 1159,
    ]
    first_valid_samples = [
        177, 177, 177, 177, 177, 144, 144, 144, 144, 144,
        144, 144, 144, 109, 109, 109, 109, 109, 109, 109, 109,
    ]
    last_valid_samples = [
        8264, 8290, 8290, 8290, 8290, 8258, 8258, 8258, 8258, 8258,
        8258, 8258, 8258, 8222, 8222, 8222, 8222, 8222, 8222, 8222, 8222,
    ]
    doppler_poly1d_means = [
        757722.593502511,
        757722.593502511,
        757722.593502511,
        757722.593502511,
        757722.593502511,
        757474.9943291758,
        757474.9943291758,
        757474.9943291758,
        757474.9943291758,
        757474.9943291758,
        757474.9943291758,
        757474.9943291758,
        757474.9943291758,
        757207.4274805713,
        757207.4274805713,
        757207.4274805713,
        757207.4274805713,
        757207.4274805713,
        757207.4274805713,
        757207.4274805713,
        757207.4274805713,
    ]
    doppler_poly1d_coeffs = [
        [-17.8658, 29624.21, -44616930.0],
        [-1.636249, -40614.04, 22983880.0],
        [-16.94384, 2916.613, 2954912.0],
        [7.834071, -71113.98, 52241540.0],
        [-5.388566, -17533.07, 15668800.0],
        [12.52405, -77214.48, 49933180.0],
        [-2.564445, -47229.0, 39757950.0],
        [-30.13674, 34855.91, -6454965.0],
        [-1.783859, -46585.86, 37822040.0],
        [9.322639, -53014.72, 34717800.0],
        [-8.235366, -2109.4, 3827758.0],
        [-10.08351, 10538.47, -7184266.0],
        [-13.24611, 17524.04, -12234850.0],
        [-17.21178, 19043.38, -12374060.0],
        [-5.485541, -719.1741, -5859630.0],
        [-6.84397, 3731.092, -10071820.0],
        [-5.947348, 1636.121, -10197640.0],
        [-8.474328, 1479.348, -9154625.0],
        [-9.371878, 4143.143, -12101640.0],
        [-7.773955, -973.2105, -9594659.0],
        [-6.94278, -1942.1, -10921540.0],
    ]
    az_fm_rate_poly1d_coeffs = [
        [-2216.788279103713, 388007.5657551105, -61214206.08930667],
        [-2216.780349228411, 388016.2859927992, -61219229.03209888],
        [-2216.772247649814, 388022.527963207, -61219149.30283939],
        [-2216.776102254683, 388035.5697346788, -61228637.08388517],
        [-2216.777865567052, 388043.9592427227, -61224251.97489108],
        [-2216.793173318385, 388060.2024769661, -61233101.92199985],
        [-2216.794110979351, 388069.2424198801, -61232783.00468411],
        [-2216.7880524017, 388078.1875537658, -61235095.15307039],
        [-2216.771504496482, 388084.1980134749, -61237805.94775411],
        [-2216.768084515585, 388093.6370369247, -61240118.09614038],
        [-2216.780773791639, 388109.143846451, -61245938.33138859],
        [-2216.756343631958, 388112.1160033888, -61246576.16560338],
        [-2216.765119280136, 388125.3760563288, -61248968.04297137],
        [-2216.788053743136, 388143.045837667, -61253273.42270596],
        [-2216.814718728535, 388162.5765242131, -61260449.05508768],
        [-2216.841599684495, 388182.6319685378, -61270973.31636792],
        [-2216.868665087355, 388201.225587932, -61274561.13276712],
        [-2216.896041702435, 388221.3052856989, -61281816.49454721],
        [-2216.92354977786, 388240.7058854239, -61286839.43761721],
        [-2216.951446197847, 388262.1548038915, -61297682.6159354],
        [-2216.979411701578, 388281.484847811, -61302067.72451281],
    ]

    for i, burst in enumerate(ew3_bursts):
        expected_burst_id = f"t132_{256982 + i}_ew3"
        assert burst.burst_id == expected_burst_id
        assert burst.i_burst == i
        assert burst.abs_orbit_number == 42554

        assert burst.radar_center_frequency == 5405000454.33435
        assert burst.wavelength == 0.05546576
        assert burst.azimuth_steer_rate == 0.04129790841679233

        assert burst.average_azimuth_pixel_spacing == 19.96579

        assert burst.starting_range == 841524.9298388781
        assert burst.ew3_mid_range == 866426.6176668337
        assert burst.iw2_mid_range is None
        assert burst.range_sampling_rate == 25023148.16
        assert burst.range_pixel_spacing == 5.990302580696545
        assert burst.shape == (1168, 8314)
        assert burst.range_bandwidth == 12900000.0

        assert burst.polarization == "HH"
        assert burst.platform_id == "S1A"

        assert burst.range_window_type == "Hamming"
        assert burst.range_window_coefficient == 0.75
        assert burst.rank == 9
        assert burst.prf_raw_data == 1647.777437113131
        assert burst.range_chirp_rate == 425245977168.2125

        assert burst.first_valid_sample == first_valid_samples[i]
        assert burst.last_valid_sample == last_valid_samples[i]
        assert burst.first_valid_line == first_valid_lines[i]
        assert burst.last_valid_line == last_valid_lines[i]

        assert burst.doppler.poly1d.order == 2
        assert burst.doppler.poly1d.mean == doppler_poly1d_means[i]
        assert burst.doppler.poly1d.std == 149896229.0
        assert burst.doppler.poly1d.coeffs == doppler_poly1d_coeffs[i]

        # compare doppler poly1d and lut2d
        r0 = burst.starting_range + 0.5 * burst.width * burst.range_pixel_spacing
        t0 = isce3.core.DateTime(burst.sensing_mid) - burst.orbit.reference_epoch
        assert np.isclose(
            burst.doppler.lut2d.eval(t0.total_seconds(), r0),
            burst.doppler.poly1d.eval(r0),
        )

        assert burst.azimuth_fm_rate.order == 2
        assert burst.azimuth_fm_rate.mean == 841524.9298388781
        assert burst.azimuth_fm_rate.std == 149896229.0
        assert burst.azimuth_fm_rate.coeffs == az_fm_rate_poly1d_coeffs[i]


def test_ew_subswath_properties(ew_bursts_by_subswath):
    """Test per-subswath burst properties: shape, ranges, sensor parameters."""
    expected_n_bursts = {1: 21, 2: 21, 3: 21, 4: 21, 5: 20}
    expected_shapes = {
        1: (1171, 8167),
        2: (1161, 6726),
        3: (1168, 8314),
        4: (1164, 8295),
        5: (1150, 7064),
    }
    expected_starting_ranges = {
        1: 756342.8271413732,
        2: 803175.0127172588,
        3: 841524.9298388781,
        4: 889405.4183663855,
        5: 937537.4996022823,
    }
    expected_az_steer_rates = {
        1: 0.04172899763854488,
        2: 0.0490699794625894,
        3: 0.04129790841679233,
        4: 0.04385479449540044,
        5: 0.03705081674498013,
    }
    expected_az_pix_spacings = {
        1: 19.92763,
        2: 19.94987,
        3: 19.96579,
        4: 19.97841,
        5: 19.9868,
    }
    expected_rg_bandwidths = {
        1: 22200000.0,
        2: 15100000.0,
        3: 12900000.0,
        4: 11293284.46790655,
        5: 10400000.0,
    }
    expected_rg_win_coefficients = {1: 0.6, 2: 0.75, 3: 0.75, 4: 0.75, 5: 0.75}
    expected_prfs = {
        1: 1647.922124950608,
        2: 1939.277821751486,
        3: 1647.777437113131,
        4: 1897.897671032007,
        5: 1630.668270049526,
    }
    expected_ranks = {1: 8, 2: 10, 3: 9, 4: 11, 5: 10}
    expected_chirp_rates = {
        1: 732927900616.9348,
        2: 584461295634.036,
        3: 425245977168.2125,
        4: 428604950131.6264,
        5: 339256269304.814,
    }

    for swath_num, bursts in ew_bursts_by_subswath.items():
        assert len(bursts) == expected_n_bursts[swath_num]

        for burst in bursts:
            assert burst.shape == expected_shapes[swath_num]
            assert burst.starting_range == expected_starting_ranges[swath_num]
            assert burst.ew3_mid_range == 866426.6176668337
            assert burst.iw2_mid_range is None
            assert burst.radar_center_frequency == 5405000454.33435
            assert burst.wavelength == 0.05546576
            assert burst.range_sampling_rate == 25023148.16
            assert burst.range_pixel_spacing == 5.990302580696545
            assert burst.abs_orbit_number == 42554
            assert burst.polarization == "HH"
            assert burst.platform_id == "S1A"
            assert burst.azimuth_steer_rate == expected_az_steer_rates[swath_num]
            assert burst.average_azimuth_pixel_spacing == expected_az_pix_spacings[swath_num]
            assert burst.range_bandwidth == expected_rg_bandwidths[swath_num]
            assert burst.range_window_type == "Hamming"
            assert burst.range_window_coefficient == expected_rg_win_coefficients[swath_num]
            assert burst.prf_raw_data == expected_prfs[swath_num]
            assert burst.rank == expected_ranks[swath_num]
            assert burst.range_chirp_rate == expected_chirp_rates[swath_num]


def test_ew3_as_isce3_radargrid(ew3_bursts):
    """Test isce3 radar grid creation for EW3 bursts."""
    for burst in ew3_bursts:
        grid = burst.as_isce3_radargrid()
        assert grid.width == burst.width
        assert grid.length == burst.length
        assert grid.starting_range == burst.starting_range
        dt = isce3.core.DateTime((burst.sensing_start - timedelta(days=2)))
        assert dt == grid.ref_epoch
        assert grid.prf == 1 / burst.azimuth_time_interval
        assert grid.range_pixel_spacing == burst.range_pixel_spacing
        assert str(grid.lookside) == "LookSide.Right"
        assert grid.wavelength == burst.wavelength


def test_ew3_as_isce3_radargrid_step_change(ew3_bursts):
    """Test changing az_step / rg_step in as_isce3_radargrid() for EW3."""
    burst = ew3_bursts[0]
    rg_step = burst.range_pixel_spacing
    az_step = burst.azimuth_time_interval
    grid = burst.as_isce3_radargrid(az_step=az_step, rg_step=rg_step)
    assert grid.width == burst.width
    assert grid.length == burst.length
    assert grid.prf == 1 / az_step

    rg_step *= 2
    grid = burst.as_isce3_radargrid(rg_step=rg_step)
    assert grid.width == burst.width // 2
    assert grid.length == burst.length

    az_step *= 2
    grid = burst.as_isce3_radargrid(az_step=az_step)
    assert grid.width == burst.width
    assert grid.length == burst.length // 2
