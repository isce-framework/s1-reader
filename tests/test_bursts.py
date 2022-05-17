import pytest

from s1reader.s1_burst_slc import Sentinel1BurstSlc


def test_burst(bursts):
    last_valid_lines = [1487, 1489, 1489, 1490, 1487, 1488, 1488, 1489, 1488]
    first_valid_lines = [28, 27, 27, 27, 28, 28, 28, 27, 28]

    for i, burst in enumerate(bursts):
        expected_burst_id = f't71_iw3_b{844 + i}'
        assert burst.burst_id == expected_burst_id
        assert burst.i_burst == i

        assert burst.radar_center_frequency == 5405000454.33435
        assert burst.wavelength == 0.05546576
        assert burst.azimuth_steer_rate == 0.024389943375862838
        assert burst.starting_range == 901673.89084624
        assert burst.iw2_mid_range == 875604.926001518
        assert burst.range_sampling_rate == 64345238.12571428
        assert burst.range_pixel_spacing == 2.329562114715323
        assert burst.shape == (1515, 24492)
        assert burst.range_bandwidth == 42789918.40322842

        assert burst.polarization == 'VV'
        assert burst.platform_id == 'S1A'

        assert burst.range_window_type == 'Hamming'
        assert burst.range_window_coefficient == 0.75
        assert burst.rank == 10
        assert burst.prf_raw_data == 1685.817302492702
        assert burst.range_chirp_rate == 801450949070.5804

        assert burst.first_valid_sample == 451
        assert burst.last_valid_sample == 24119
        assert burst.first_valid_line == first_valid_lines[i]
        assert burst.last_valid_line == last_valid_lines[i]
