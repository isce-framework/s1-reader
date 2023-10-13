def test_correction_shapes(bursts, test_paths):
    az_step, rg_step = 0.25, 200

    for burst in bursts:
        az_fm_mismatch = burst.az_fm_rate_mismatch_mitigation(
            test_paths.dem_file, range_step=rg_step, az_step=az_step
        )
        shape = (az_fm_mismatch.length, az_fm_mismatch.width)

        # Make sure the other two corrections have the same shape
        geometrical_steering_doppler = burst.doppler_induced_range_shift(
            range_step=rg_step, az_step=az_step
        )
        assert shape == (
            geometrical_steering_doppler.length,
            geometrical_steering_doppler.width,
        )

        bistatic_delay = burst.bistatic_delay(range_step=rg_step, az_step=az_step)
        assert shape == (bistatic_delay.length, bistatic_delay.width)
