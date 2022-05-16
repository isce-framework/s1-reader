import pytest

from s1reader.s1_burst_slc import Sentinel1BurstSlc

def test_burst(bursts):
    for i, burst in enumerate(bursts):
        expected_burst_id = f't71_iw3_b{844 + i}'
        assert burst.burst_id == expected_burst_id
