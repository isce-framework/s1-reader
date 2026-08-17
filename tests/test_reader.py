def test_burst_from_zip(bursts):
    assert len(bursts) == 9


def test_burst_from_zip_ew3(ew3_bursts):
    assert len(ew3_bursts) == 21


def test_ew_burst_counts(ew_bursts_by_subswath):
    expected_counts = {1: 21, 2: 21, 3: 21, 4: 21, 5: 20}
    for swath_num, bursts in ew_bursts_by_subswath.items():
        assert len(bursts) == expected_counts[swath_num]
