import datetime

from packaging import version

# Minimum IPF version from which the S1 product's Noise Annotation
# Data Set (NADS) includes azimuth noise vector annotation
min_ipf_version_az_noise_vector = version.parse('2.90')

def as_datetime(t_str, fmt = "%Y-%m-%dT%H:%M:%S.%f"):
    '''Parse given time string to datetime.datetime object.

    Parameters:
    ----------
    t_str : string
        Time string to be parsed. (e.g., "2021-12-10T12:00:0.0")
    fmt : string
        Format of string provided. Defaults to az time format found in annotation XML.
        (e.g., "%Y-%m-%dT%H:%M:%S.%f").

    Returns:
    ------
    _ : datetime.datetime
        datetime.datetime object parsed from given time string.
    '''
    return datetime.datetime.strptime(t_str, fmt)
