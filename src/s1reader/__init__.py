# version info
from s1reader.version import release_version as __version__

# top-level functions to be easily used
from s1reader.s1_burst_slc import Sentinel1BurstSlc
from s1reader.s1_reader import load_bursts
from s1reader.s1_orbit import get_orbit_file_from_dir
