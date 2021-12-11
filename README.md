### Features:
- Create ISCE3-compatible Sentinel1 burst class given:

    0. S1 SAFE
    1. subswath index
    2. polarization
    3. path to orbit directory
- Monotonically increasing bursts IDs.

### Install:

1. Set up and activate virtual environment with ISCE3.
2. Clone repository.
```
$ cd ~/src
$ git clone https://github.com/LiangJYu/sentinel1-reader.git
```
3. Install into virtual environment with pip. From clone directory:
```
$ cd sentinel1-reader
$ pip install .
```

### Usage:
The following sample code demonstrates how to process a single burst from a S1 SAFE zip:
```python
from sentinel1_reader import sentinel1_reader, sentinel1_orbit_reader

zip_path = "S1A_IW_SLC__1SDV_20190909T134419_20190909T134446_028945_03483B_B9E1.zip"
i_subswath = 2
pol = "HH"

# read orbits
orbit_dir = '/home/user/data/sentinel1_orbits'
osv_list = sentinel1_orbit_reader.get_swath_osv_list(zip_path, orbit_dir)

# returns the list of the bursts
bursts = sentinel1_reader.zip2bursts(zip_path, osv_list, i_subswath, pol)
```
