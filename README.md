### Features

+ Create ISCE3-compatible Sentinel1 burst class given:

  - S1 SAFE
  - subswath index
  - polarization
  - path to orbit directory

+ Monotonically increasing bursts IDs.

### Install

1. Download source code:

```bash
git clone https://github.com/opera-adt/sentinel1-reader.git
```

2. Install dependencies:

```bash
conda install -c conda-forge --file sentinel1-reader/docs/requirements.txt
```

3. Install `sentinel1-reader` via pip:

```bash
# run "pip install -e" to install in development mode
python -m pip install ./sentinel1-reader
```

### Usage

The following sample code demonstrates how to process a single burst from a S1 SAFE zip:

```python
from sentinel1_reader import sentinel1_reader, sentinel1_orbit_reader

zip_path = "S1A_IW_SLC__1SDV_20190909T134419_20190909T134446_028945_03483B_B9E1.zip"
i_subswath = 2
pol = "HH"

# read orbits
orbit_dir = '/home/user/data/sentinel1_orbits'
orbit_path = sentinel1_orbit_reader.get_swath_orbit_file_from_dir(zip_path, orbit_dir)

# returns the list of the bursts
bursts = sentinel1_reader.burst_from_zip(zip_path, orbit_path, i_subswath, pol)
```
