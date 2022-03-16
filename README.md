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
git clone https://github.com/opera-adt/s1-reader.git
```

2. Install dependencies:

```bash
conda install -c conda-forge --file s1-reader/requirements.txt
```

3. Install `s1-reader` via pip:

```bash
# run "pip install -e" to install in development mode
python -m pip install ./s1-reader
```

### Usage

The following sample code demonstrates how to process a single burst from a S1 SAFE zip:

```python
import s1reader

zip_path = "S1A_IW_SLC__1SDV_20190909T134419_20190909T134446_028945_03483B_B9E1.zip"
swath_num = 2
pol = "VV"

# read orbits
orbit_dir = '/home/user/data/sentinel1_orbits'
orbit_path = s1reader.get_orbit_file_from_dir(zip_path, orbit_dir)

# returns the list of the bursts
bursts = s1reader.burst_from_zip(zip_path, orbit_path, swath_num, pol)
```
