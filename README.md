## s1-reader

A package to read Sentinel-1 data into the ISCE3-compatible burst class.

### Features

+ Create ISCE3-compatible Sentinel1 burst class given:

  - S1 SAFE
  - subswath index
  - polarization
  - path to orbit directory

+ Monotonically increasing bursts IDs.

🚨 This toolbox is still in **pre-alpha** stage and undergoing **rapid development**. 🚨 

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

### License

**Copyright (c) 2021** California Institute of Technology (“Caltech”). U.S. Government
sponsorship acknowledged.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
* Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the
names of its contributors may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
