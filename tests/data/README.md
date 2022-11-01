- `make_empty_safe.sf` converts a full SDV SAFE folder into a 2.1 Mb folder of 1 vv-polarization annotation files.
- `make_empty_img.py` Converts a measurement/ .tiff file into an all-zeros file < 50kb using SPARSE_OK=TRUE.

Example:
```bash
./make_empty_safe.sh /home/staniewi/dev/hawaii/S1A_IW_SLC__1SDV_20220828T042306_20220828T042335_044748_0557C6_F396.SAFE
```

Example datasets:

- S1A_IW_SLC__1SDV_20220828T042306_20220828T042335_044748_0557C6_F396.zip
  - Hawaii ascending pass, was an exmaple area experience the off-by-one burst_id label error
  - orbit file was downloaded and OSV list truncated for space
  - Added by Scott Staniewicz

- S1A_IW_SLC__1SDV_20221024T184148_20221024T184218_045587_05735F_D6E2.zip
  - Equator ascending data which contains a node crossing. The final bursts are a different track than initial bursts.
  - orbit file was downloaded and OSV list truncated for space
  - This one was created using `asfsmd`:
```bash
asfsmd S1A_IW_SLC__1SDV_20221024T184148_20221024T184218_045587_05735F_D6E2 --do-noise --do-cal -iw 2
```

- S1A_IW_SLC__1SDV_20221011T162212_20221011T162240_045397_056D9F_6E2D.zip
  - This one is in track 175, and contains a strange corner case where the metadata provided is from the previous node crossing.


- S1A_IW_SLC__1SDV_20141004T031312_20141004T031339_002674_002FB6_07B5
  - Example of one of the earliest datasets in 2014

**Sample ESA burst database**

- `burst_db_esa.csv` is a sample burst database from ESA. 
It is a subset of the full burst database, and contains only the bursts from test datasets.
It was created using `create_esa_db_sample.py`, which relies on the ESA burst database being available via https://github.com/opera-adt/burst_db .