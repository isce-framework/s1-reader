- `make_empty_safe.sf` converts a full SDV SAFE folder into a 2.1 Mb folder of 1 vv-polarization annotation files.
- `make_empty_img.py` Converts a measurement/ .tiff file into an all-zeros file < 50kb using SPARSE_OK=TRUE.

Example:
```
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
    - Added by Scott Staniewicz

