import os
import shutil
from urllib.request import urlopen

s3_url = "http://sentinel1-slc-seasia-pds.s3-website-ap-southeast-1.amazonaws.com/datasets/slc/v1.1"
data_info = "2021/08/15"
granule_id = "S1A_IW_SLC__1SDV_20210815T100025_20210815T100055_039238_04A1E8_D145"


def download_granule(in_url):
    zip_file = in_url.split('/')[-1]
    if os.path.isfile(zip_file) == False:
        print(f'Start downloading of {in_url}')
        with urlopen(in_url) as response, open(zip_file, 'wb') as ofile:
            shutil.copyfileobj(response, ofile)


def test_data_download():
    '''
    Test S1-A/B data download. Use seasia AWS s3 bucket
    to download data. Note, it contains only data acquired
    South East Asia, Taiwan, Korea, and Japan.
    '''
    # Get Full data URL
    data_url = f'{s3_url}/{data_info}/{granule_id}/{granule_id}.zip'
    download_granule(data_url)
    print('Done')


if __name__ == "__main__":
    test_data_download()
