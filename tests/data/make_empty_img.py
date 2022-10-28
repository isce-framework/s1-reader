import sys

from osgeo import gdal

infile, outfile = sys.argv[1:]

ds_in = gdal.Open(infile)
drv = ds_in.GetDriver()
data_type = ds_in.GetRasterBand(1).DataType
xsize, ysize = ds_in.RasterXSize, ds_in.RasterYSize
opts = ["SPARSE_OK=TRUE", "TILED=YES"]
ds_out = drv.Create(outfile, xsize, ysize, 1, data_type, options=opts)
ds_out = None
