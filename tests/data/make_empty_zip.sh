#!/bin/bash

set -e

INDIR=$1
echo $INDIR

mkdir -p out

SAFENAME=$(basename $INDIR)

cp -r $INDIR out/


NEWSAFE=out/$SAFENAME 

echo "Keeping only vv, subswath 2"
# Keep only vv files
find "$NEWSAFE" -name "*vh*" -delete

# Keep only 1 polarization
find "$NEWSAFE" -name "*vh*" -delete

# Keep only 1 subswath
find "$NEWSAFE" -name "*iw1*" -delete
find "$NEWSAFE" -name "*iw3*" -delete

find "$NEWSAFE" -name "*pdf" -delete
find "$NEWSAFE" -name "*kml" -delete
find "$NEWSAFE" -name "*xsd" -delete
find "$NEWSAFE" -name "*html" -delete
find "$NEWSAFE" -name "*kml" -delete
find "$NEWSAFE" -name "*png" -delete

rmdir "$NEWSAFE"/preview/icons
rmdir "$NEWSAFE"/preview
rmdir "$NEWSAFE"/support

INIMAGE=$(find "$NEWSAFE" -name "*iw2*tiff")
echo "Converting $INIMAGE to empty file"

# Make empty .tiff files with rasterio
python make_empty_img.py "$INIMAGE" "$NEWSAFE"/test.tiff
mv "$NEWSAFE"/test.tiff "$INIMAGE"

# Clean up temp out folder
mv out/* .
rmdir out

# cp $INIMAGE "$NEWSAFE"/test.tiff
# gdal_calc.py -A $INIMAGE  --calc "A * 0" --outfile "$NEWSAFE"/test.tiff
# gdal_translate -co "COMPRESS=LZW" "$NEWSAFE"/test.tiff "$NEWSAFE"/test2.tiff

# http://stackoverflow.com/a/9559024/1709587
zip -r "${NEWSAFE%.SAFE}.zip" "$NEWSAFE"

# PLEASE manually remove so I dont put rm -rf in here
# rm -rf "$NEWSAFE"
echo "Created ${NEWSAFE%.SAFE}.zip, please remove $NEWSAFE"