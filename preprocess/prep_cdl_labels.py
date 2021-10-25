# Hannah Kerner
# August 18, 2020
# Format CDL for model labels
# Must be run in uscorn conda environment

import numpy as np
import os
from glob import glob
from subprocess import call

import rasterio
import gdal
from gdalconst import GA_ReadOnly

CDL_CORN = 1
CDL_SOYBEAN = 5
OTHER = 2

def remap_labels(labels):
    _labels = np.full_like(labels, fill_value=OTHER).astype(np.uint8)
    # Set all corn labels to 0
    _labels[np.where(labels==CDL_CORN)] = 0
    # Set all soybean labels to 1
    _labels[np.where(labels==CDL_SOYBEAN)] = 1
    return _labels

def main(cdlpath, tile_id, year, hlsimg):
    # Check if we've already made this label file
    if not os.path.exists('data/cdl/cdl_labels_%s_%s.npy' % (tile_id, year)):
        # Reproject the national CDL using the tile zone, if not done already
        if not os.path.exists('data/cdl/national_%s_utm%s.tif' % (year, tile_id[:2])):
            print('Reprojecting national CDL to zone 326%s' % tile_id[:2])
            call('gdalwarp -t_srs EPSG:326%s -tr 30 30 %s data/cdl/national_%s_utm%s.tif' % (tile_id[:2], cdlpath, year, tile_id[:2]), shell=True)
        if not os.path.exists('data/cdl/cdl_%s_%s.tif' % (year, tile_id)):
            # Get the bounds of the HLS tile we want to clip to
            data = gdal.Open(hlsimg, GA_ReadOnly)
            # Get the transform info
            geoTransform = data.GetGeoTransform()
            minx = geoTransform[0]
            maxy = geoTransform[3]
            maxx = minx + geoTransform[1] * data.RasterXSize
            miny = maxy + geoTransform[5] * data.RasterYSize
            # Clip the CDL to the bounds of our tile
            call('gdal_translate -projwin ' + 
                  ' '.join([str(x) for x in [minx, maxy, maxx, miny]]) + 
                  ' -of GTiff data/cdl/national_%s_utm%s.tif data/cdl/cdl_%s_%s.tif' 
                  % (year, tile_id[:2], year, tile_id), shell=True)
        # Load the cropped raster
        with rasterio.open('data/cdl/cdl_%s_%s.tif' % (year, tile_id)) as src:
            cdl = src.read(1)
            print(src.meta)
        cdl = remap_labels(cdl)
        # Write the labels to .npy files
        np.save('data/cdl/cdl_labels_%s_%s.npy' % (tile_id, year), cdl)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, 
                                     description='Format CDL for model labels')
    parser.add_argument('--cdlpath', help='Path to national CDL')
    parser.add_argument('--tile_id', help='ID of HLS tile to process (e.g., 16TBL)')
    parser.add_argument('--year', choices=['2016', '2017', '2018', '2019', '2020'], help='year of data')
    parser.add_argument('--hlsimg', help='path to an example HLS image we want to match the CDL raster to')

    args = parser.parse_args()
    main(**vars(args))