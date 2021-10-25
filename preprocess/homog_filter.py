# Hannah Kerner
# August 6, 2020
# The USDA Cropland Data Layer (CDL) is used as a surrogate for groundtruth labels for our classifier. 
# Since the CDL is itself a classification (based on a large number of very high-quality groundtruth labels), 
# there are errors that we want to minimize in our labels. Thus we only train with examples that are within 
# a homogeneous (non-speckled) region of the CDL. This script creates a mask of homoegeneous regions.

import numpy as np
import os

def main(labeldir, filtersize, tile_id, year, outdir):
    # Create the filename
    outpath = os.path.join(outdir, 'cdl_labels_%s_%s_homogmask_%dx%d.npy' % (tile_id, year, filtersize, filtersize))
    # Check that we didn't already make it
    if not os.path.exists(outpath):
        # Load the labels
        cdl = np.load(os.path.join(labeldir, 'cdl_labels_%s_%s.npy' % (tile_id, year)))
        # Initialize homogeneity mask
        homog_mask = np.zeros(cdl.shape)
        # Compute the mask
        for r in range(0, cdl.shape[0]-filtersize, filtersize):
            for c in range(0, cdl.shape[1]-filtersize, filtersize):
                patch = cdl[r:r+filtersize,c:c+filtersize]
                if np.all(patch == patch[0,0]):
                    homog_mask[r:r+filtersize,c:c+filtersize] = np.ones(patch.shape)
        # Save the mask
        np.save(outpath, homog_mask)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, 
                                     description='Detect phenology in HLS time series')
    parser.add_argument('--labeldir', help='directory containing labels (numpy files)')
    parser.add_argument('--filtersize', type=int, default=3, help='size of filter to use (default: 3x3)')
    parser.add_argument('--tile_id', help='ID of HLS tile to process (e.g., 16TBL)')
    parser.add_argument('--year', choices=['2016', '2017', '2018', '2019', '2020'], help='year of data')
    parser.add_argument('--outdir', help='directory to store homogeneity mask in')

    args = parser.parse_args()
    main(**vars(args))