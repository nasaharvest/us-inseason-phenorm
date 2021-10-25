# Hannah Kerner
# October 23, 2020
# Detect phenology in raw/linearly interpolated data and assemble inputs

import numpy as np
import os
import rasterio as rio
import pandas as pd
import sys
from glob import glob

import xarray as xr
import dask
from dask.distributed import Client

HEIGHT = 3660
WIDTH = 3660

def remove_dups(obs):
    doys = []
    _obs = []
    for ob in sorted(obs):
        doy = ob.split('.')[3][4:]
        if doy in doys:
            print('Skipping %s because we already have an observation on DOY %s' % (ob, doy))
            continue
        else:
            doys.append(doy)
            _obs.append(ob)
    return sorted(_obs, key=lambda name: name.split('.')[3][4:])

def get_ndvi(red, nir):
    # Compute numerator and denominator
    num = nir-red
    denom = nir+red
    # Set 0s in denominator to small positive value
    denom[np.where(denom==0)] = 0.00000000001
    return num/denom

def find_greenup(s, year):
    # find the biggest increase in the first n% of observations
    start = '%s-04-15' % year
    end = '%s-07-01' % year
    s = s.loc[start:end]
    d_ndvi = s.diff()
    greenup = d_ndvi[d_ndvi==d_ndvi.sort_values().dropna()[-1]].index[0]
    # print(greenup)
    return greenup

def find_senescence(s, peak, year):
    # find the biggest decrease after peak
    s = s.loc[peak:]
    d_ndvi = s.diff()
    senescence = d_ndvi[d_ndvi==d_ndvi.sort_values().dropna()[0]].index[0]
    # print(senescence)
    return senescence

def find_peak(s, greenup, year):
    # find the highest NDVI between greenup and senescence
    end = '%s-09-30' % year
    s = s.loc[greenup:end]
    peak = s[s==s.max()].index[0]
    # print(peak)
    return peak

def date_to_doy(date):
    # print(date.timetuple().tm_yday)
    return date.timetuple().tm_yday

def extract_pheno_points(series, doys, timesteps, basepath, tile, year):
    print('Beginning phenology detection')
    # Convert the DOYs to dates
    dates = (np.asarray(np.full(len(doys), int(year)), dtype='datetime64[Y]')-1970)+(np.asarray(doys, dtype='timedelta64[D]')-1)
    # Make an array to hold the doys of interest (DOIs)
    ndvi_dois = np.zeros([HEIGHT, WIDTH, timesteps])
    n_invalid = 0
    for row in range(HEIGHT):
        for col in range(WIDTH):
            # Filter out NaNs and all-zeros
            if np.all(np.isnan(series[row,col])) or np.all(series[row,col]==0):
                print('Series of NaNs or zeros in %d, %d' % (row, col))
                n_invalid += 1
                continue
            # Resample and interpolate the time series to 5-day intervals            
            interped = pd.Series(series[row,col], index=dates).interpolate(limit_direction='both')
            resampled = interped.resample('5D').pad()
            # Calculate the approximate greenup date
            greenup = find_greenup(resampled, year)
            # Convert to DOY and store
            ndvi_dois[row,col,0] = date_to_doy(greenup)
            if timesteps > 1:
                # Get the DOY of peak NDVI
                peak = find_peak(resampled, greenup, year)
                # Convert to DOY and store
                ndvi_dois[row,col,1] = date_to_doy(peak)
            # Only calculate the senescence DOY if we want 3 timesteps
            if timesteps > 2:
                # Calculate the approximate senescence date
                senescence = find_senescence(resampled, peak, year)
                # Convert to DOY and store
                ndvi_dois[row,col,2] = date_to_doy(senescence)

    print('%d invalid pixel locations (all nan or all zero) found. These will have DOYs=0.' % n_invalid)
    return ndvi_dois

def detect_stages(chunk, dates, timesteps, year):
    ndvi_dois = np.zeros([chunk.shape[0], 3])
    for c in range(chunk.shape[0]):
        # Filter out NaNs and all-zeros
        if np.all(np.isnan(chunk[c])) or np.all(chunk[c]==0):
            continue
        # Resample and interpolate the time series to 5-day intervals            
        interped = pd.Series(chunk[c], index=dates).interpolate(limit_direction='both')
        resampled = interped.resample('5D').pad()
        # Calculate the approximate greenup date
        greenup = find_greenup(resampled, year)
        # Convert to DOY and store
        ndvi_dois[c,0] = date_to_doy(greenup)
        if timesteps > 1:
            # Get the DOY of peak NDVI
            peak = find_peak(resampled, greenup, year)
            # Convert to DOY and store
            ndvi_dois[c,1] = date_to_doy(peak)
        # Only calculate the senescence DOY if we want 3 timesteps
        if timesteps > 2:
            # Calculate the approximate senescence date
            senescence = find_senescence(resampled, peak, year)
            # Convert to DOY and store
            ndvi_dois[c,2] = date_to_doy(senescence)

    return ndvi_dois

def extract_pheno_points_dask(series, doys, timesteps, basepath, tile, year):
    print('Beginning phenology detection')
    # Convert the DOYs to dates
    dates = (np.asarray(np.full(len(doys), int(year)), dtype='datetime64[Y]')-1970)+(np.asarray(doys, dtype='timedelta64[D]')-1)
    print(dates)
    # Create an xarray DataArray of the time series
    tseries_xr = xr.DataArray(series.reshape(series.shape[0]*series.shape[1], series.shape[2]))
    tseries = dask.delayed(tseries_xr)
    delayed = []
    chunk_size = 500
    for chunk in range(int(series.shape[0]*series.shape[1]/chunk_size)):
        ts = dask.delayed(detect_stages)(tseries[chunk*chunk_size:chunk*chunk_size+chunk_size], dates, timesteps, year)
        delayed.append(ts)

    futures = dask.persist(*delayed)
    results = dask.compute(*futures)
    chunked_dois = np.array(results)
    
    # Put them back into the proper shape
    dois_reshaped = np.ndarray([tseries_xr.shape[0], 3])
    for chunk in range(chunked_dois.shape[0]):
        dois_reshaped[chunk*chunk_size:chunk*chunk_size+chunk_size] = chunked_dois[chunk]
    
    return np.reshape(dois_reshaped, [3660, 3660, 3])

def get_img_path(basepath, fp, band, imname):
    if 'S30' in imname:
        path = os.path.join(basepath, fp % (band, 'S'), imname)
    else:
        path = os.path.join(basepath, fp % (band, 'L'), imname)
    return path

# Load time series
def load_time_series(basepath, band, tile, year):
    print('Loading time series for band %s' % band)
    # Load the filepaths for all observations
    fp = os.path.join('%s', tile, '%s30', year)
    # check if the ndvi data was created
    if os.path.exists(os.path.join(basepath, fp % (band, 'S'))) and \
        (len(os.listdir(os.path.join(basepath, fp % (band, 'S')))) == \
        len(os.listdir(os.path.join(basepath, fp % ('red', 'S'))))):
        # Gather a list of all the observations
        total_obs = np.concatenate([os.listdir(os.path.join(basepath, fp % (band, 'S'))),
                                    os.listdir(os.path.join(basepath, fp % (band, 'L')))
                                   ])
        # Sort the observations by the day of year
        total_obs = sorted(total_obs, key=lambda name: name.split('.')[3][4:])
        # Get the list of unique DOYs
        doys = list(set([name.split('.')[3][4:] for name in total_obs]))
        # Create an array for the DOYs
        X_doy = np.ndarray([len(doys)], dtype=np.uint16)
        # Create an array for the data
        X = np.zeros([3660, 3660, len(doys)])
        for t, doy in enumerate(sorted(doys)):
            # Save the DOY 
            X_doy[t] = int(doy)
            # Get the corresponding path
            obs = [f for f in total_obs if '%s%s' % (year, doy) in f]
            if len(obs) == 2:
                # load both instruments and merge them
                p1 = get_img_path(basepath, fp, band, obs[0])
                p2 = get_img_path(basepath, fp, band, obs[1])
                # Read each image
                with rio.open(p1) as src:
                    im1 = src.read(1)
                with rio.open(p2) as src:
                    im2 = src.read(1)
                # Merge the images by taking the maximum value
                img = np.maximum(im1, im2)
            elif len(obs) == 1: 
                # Open/read the raster file
                path = get_img_path(basepath, fp, band, obs[0])
                # Read the image
                with rio.open(path) as src:
                    img = src.read(1)
            else:
                print('More than 2 observations found for one DOY: %s' % doy)
                sys.exit(0)
            # Add this observation to the time series cube
            X[...,t] = img
    # if ndvi wasn't created, compute it now
    else:
        total_obs_red = np.concatenate([os.listdir(os.path.join(basepath, fp % ('red', 'S'))),
                                os.listdir(os.path.join(basepath, fp % ('red', 'L')))
                               ])
        total_obs_nir = np.concatenate([os.listdir(os.path.join(basepath, fp % ('nir_narrow', 'S'))),
                                os.listdir(os.path.join(basepath, fp % ('nir_narrow', 'L')))
                               ])
        # Sort the observations by the day of year
        total_obs_red = sorted(total_obs_red, key=lambda name: name.split('.')[3][4:])
        total_obs_nir = sorted(total_obs_nir, key=lambda name: name.split('.')[3][4:])
        # Get the list of unique DOYs
        doys = list(set([name.split('.')[3][4:] for name in total_obs_red]))
        # Create an array for the DOYs
        X_doy = np.ndarray([len(doys)], dtype=np.uint16)
        # Create an array for the data
        X = np.zeros([3660, 3660, len(doys)])
        for t, doy in enumerate(sorted(doys)):
            # Save the DOY 
            X_doy[t] = int(doy)
            # Get the corresponding path
            obs_red = [f for f in total_obs_red if '%s%s' % (year, doy) in f]
            obs_nir = [f for f in total_obs_nir if '%s%s' % (year, doy) in f]
            if len(obs_red) == 2:
                # load both instruments and merge them
                p1_red = get_img_path(basepath, fp, 'red', obs_red[0])
                p1_nir = get_img_path(basepath, fp, 'nir_narrow', obs_nir[0])
                p2_red = get_img_path(basepath, fp, 'red', obs_red[1])
                p2_nir = get_img_path(basepath, fp, 'nir_narrow', obs_nir[1])
                # Read each image
                with rio.open(p1_red) as src:
                    im1_red = src.read(1)
                with rio.open(p1_nir) as src:
                    im1_nir = src.read(1)
                with rio.open(p2_red) as src:
                    im2_red = src.read(1)
                with rio.open(p2_nir) as src:
                    im2_nir = src.read(1)
                # Merge the images by taking the maximum value
                img_red = np.maximum(im1_red, im2_red)
                img_nir = np.maximum(im1_nir, im2_nir)
            elif len(obs_red) == 1: 
                # Open/read the raster file
                path_red = get_img_path(basepath, fp, 'red', obs_red[0])
                path_nir = get_img_path(basepath, fp, 'nir_narrow', obs_nir[0])
                # Read the image
                with rio.open(path_red) as src:
                    img_red = src.read(1)
                with rio.open(path_nir) as src:
                    img_nir = src.read(1)
            else:
                print('More than 2 observations found for one DOY: %s' % doy)
                sys.exit(0)
            # Calculate ndvi from the red and blue bands
            denom = img_red + img_nir
            # if there is going to be a 0 in the denominator, set it to some 
            # very small value to avoid nans
            denom[np.where(denom == 0)] = 0.000001
            ndvi = (img_nir - img_red) / denom
            # clip the values to -1, 1
            ndvi = np.clip(ndvi, -1, 1)
            # Add this observation to the time series cube
            X[...,t] = ndvi
    return X, X_doy

def main(datadir, tile_id, year, outdir, timesteps, dask=False):
    print('Beginning phenology detection for HLS tile %s (%s)' % (tile_id, year))
    ndvi, doys = load_time_series(datadir, 'ndvi', tile_id, year)
    print('Loaded NDVI time series with shape: ', ndvi.shape)
    if dask:
        # Start the dask client 
        client = Client(threads_per_worker=4, n_workers=5)
        client.cluster.scale(15)
        print('Started dask client')
        # Detect phenology in each pixel
        ndvi_dois = extract_pheno_points_dask(ndvi, doys, timesteps, datadir, tile_id, year)
    else:
        # Detect phenology in each pixel
        ndvi_dois = extract_pheno_points(ndvi, doys, timesteps, datadir, tile_id, year)
    # Save the phenology map
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    np.save(os.path.join(outdir, '%s_%s_phenology.npy' % (tile_id, year)), ndvi_dois)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, 
                                     description='Detect phenology in HLS time series')
    parser.add_argument('--datadir', default='/gpfs/data1/cmongp1/GEOGLAM/Input/field_data/hls/tif', 
                                     help='directory containing HLS data (subfolders are tile ids)')
    parser.add_argument('--tile_id', help='ID of HLS tile to process (e.g., 16TBL)')
    parser.add_argument('--year', choices=['2016', '2017', '2018', '2019', '2020', '2021'], help='year of data')
    parser.add_argument('--outdir', help='directory to store phenology products in')
    parser.add_argument('--timesteps', type=int, default=3, help='number of phenology stages to detect (1, 2, or 3)')
    parser.add_argument('--dask', action='store_true', help='Use dask to speed up computation')

    args = parser.parse_args()
    main(**vars(args))