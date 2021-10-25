# Hannah Kerner
# August 20, 2020
# Make inputs from phenology DOYs using raw interpolated data

import numpy as np
import os
import rasterio as rio
import pandas as pd
import sys

import xarray as xr
import dask.dataframe as dd
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
    return X, X_doy


def process_chunk(chunk, bands, dois, doys, year):
    dates = (np.asarray(np.full(len(doys), int(year)), dtype='datetime64[Y]')-1970)+(np.asarray(doys, dtype='timedelta64[D]')-1)
    x = np.zeros([chunk.shape[0], 3])
    for c in range(chunk.shape[0]):
        # If they are all zero, there were too few observations to select
        if np.all(dois[c]) == 0:
            continue
        # also skip if the time series is all nan (and somehow wasn't caught in the last step)
        elif np.all(np.isnan(chunk[c])):
            continue
        # linearly interpolate to fill nans and resample to 5D intervals
        interped = pd.Series(chunk[c], index=dates).interpolate(limit_direction='both')
        interped = interped.resample('5D').pad()
        # Load the greenup reflectances                
        greenup_doy = np.where(date_to_doy(interped.index)==int(dois[c,0]))[0][0]
        if timesteps > 1:
            # Load the peak reflectances
            peak_doy = np.where(date_to_doy(interped.index)==int(dois[c,1]))[0][0]
        if timesteps > 2:
            # Load the senescence reflectances
            sen_doy = np.where(date_to_doy(interped.index)==int(dois[c,2]))[0][0]
        # Store them in the input array
        if augment:
            # add one timestep 5 observations before greenup
            x[c,0] = interped[greenup_doy-5]
            # greenup
            x[c,1] = interped[greenup_doy]
            # add one timestep at midpoint between greenup and peak
            mid = int((peak_doy-greenup_doy)/2)
            x[c,2] = interped[greenup_doy+mid]
            # peak
            x[c,3] = interped[peak_doy]
            # add one timestep at midpoint between peak and senescence
            mid = int((sen_doy-peak_doy)/2)
            x[c,4] = interped[peak_doy+mid]
            # senescence
            x[c,5] = interped[sen_doy]
            # add one timestep 5 observations after senescence
            x[c,6] = interped[sen_doy+5]
        else:
            x[c,0] = interped[greenup_doy]
            if timesteps > 1:
                x[c,1] = interped[peak_doy]
            if timesteps > 2:
                x[c,2] = interped[sen_doy]
    return x

def date_to_doy(dates):
    return np.array([date.timetuple().tm_yday for date in dates])

def doy_to_date(doy, year):
    return (np.asarray(year, dtype='datetime64[Y]'))+(np.asarray(doy, dtype='timedelta64[D]')-1)

def clean_nans(x, threshold=2):
    # Count nans along time axis
    valid_count = x.notnull().sum(dim='time')
    # keep pixels that at least 2 valid (non-nan) entries,
    # but otherwise set all values to 0
    x = x.where(valid_count >= threshold, 0)
    return x

def get_time(dates):
    return pd.DatetimeIndex([pd.Timestamp(date) for date in dates])

def main(datadir, phenodir, tile_id, year, timesteps, outdir, augment=False, usedask=False):
    bands = ['blue', 'green', 'red', 'nir_narrow', 'swir1', 'swir2']
    # Load the phenology days of interest (DOIs)
    dois = np.load(os.path.join(phenodir, '%s_%s_phenology.npy' % (tile_id, year))).astype(np.uint16)
    print(dois)
    # If we want to speed up computation using dask...
    if usedask:
        X = np.zeros([HEIGHT*WIDTH, len(bands), timesteps])
        # Load the time series for one band
        for b, band in enumerate(bands):
            print('Processing band %s' % band)
            tseries, doys = load_time_series(datadir, band, tile_id, year)
            # Convert the DOYs to dates
            dates = (np.asarray(np.full(len(doys), int(year)), dtype='datetime64[Y]')-1970)+(np.asarray(doys, dtype='timedelta64[D]')-1)
            # Start the dask client 
            client = Client(threads_per_worker=4, n_workers=5)
            client.cluster.scale(15)
            print('Started dask client')
            chunk_size = 300
            delayed = []
            series_xr = dask.delayed(xr.DataArray(tseries.reshape(tseries.shape[0]*tseries.shape[1], 
                                     tseries.shape[2])))
            for chunk in range(int(WIDTH*HEIGHT/chunk_size)):
                x = dask.delayed(process_chunk)(series_xr[chunk*chunk_size:chunk*chunk_size+chunk_size], 
                                                bands, dois, doys.astype(np.uint16), year)
                delayed.append(x)
            # Build and execute graph
            futures = dask.persist(*delayed)
            results = dask.compute(*futures)
            chunked_inputs = np.array(results)
            # Put them back into the proper shape
            for chunk in range(chunked_inputs.shape[0]):
                X[chunk*chunk_size:chunk*chunk_size+chunk_size,b] = chunked_inputs[chunk]
            # Delete the data from RAM
            del delayed
        # reshape again
        X = np.reshape(X, [HEIGHT, WIDTH, len(bands), timesteps])
    else:
        # Load the time series for each band
        tseries = {}
        for b in bands:
            tseries[b], doys = load_time_series(datadir, b, tile_id, year)
            tseries[b] = xr.DataArray(tseries[b], dims=['x', 'y', 'time'])
            # Convert the DOYs to dates
            dates = (np.asarray(np.full(len(doys), int(year)), dtype='datetime64[Y]')
                      - 1970) + (np.asarray(doys, dtype='timedelta64[D]')-1)
            time = xr.Variable('time', get_time(dates))
            tseries[b].coords['time'] = time
            print(tseries[b])
            # Set nans to 0 before interpolation
            print('Setting all NaN time series to 0')
            tseries[b] = clean_nans(tseries[b])

        # TODO: make this faster using multi-dimensional selection with xarray
        for r in range(HEIGHT):
            for c in range(WIDTH):
                for b, band in enumerate(bands):
                    # If they are all zero, there were too few observations to select
                    if np.all(dois[r,c] == 0):
                        print('skipping all zero DOYs: %d, %d' % (r,c))
                        tseries[band][r,c] = np.zeros(tseries[band][r,c].shape)

        print('Interpolating and resampling to 5D')
        # TODO: can make this even faster by making band a dimension in the DataArray
        for b, band in enumerate(bands):
            # resample and interpolate
            tseries[band] = tseries[band].interpolate_na(
                                            dim='time',
                                            method='linear',
                                            fill_value='extrapolate').resample(time='5D').mean()

        # Create an array to hold the data
        if augment:
            if timesteps != 3:
                print('Invalid number of timesteps for augment option.')
                sys.exit(0)
            X = np.zeros([HEIGHT, WIDTH, len(bands), timesteps + 4])
        else:
            X = np.zeros([HEIGHT, WIDTH, len(bands), timesteps])
            print(X.shape)

        print('Selecting nearest observations')
        # Select the observations corresponding with each input
        for b, band in enumerate(bands):
            for y in range(dois.shape[1]): # go through each row (y)
                print('row', y)
                greenup = [doy_to_date(dois[i,y,0], year) for i in range(dois.shape[1])]
                X[:,y,b,0] = tseries[band].isel(x=xr.DataArray(range(HEIGHT),dims='z'),
                                                y=y).sel(time=xr.DataArray(greenup, dims='z'),
                                                         method='nearest')
                if timesteps > 1:
                    peak = [doy_to_date(dois[i,y,1], year) for i in range(dois.shape[1])]
                    X[:,y,b,1] = tseries[band].isel(x=xr.DataArray(range(HEIGHT),dims='z'),
                                                    y=y).sel(time=xr.DataArray(peak, dims='z'),
                                                             method='nearest')
                if timesteps > 2:
                    senes = [doy_to_date(dois[i,y,2], year) for i in range(dois.shape[1])]
                    X[:,y,b,2] = tseries[band].isel(x=xr.DataArray(range(HEIGHT),dims='z'),
                                                    y=y).sel(time=xr.DataArray(senes, dims='z'),
                                                             method='nearest')

        # for r in range(HEIGHT):
        #     print('row', r)
        #     for c in range(WIDTH):
        #         for b, band in enumerate(bands):
        #             # If they are all zero, there were too few observations to select
        #             # if np.all(dois[r,c]) == 0:
        #             #     print('skipping all zero DOYs: %d, %d' % (r,c))
        #             #     continue
        #             # # also skip if the time series is all nan (and somehow wasn't caught in the last step)
        #             # elif np.all(np.isnan(tseries[band][r,c])):
        #             #     print('skipping all NaN time series: %d, %d' % (r,c))
        #             #     continue
        #             # linearly interpolate to fill nans and resample to 5D intervals
        #             #interped = pd.Series(tseries[band][r,c], index=dates).interpolate(limit_direction='both')
        #             #interped = interped.resample('5D').pad()
        #             # Load the greenup reflectances
        #             #greenup_doy = np.where(date_to_doy(interped.index)==dois[r,c,0])[0][0]
        #             #print(greenup_doy)
        #             greenup_doy = np.argsort(np.abs(doys-dois[r,c,0]))[0]
        #             if timesteps > 1:
        #                 # Load the peak reflectances
        #                 #peak_doy = np.where(date_to_doy(interped.index)==dois[r,c,1])[0][0]
        #                 peak_doy = np.argsort(np.abs(doys-dois[r,c,1]))[0]
        #             if timesteps > 2:
        #                 # Load the senescence reflectances
        #                 #sen_doy = np.where(date_to_doy(interped.index)==dois[r,c,2])[0][0]
        #                 sen_doy = np.argsort(np.abs(doys-dois[r,c,2]))[0]
        #             # Store them in the input array
        #             if augment:
        #                 # add one timestep 5 observations before greenup
        #                 X[r,c,b,0] = tseries[band][r,c][greenup_doy-5]
        #                 # greenup
        #                 X[r,c,b,1] = tseries[band][r,c][greenup_doy]
        #                 # add one timestep at midpoint between greenup and peak
        #                 mid = int((peak_doy-greenup_doy)/2)
        #                 X[r,c,b,2] = tseries[band][r,c][greenup_doy+mid]
        #                 # peak
        #                 X[r,c,b,3] = tseries[band][r,c][peak_doy]
        #                 # add one timestep at midpoint between peak and senescence
        #                 mid = int((sen_doy-peak_doy)/2)
        #                 X[r,c,b,4] = tseries[band][r,c][peak_doy+mid]
        #                 # senescence
        #                 X[r,c,b,5] = tseries[band][r,c][sen_doy]
        #                 # add one timestep 5 observations after senescence
        #                 X[r,c,b,6] = tseries[band][r,c][sen_doy+5]
        #             else:
        #                 X[r,c,b,0] = tseries[band][r,c][greenup_doy]
        #                 if timesteps > 1:
        #                     X[r,c,b,1] = tseries[band][r,c][peak_doy]
        #                 if timesteps > 2:
        #                     X[r,c,b,2] = tseries[band][r,c][sen_doy]
    # Save the inputs
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    np.save(os.path.join(outdir, 'pheno_timeseries_%s_%s.npy' % (tile_id, year)), X)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, 
                                     description='Assemble inputs based on detected phenology')
    parser.add_argument('--datadir', default='/gpfs/data1/cmongp1/GEOGLAM/Input/field_data/hls/tif', 
                                     help='directory containing raster data')
    parser.add_argument('--phenodir', default='/gpfs/data1/cmongp1/hkerner/us-corn/data/phenology', 
                                     help='directory containing phenology data')
    parser.add_argument('--tile_id', help='ID of HLS tile to process (e.g., 16TBL)')
    parser.add_argument('--year', choices=['2016', '2017', '2018', '2019', '2020', '2021'], help='year of data')
    parser.add_argument('--timesteps', type=int, default=3, help='number of phenology stages to detect (1, 2, or 3)')
    parser.add_argument('--outdir', help='directory to store output products in')
    parser.add_argument('--augment', action='store_true', help='add timesteps to augment each stage')
    parser.add_argument('--usedask', action='store_true', help='Use dask to speed up computation')
    # TODO: add option to include SAR data

    args = parser.parse_args()
    main(**vars(args))