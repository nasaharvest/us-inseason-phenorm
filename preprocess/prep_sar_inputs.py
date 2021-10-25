# Hannah Kerner
# July 27, 2020
# Align SAR data with HLS data
# Needs to be run using uscorn venv, otherwise issues with GDAL

import numpy as np
import pandas as pd
import xarray as xr 
import os
from glob import glob
from subprocess import call

IMG_DIM = 3660

def doy_to_date(doy, year):
    return (np.asarray(year, dtype='datetime64[Y]')-1970)+(np.asarray(doy, dtype='timedelta64[D]')-1)

def open_img(path, band):
    chunks = {'band': 3, 'x': 1000, 'y': 1000}
    img = xr.open_rasterio(path, chunks=chunks)

    if band == 'vv':
        return img[0].load()
    elif band == 'vh':
        return img[1].load()
    elif band == 'ia':
        return img[2].load()

def get_time(dates):
    return pd.DatetimeIndex([pd.Timestamp(date) for date in dates])

def main(sardir, phenology, tile_id, year, outdir, timesteps, ia=False):
    # Load paths to clipped rasters
    m_paths = sorted(glob(os.path.join(sardir, 'merged_by_date', '%s_%s*_clipped.tif' % (tile_id, year))))
    print(m_paths)
    # Load the dates
    dates = sorted([p.split('_')[-2] for p in m_paths])
    time = xr.Variable('time', get_time(dates))
    # Load the clipped data as an xarray time series
    vh_series = xr.concat([open_img(p, 'vh') for p in m_paths], dim=time, join='exact')
    vv_series = xr.concat([open_img(p, 'vv') for p in m_paths], dim=time, join='exact')
    if ia:
        ia_series = xr.concat([open_img(p, 'ia') for p in m_paths], dim=time, join='exact')

    # Load the phenology DOIs
    dois = np.load(phenology)

    vv_dois = np.zeros(dois.shape)
    vh_dois = np.zeros(dois.shape)
    if ia:
        ia_dois = np.zeros(dois.shape)

    for y in range(vv_dois.shape[0]): # go through each row (y)
        # Convert greenup DOI to pd.Datetime
        greenup = [doy_to_date(dois[y,x,0], year) for x in range(dois.shape[1])]
        # Convert peak DOI to pd.Datetime
        peak = [doy_to_date(dois[y,x,1], year) for x in range(dois.shape[1])]
        if timesteps == 3:
            # Convert senescence DOI to pd.Datetime
            senes = [doy_to_date(dois[y,x,2], year) for x in range(dois.shape[1])]
        # get the closest SAR observation for each date
        vv_dois[y,:,0] = vv_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(greenup, dims='z'), method='nearest')
        vv_dois[y,:,1] = vv_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(peak, dims='z'), method='nearest')
        if timesteps == 3:
            vv_dois[y,:,2] = vv_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(senes, dims='z'), method='nearest')
        # and for VH
        vh_dois[y,:,0] = vh_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(greenup, dims='z'), method='nearest')
        vh_dois[y,:,1] = vh_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(peak, dims='z'), method='nearest')
        if timesteps == 3:
            vh_dois[y,:,2] = vh_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(senes, dims='z'), method='nearest')
        if ia:
            # and for IA
            ia_dois[y,:,0] = ia_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(greenup, dims='z'), method='nearest')
            ia_dois[y,:,1] = ia_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(peak, dims='z'), method='nearest')
            if timesteps == 3:
                ia_dois[y,:,2] = ia_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(senes, dims='z'), method='nearest')

    # Save the images to a numpy array
    np.save(os.path.join(outdir, 'vv_timeseries_%s_%s.npy' % (tile_id, year)), vv_dois)
    np.save(os.path.join(outdir, 'vh_timeseries_%s_%s.npy' % (tile_id, year)), vh_dois)
    if ia:
        np.save(os.path.join(outdir, 'ia_timeseries_%s_%s.npy' % (tile_id, year)), ia_dois)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, 
                                     description='Assemble inputs based on detected phenology')
    parser.add_argument('--sardir', help='directory containing aligned SAR data')
    parser.add_argument('--phenology', help='path to detected phenology raster for this tile')
    parser.add_argument('--tile_id', help='ID of HLS tile to process (e.g., 16TBL)')
    parser.add_argument('--year', choices=['2016', '2017', '2018', '2019', '2020'], help='year of data')
    parser.add_argument('--outdir', help='directory to store output products in')
    parser.add_argument('--timesteps', type=int, help='Number of timesteps to use')
    parser.add_argument('--ia', action='store_true', help='Data includes incidence angle (IA) band')

    args = parser.parse_args()
    main(**vars(args))