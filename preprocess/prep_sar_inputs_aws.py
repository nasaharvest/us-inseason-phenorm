# Hannah Kerner
# October 9, 2020
# Make growth stage normalized SAR inputs

import numpy as np
import pandas as pd
import xarray as xr 
import os
from glob import glob
from subprocess import call
import sys

import rasterio
from rasterio.vrt import WarpedVRT
from affine import Affine

IMG_DIM = 3660
HLS_RES = 30

def doy_to_date(doy, year):
    return (np.asarray(year, dtype='datetime64[Y]'))+(np.asarray(doy, dtype='timedelta64[D]')-1)

def open_img(path, band):
    print('Loading %s for %s' % (band, path))
    # Format the filename
    if band == 'vv':
        infile = os.path.join(path, 'Gamma0_VV.tif')
    elif band == 'vh':
        infile = os.path.join(path, 'Gamma0_VH.tif')
    elif band == 'ia':
        infile = os.path.join(path, 'local_incident_angle.tif')
    # Read and resample to 30m
    with rasterio.open(infile) as src:
        tr = src.profile['transform']
        dst_transform = Affine(HLS_RES, tr.b, tr.c, tr.d, -HLS_RES, tr.f)
        vrt_options = {
            'transform': dst_transform,
            'height': IMG_DIM,
            'width': IMG_DIM
        }
        with WarpedVRT(src, **vrt_options) as vrt:
            img = xr.open_rasterio(vrt)
    # Return with nodata (0) values set to nan for later interpolation
    return img.where(img != 0)

def get_time(dates):
    return pd.DatetimeIndex([pd.Timestamp(date) for date in dates])

def remove_dups(obs):
    doys = []
    _obs = []
    for ob in obs:
        doy = ob.split('_')[-3]
        if doy in doys:
            print('Skipping %s because we already have an observation on DOY %s' % (ob, doy))
            continue
        else:
            doys.append(doy)
            _obs.append(ob)
    return sorted(_obs, key=lambda x: x.split('_')[-3])

def clean_nans(x, threshold=2):
    # Count nans along time axis
    valid_count = x.notnull().sum(dim='time')
    # keep pixels that at least 2 valid (non-nan) entries, 
    # but otherwise set all values to 0
    x = x.where(valid_count >= threshold, 0)
    return x

def main(sardir, phenology, tile_id, year, outdir, timesteps, ia=False):
    # Load paths to observations, sorted by date
    m_paths = sorted(glob(os.path.join(sardir, tile_id, '*%s*%s*' % (year, tile_id))), 
                     key=lambda x: x.split('_')[-3])
    # TODO: would be better to combine duplicate observations
    m_paths = remove_dups(m_paths) 
    # Load the dates
    dates = sorted([p.split('_')[-3] for p in m_paths])
    time = xr.Variable('time', get_time(dates))
    # Load the clipped data as an xarray time series
    vh_series = xr.concat([open_img(p, 'vh') for p in m_paths], dim=time, join='exact')
    # If fewer than 2 valid entries, set all to 0
    vh_series = clean_nans(vh_series)
    vh_series = vh_series.interpolate_na(dim='time', fill_value='extrapolate')
    # do the same for VV
    vv_series = xr.concat([open_img(p, 'vv') for p in m_paths], dim=time, join='exact')
    # If fewer than 2 valid entries, set all to 0
    vv_series = clean_nans(vv_series)
    vv_series = vv_series.interpolate_na(dim='time', fill_value='extrapolate')
    if ia:
        ia_series = xr.concat([open_img(p, 'ia') for p in m_paths], dim=time, join='exact')
        ia_series = ia_series.interpolate_na(dim='time', fill_value='extrapolate')

    # Load the phenology DOIs
    dois = np.load(phenology).astype(np.uint16)
    print(dois)

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
        vv_dois[y,:,0] = vv_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(greenup, dims='z'), method='nearest')[:,0]
        vv_dois[y,:,1] = vv_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(peak, dims='z'), method='nearest')[:,0]
        if timesteps == 3:
            vv_dois[y,:,2] = vv_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(senes, dims='z'), method='nearest')[:,0]
        # and for VH
        vh_dois[y,:,0] = vh_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(greenup, dims='z'), method='nearest')[:,0]
        vh_dois[y,:,1] = vh_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(peak, dims='z'), method='nearest')[:,0]
        if timesteps == 3:
            vh_dois[y,:,2] = vh_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(senes, dims='z'), method='nearest')[:,0]
        if ia:
            # and for IA
            ia_dois[y,:,0] = ia_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(greenup, dims='z'), method='nearest')[:,0]
            ia_dois[y,:,1] = ia_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(peak, dims='z'), method='nearest')[:,0]
            if timesteps == 3:
                ia_dois[y,:,2] = ia_series.isel(x=xr.DataArray(range(IMG_DIM),dims='z'), y=y).sel(time=xr.DataArray(senes, dims='z'), method='nearest')[:,0]

    print(vv_dois)
    print(vh_dois)
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