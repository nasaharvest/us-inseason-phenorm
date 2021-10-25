# Hannah Kerner
# July 7, 2020

import numpy as np
import os
from glob import glob

import rasterio
from rasterio.windows import Window

# band order
BLUE = 0
GREEN = 1
RED = 2
NIR = 3
SWIR1 = 4
SWIR2 = 5
# image dims
HEIGHT = 3660
WIDTH = 3660
# EVI constants
G = 2.5
C1 = 6
C2 = 7.5
# Default DOYs to use at test time when there were fewer 
# than 3 data points and phenology couldn't be detected
PHENO_DEFAULT = [152, 211, 274]

def load_labels(labeldir, year, tile_id):
    y_path = os.path.join(labeldir, 'cdl_labels_%s_%s.npy' % (tile_id, year))
    y = np.load(y_path)
    return y

def load_dois(phenodir, year, tile_id):
    path = os.path.join(phenodir, '%s_%s_phenology.npy' % (tile_id, year))
    dois = np.load(path).astype(np.int16)
    return dois

def normalized_diff(b1, b2):
    num = b1-b2
    denom = b1+b2
    denom[np.where(denom==0)] = 0.00000001
    return num/denom

def doy_to_date(doy, year):
    mindate = np.array('%s-03-01' % str(year), dtype=np.datetime64)
    maxdate = np.array('%s-11-30' % str(year), dtype=np.datetime64)
    date = (np.asarray(year, dtype='datetime64[Y]')-1970)+(np.asarray(doy, dtype='timedelta64[D]')-1)
    if date > maxdate:
        return maxdate
    elif date < mindate:
        return mindate
    else:
        return date

def make_inputs(x, labels, delta_s, delta_p, homog_mask=None, valid_mask=None, patchsize=5, dataset='train', timesteps=3):
    # Initialize lists to hold the examples
    pixels = []
    patches = []
    y = []
    d_s = []
    d_p = []
    inds = []
    # Define buffer variables based on patch size
    w = int(patchsize/2)
    margin = w+patchsize%2
    # Iterate through pixels with buffer for patch size
    for i in range(margin, x.shape[0]-margin):
        for j in range(margin, x.shape[1]-margin):
            # Check if pixels are valid for training examples
            if dataset == 'train' and (np.any(homog_mask[i-w:i+w+1, j-w:j+w+1] != 1) or valid_mask[i,j] != 1):
                continue
            # Get the pixel representation
            pixel = x[i,j]
            pixels.append(pixel)
            # Get the patch representation
            patch = x[i-w:i+w+1, j-w:j+w+1]
            patch = np.reshape(patch, [patch.shape[0], patch.shape[1], patch.shape[2]*patch.shape[3]], order='F')
            patches.append(patch)
            if dataset != 'predict':
                # Store the label
                y.append(labels[i,j])  
            # Store the senescence-greenup deltas
            if timesteps < 3:
                d_s.append(0)
            else:
                d_s.append(delta_s[i,j])
            if timesteps < 2:
                d_p.append(0)
            else:
                d_p.append(delta_p[i,j])
            inds.append(np.array([i, j]))
    return np.array(pixels), np.array(patches), np.array(y), np.array(d_s), np.array(d_p), np.array(inds)

def make_inputs_l3h(x, labels, delta_s, delta_p, year, homog_mask=None, valid_mask=None, dataset='train', seed=None):
    # Initialize lists to hold the examples
    pixels = []
    patches = []
    y = []
    d_s = []
    d_p = []
    # Iterate through indices in L3h patches
    if dataset == 'test':
        patchfiles = sorted(glob('/gpfs/data1/cmongp1/hkerner/us-corn/data/l3h_patches/%s*%d.npy' % (dataset, year)))
    else:
        patchfiles = sorted(glob('/gpfs/data1/cmongp1/hkerner/us-corn/data/l3h_patches/%s*seed%d*%d.npy' % (dataset, seed, year)))
    print('Found %d L3H patches' % len(patchfiles))
    for f in patchfiles:
        if dataset == 'test':
            i = int(os.path.basename(f).split('_')[2])
            j = int(os.path.basename(f).split('_')[3])
        else:
            i = int(os.path.basename(f).split('_')[3])
            j = int(os.path.basename(f).split('_')[4])
        # Check if pixels are valid 
        if (dataset == 'train' or dataset == 'val') and ((homog_mask[i,j] != 1) or valid_mask[i,j] != 1):
            continue
        # Get the pixel representation
        pixel = x[i,j]
        # Load the patch for this index
        patch = np.load(f)
        # Check that the patch is valid, since some patches might have 0 pixels from the projection
        if dataset == 'train' and np.any(patch == 0):
            continue
        pixels.append(pixel)
        patches.append(patch)
        # Store the label
        y.append(labels[i,j])  
        # Store the senescence-greenup deltas
        d_s.append(delta_s[i,j])
        d_p.append(delta_p[i,j])
    return np.array(pixels), np.array(patches), np.array(y), np.array(d_s), np.array(d_p)

def add_sar_channel(X, band, path):
    _X = np.ndarray([HEIGHT, WIDTH, X.shape[2]+1, X.shape[3]])
    # copy the values from the original array
    _X[:,:,:X.shape[2],:] = X
    # Get the path of the corresponding SAR band
    if band=='vv' or band=='VV':
        sarpath = path.replace('pheno_timeseries', 'vv_timeseries')
    elif band=='vh' or band=='VH':
        sarpath = path.replace('pheno_timeseries', 'vh_timeseries')
    elif band=='ia' or band=='IA':
        sarpath = path.replace('pheno_timeseries', 'ia_timeseries')
    elif band=='alpha':
        sarpath = path.replace('pheno_timeseries', 'alpha_timeseries')
    elif band=='entropy':
        sarpath = path.replace('pheno_timeseries', 'entropy_timeseries')
    elif band=='anisotropy':
        sarpath = path.replace('pheno_timeseries', 'anisotropy_timeseries')
    elif band=='mchi_b':
        sarpath = path.replace('pheno_timeseries', 'mchib_timeseries')
    elif band=='mchi_g':
        sarpath = path.replace('pheno_timeseries', 'mchig_timeseries')
    elif band=='mchi_r':
        sarpath = path.replace('pheno_timeseries', 'mchir_timeseries')
    # Load the band
    sar = np.load(sarpath)
    # Truncate outliers
    if band != 'ia' and band != 'IA':
        sar[np.where(sar > np.mean(sar)+3*np.std(sar))] = np.mean(sar)+3*np.std(sar)
    # Add it to the array 
    _X[:,:,-1,:] = sar
    return _X

def add_bsi_channel(X):
    _X = np.ndarray([HEIGHT, WIDTH, X.shape[2]+1, X.shape[3]])
    # copy the values from the original array
    _X[:,:,:X.shape[2],:] = X
    # calculate values for BSI channel
    num = (X[:,:,SWIR1]+X[:,:,RED])-(X[:,:,NIR]+X[:,:,BLUE])
    denom = (X[:,:,SWIR1]+X[:,:,RED])+(X[:,:,NIR]+X[:,:,BLUE])
    denom[np.where(denom==0)] = 0.00000001
    _X[:,:,-1] = num / denom
    return _X

def add_evi_channel(X):
    _X = np.ndarray([HEIGHT, WIDTH, X.shape[2]+1, X.shape[3]])
    # copy the values from the original array
    _X[:,:,:X.shape[2],:] = X
    # calculate values for EVI channel
    num = G*(X[:,:,NIR]-X[:,:,RED])
    denom = X[:,:,NIR]+C1*X[:,:,RED]-C2*X[:,:,BLUE]+1
    denom[np.where(denom==0)] = 0.00000001
    _X[:,:,-1] = num / denom
    return _X

def add_lswi_channel(X):
    _X = np.ndarray([HEIGHT, WIDTH, X.shape[2]+1, X.shape[3]])
    # copy the values from the original array
    _X[:,:,:X.shape[2],:] = X
    # calculate LSWI
    num = X[:,:,NIR]-X[:,:,SWIR1]
    denom = X[:,:,NIR]+X[:,:,SWIR1]
    denom[np.where(denom==0)] = 0.00000001
    _X[:,:,-1] = num / denom
    return _X

def add_ndvi_channel(X):
    _X = np.ndarray([HEIGHT, WIDTH, X.shape[2]+1, X.shape[3]])
    # copy the values from the original array
    _X[:,:,:X.shape[2],:] = X
    # calculate NDVI
    num = X[:,:,NIR]-X[:,:,RED]
    denom = X[:,:,NIR]+X[:,:,RED]
    denom[np.where(denom==0)] = 0.00000001
    _X[:,:,-1] = num / denom
    return _X

def add_ndwi_channel(X):
    _X = np.ndarray([HEIGHT, WIDTH, X.shape[2]+1, X.shape[3]])
    # copy the values from the original array
    _X[:,:,:X.shape[2],:] = X
    # calculate NDWI
    num = X[:,:,GREEN]-X[:,:,SWIR1]
    denom = X[:,:,GREEN]+X[:,:,SWIR1]
    denom[np.where(denom==0)] = 0.00000001
    _X[:,:,-1] = num / denom
    return _X

def add_gcvi_channel(X):
    _X = np.ndarray([HEIGHT, WIDTH, X.shape[2]+1, X.shape[3]])
    # copy the values from the original array
    _X[:,:,:X.shape[2],:] = X
    # calculate GCVI
    num = X[:,:,NIR]
    denom = X[:,:,GREEN]
    denom[np.where(denom==0)] = 0.00000001
    _X[:,:,-1] = (num/denom)-1
    return _X

def band_list(ndvi, bsi, evi, lswi, ndwi, gcvi, vv, vh, ia, alpha, entropy, anisotropy, mchi_b, mchi_g, mchi_r):
    # Base bands that are always used
    bands=['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    # add them in same order as utils.load_data
    if vv:
        bands.append('vv')
    if vh:
        bands.append('vh')
    if ia: 
        bands.append('ia')
    if alpha:
        bands.append('alpha')
    if entropy:
        bands.append('entropy')
    if anisotropy: 
        bands.append('anisotropy')
    if mchi_b:
        bands.append('mchi_b')
    if mchi_g:
        bands.append('mchi_g')
    if mchi_r: 
        bands.append('mchi_r')
    if ndvi:
        bands.append('ndvi')
    if gcvi:
        bands.append('gcvi')
    if bsi: 
        bands.append('bsi')
    if lswi:
        bands.append('lswi')
    if evi:
        bands.append('evi')
    if ndwi: 
        bands.append('ndwi')
    return bands


def load_data(datadir, year, tile_id, 
              ndvi, bsi, evi, lswi, ndwi, gcvi, 
              vv, vh, ia, alpha=False, entropy=False, anisotropy=False, mchi_b=False, mchi_g=False, mchi_r=False,
              flatten=False):
    # Load the time series image data
    x_path = os.path.join(datadir, 'pheno_timeseries_%s_%s.npy' % (tile_id, year))
    print('Loading %s' % x_path)
    X = np.load(x_path)
    # Identify invalid data (all zeros)
    # TODO: likely a faster way to do this
    valid_mask = np.ones([HEIGHT, WIDTH])
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if np.all(X[i,j]==0):
                valid_mask[i,j] = 0
    # Add SAR bands
    if vv:
        X = add_sar_channel(X, 'vv', x_path)
    if vh:
        X = add_sar_channel(X, 'vh', x_path)
    if ia:
        X = add_sar_channel(X, 'ia', x_path)
    if alpha:
        X = add_sar_channel(X, 'alpha', x_path)
    if entropy:
        X = add_sar_channel(X, 'entropy', x_path)
    if anisotropy:
        X = add_sar_channel(X, 'anisotropy', x_path)
    if mchi_b:
        X = add_sar_channel(X, 'mchi_b', x_path)
    if mchi_g:
        X = add_sar_channel(X, 'mchi_g', x_path)
    if mchi_r:
        X = add_sar_channel(X, 'mchi_r', x_path)
    # Add band indices
    if ndvi:
        X = add_ndvi_channel(X)
    if gcvi:
        X = add_gcvi_channel(X)
    if bsi:
        X = add_bsi_channel(X)
    if lswi:
        X = add_lswi_channel(X)
    if evi:
        X = add_evi_channel(X)
    if ndwi:
        X = add_ndwi_channel(X)
    # Make sure there are no nans
    X[np.where(np.isnan(X))] = 0
    # Optionally flatten the data matrix
    if flatten:
        X = np.reshape(X, X.shape[0]*X.shape[1], X.shape[2], X.shape[3])
    # Return the data matrix
    return X, valid_mask

# Return a list of adjacent tile IDs
def get_adjacent(tile_id):
    if tile_id == '15TUG':
        return ['15TTG', '15TVG', '15TTF', '15TUF', '15TVF', '14TQN', '15TUH', '15TVH']
    elif tile_id == '15TVG':
        return ['15TUG', '15TUF', '15TVF', '15TWF', '15TWG', '15TWH', '15TVH', '15TUH']
    elif tile_id == '15TWG':
        return ['15TVG', '15TVF', '15TWF', '15TXF', '15TXG', '15TXH', '15TWH', '15TVH']
    elif tile_id == '15TXG':
        return ['15TWG', '15TWF', '15TXF', '16TBL', '16TBM', '15TXH', '15TWH'] # 15TYH not processed
    elif tile_id == '15TTG':
        return ['15TTF', '15TUF', '15TUG', '15TUH', '14TQN', '14TPN'] # 14TPM, 14TPL not processed 
    elif tile_id == '15TTF':
        return ['15TTG', '15TUG', '15TUF'] # 14TPL, 14TPM, 14TPK, 15TTE, 15TUE not processed
    elif tile_id == '15TUF':
        return ['15TTF', '15TTG', '15TUG', '15TVG', '15TVF'] # 15TVE, 15TUE, 15TTE not processed
    elif tile_id == '15TVF':
        return ['15TUF', '15TUG', '15TVG', '15TWG', '15TWF', ] # 15TWE, 15TVE, 15TUE not processed
    elif tile_id == '15TWF':
        return ['15TVF', '15TVG', '15TWG', '15TXG', '15TXF', '15TXE'] # 15TWE, 15TVE not processed
    elif tile_id == '15TXF': 
        return ['15TWF', '15TWG', '15TXG', '16TBM', '16TBL', '15TYE', '15TXE'] # 15TWE not processed
    elif tile_id == '15TXE':
        return ['15TWF', '15TXF', '16TBL', '15TYE', '16SBJ', '15SXD'] # 15TWE, 15SWD not processed 
    elif tile_id == '14TQN':
        return ['14TPN', '14TPP', '14TQP', '15TUJ', '15TUH', '15TUG', '15TTG'] # 14TPM not processed
    elif tile_id == '15TUH':
        return ['14TQN', '14TQP', '15TUJ', '15TVJ', '15TVH', '15TVG', '15TUG', '15TTG']
    elif tile_id == '15TVH':
        return ['15TUH', '15TUJ', '15TVJ', '15TWJ', '15TWH', '15TWG', '15TVG', '15TUG']
    elif tile_id == '15TWH':
        return ['15TVH', '15TVJ', '15TWJ', '15TXJ', '15TXH', '15TXG', '15TWG', '15TVG']
    elif tile_id == '15TXH':
        return ['15TWH', '15TWJ', '15TXJ', '16TBM', '15TXG', '15TWG'] # 15TYJ, 15TYH not processed
    elif tile_id == '14TPP':
        return ['14TQP', '14TQN', '14TPN'] # 14TNP, 14TNQ, 14TPQ, 14TQQ, 14TNN not processed
    elif tile_id == '14TQP':
        return ['14TPP', '15TUJ', '15TUH', '14TQN', '14TPN'] # 14TPQ, 14TQQ, 15TUK not processed
    elif tile_id == '15TUJ':
        return ['14TQP', '15TVJ', '15TVH', '15TUH', '14TQN'] # 14TQQ, 15TUK, 15TVK not processed
    elif tile_id == '15TVJ':
        return ['15TUJ', '15TWJ', '15TWH', '15TVH', '15TUH'] # 15TUK, 15TVK, 15TWK not processed
    elif tile_id == '15TWJ':
        return ['15TVJ', '15TXJ', '15TXH', '15TWH', '15TVH'] # 15TVK, 15TWK, 15TXK not processed
    elif tile_id == '15TXJ':
        return ['15TWJ', '15TXH', '15TWH', '15TYH'] # 15TWK, 15TXK, 15TYK, 15TYJ  not processed
    elif tile_id == '16TBM':
        return ['15TXG', '15TXH', '16TBL', '15TXF', '15TYH', '16TCN', '16TCM', '16TCL']
    elif tile_id == '14TPN':
        return ['14TPP', '14TQP', '14TQN', '15TTG'] # 14TNN, 14TNP, 14TPM, 14TNM not processed
    elif tile_id == '16TBL':
        return ['16TBM', '15TXG', '15TXF', '15TXE', '15TYE', '16TBK', '16TCK'] # 16TCL, 16TCM not processed
    elif tile_id == '16SBJ':
        return ['15SXD', '15TXE', '15TYE', '16TBK', '16TCK', '16SCJ', '16SCH'] # 15SYD, 16SBH not processed
    elif tile_id == '15TYE':
        return ['15TXE', '15TXF', '15SXD', '16TBK', '16TBL', '16TCK', '16SBJ', '16SCJ'] # 16TCL not processed
    elif tile_id == '16TBK':
        return ['15TYE', '15TXE', '15TXF', '15SXD', '16TBL', '16TCK', '16SBJ', '16SCJ', '16TCL'] 
    elif tile_id == '16TCK':
        return ['15TYE', '15SXD', '16TBK', '16TBL', '16SBJ', '16SCJ', '16TCL'] 
    elif tile_id == '16SCJ':
        return ['16SBJ', '16TBK', '16TCK', '16TDK', '16SDJ', '16SDH', '16SCH', '16SBH'] 
    elif tile_id == '16SCH':
        return ['16SBJ', '16SCJ', '16SDH', '16SBH', '16SDJ', '16SDG', '16SCG', '16SBG', '16SBH'] 
    elif tile_id == '16TDK':
        return ['16TCK', '16TDL', '16TEL', '16TEK', '16SDJ', '16SCJ', '16SBJ', '16TBK', '16TCL'] # 16SEJ not processed
    elif tile_id == '16TDL':
        return ['16TEL', '16TEK', '16TDK', '16TCK', '16TCL', '16TCM', '16TDM'] #  16TEM not processed
    elif tile_id == '16TEK':
        return ['16TDK', '16TDL', '16TEL', '16SDJ'] # 16TFL, 16TFK, 16SFJ, 16SEJ not processed
    elif tile_id == '16TEL':
        return ['16TDL', '16TEK', '16TDK', '16TDM'] # 16TEM, 16TFM, 16TFL, 16TFK not processed
    elif tile_id == '16SDJ':
        return ['16SCJ', '16TCK', '16TDK', '16TEK', '16SDH', '16SCH'] # 16SEJ, 16SEH not processed
    elif tile_id == '16SDH':
        return ['16SCH', '16SCJ', '16SDJ', '16SDG', '16SCG'] # 16SEJ, 16SEH, 16SEG, 
    elif tile_id == '15SXD':
        return ['15TXE', '15TYE', '16TBK', '16SBJ', '16SBH'] # 15SWD, 15TWE, 15SXC, 15SWC not processed
    elif tile_id == '15TYE':
        return ['16TCK', '16TBK', '16TBL', '16SCJ', '16SBJ', '15TXE', '15TXF', '15SXD', '16TCL'] 
    elif tile_id == '15TYH':
        return ['15TXH', '15TXJ', '16TCN', '16TCM', '16TBM', '15TXG'] # 15TYJ, 16TCP not processed 
    elif tile_id == '16TCN':
        return ['15TYH', '16TDN', '16TDM', '16TBM', '16TCM'] # 15TYJ, 16TCP, 16TDP not processed
    elif tile_id == '16TDN':
        return ['16TCN', '16TDM', '16TCM'] # 16TCP, 16TDP, 16TEP, 16TEN, 16TEM not processed
    elif tile_id == '16TCM':
        return ['15TYH', '16TBM', '16TCN', '16TDM', '16TDL', '16TCL', '16TBL', '16TDN']
    elif tile_id == '16TDM':
        return ['16TCM', '16TCN', '16TDN', '16TEL', '16TDL', '16TCL'] # 16TEN, 16TEM not processed
    elif tile_id == '16TCL':
        return ['16TBL', '16TBM', '16TCM', '16TDM', '16TDL', '16TDK', '16TCK', '16TBK']
    elif tile_id == '16SBH':
        return ['16SBJ', '16SCJ', '16SBG', '16SCH', '15SXD', '16SCG'] # 15SYC, 15SXC, 15SYD, 15SYB, 15SXB not processed
    elif tile_id == '16SDG':
        return ['16SCG', '16SCH', '16SDH'] # 16SEH, 16SEG, 16SEF, 16SDF, 16SCF not processed
    elif tile_id == '16SCG':
        return ['16SBG', '16SBH', '16SCH', '16SDH', '16SDG'] # 16SDF, 16SCF, 16SBF not processed
    elif tile_id == '16SBG':
        return ['16SCH', '16SCG'] # 15SYB, 15SXB, 15SXC, 15SYC, 16SCF, 16SBF, 15SYA, 15SXA not processed
    elif tile_id == '15SYD':
        return ['16TBK', '16TCK', '16SCJ', '16SCH', '16SBH', '15SXD', '15TXE', '15TYE']

