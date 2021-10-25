# Hannah Kerner
# July 6, 2020

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import utils
import rasterio as rio

# random seeds
SEED = 488
np.random.seed(SEED)
# class labels
CORN = 0
SOYBEAN = 1
OTHER = 2
N_CLASSES = 3
# size of training and validation subsets
TRAINSIZE = 100000
VALSIZE = 10000

        

def shuffle_together(x_px, x_pa, y, ds, dp):
    perm = np.random.permutation(range(x_px.shape[0]))
    return x_px[perm], x_pa[perm], y[perm], ds[perm], dp[perm]

def subset_data(x_px, x_pa, y, ds, dp, n):
    # Shuffle the data first
    x_px, x_pa, y, ds, dp = shuffle_together(x_px, x_pa, y, ds, dp)
    x_px = x_px[:n]
    x_pa = x_pa[:n]
    y = y[:n]
    ds = ds[:n]
    dp = dp[:n]
    return x_px, x_pa, y, ds, dp


def main(traindir, labeldir, phenodir, tile_id, start_year, end_year, pred_year, homogmask_dir, timesteps,
         patchdim, savedir, hlsimg, ndvi=False, bsi=False, evi=False, lswi=False, ndwi=False, gcvi=False,
         vv=False, vh=False, ia=False, use_cached=False):
    
        
    if use_cached:
        # Load saved data
        X_px = np.load(os.path.join(savedir, 'train_X_px.npy'))
        X_pa = np.load(os.path.join(savedir, 'train_X_pa.npy'))
        Y = np.load(os.path.join(savedir, 'train_Y.npy'))
        delta_S = np.load(os.path.join(savedir, 'train_delta_S.npy'))
        delta_P = np.load(os.path.join(savedir, 'train_delta_P.npy'))
        inds = np.load(os.path.join(savedir, 'train_inds.npy'))
    else:
        # Make lists to hold the data for each year
        X_px = {}
        X_pa = {}
        Y = {}
        delta_S = {}
        delta_P = {}
        indices = {}
        # Load the training data for each year
        for year in range(start_year, end_year+1):
            x, valid_mask = utils.load_data(traindir, year, tile_id,
                                      ndvi, bsi, evi, lswi, ndwi, gcvi, 
                                      vv, vh, ia)
            y = utils.load_labels(labeldir, year, tile_id)
            dois = utils.load_dois(phenodir, year, tile_id)
            if timesteps < 3:
                delta_s = np.zeros(dois[...,0].shape)
            else:
                delta_s = dois[...,2]-dois[...,0]
            delta_p = dois[...,1]-dois[...,0]
            homog_mask = np.load(os.path.join(homogmask_dir, 'cdl_labels_%s_%s_homogmask_5x5.npy' % (tile_id, year)))
            # Make inputs in the format they go to the model
            x_px, x_pa, y, delta_s, delta_p, inds = utils.make_inputs(x, y, delta_s, delta_p, 
                                                          homog_mask, valid_mask, patchsize=patchdim, timesteps=timesteps)
            print(x_px.shape)
            print(x_pa.shape)

            # Add to the dictionary for all data
            X_px[year] = x_px
            X_pa[year] = x_pa
            Y[year] = y
            delta_S[year] = delta_s
            delta_P[year] = delta_p
            indices[year] = inds

        # Combine the inputs for all years
        X_px = np.concatenate([X_px[year] for year in X_px.keys()], axis=0)
        X_pa = np.concatenate([X_pa[year] for year in X_pa.keys()], axis=0)
        Y = np.concatenate([Y[year] for year in Y.keys()], axis=0)
        delta_S = np.concatenate([delta_S[year] for year in delta_S.keys()], axis=0)
        delta_P = np.concatenate([delta_P[year] for year in delta_P.keys()], axis=0)
        indices = np.concatenate([indices[year] for year in indices.keys()], axis=0)

    print("data shapes")
    print(X_px.shape)
    print(X_pa.shape)
    print(Y.shape)
    print(delta_S.shape)
    print(delta_P.shape)

    # Subset the data
    X_px, X_pa, Y, delta_S, delta_P = subset_data(X_px, X_pa, Y, delta_S, delta_P, TRAINSIZE)

    print("data shapes after subset")
    print(X_px.shape)
    print(X_pa.shape)
    print(Y.shape)
    print(delta_S.shape)
    print(delta_P.shape)

    # Limit the timesteps if necessary
    if timesteps < 3 and X_px.shape[-2] != timesteps:
        X_px = X_px[:,:timesteps]
        X_pa = X_pa[...,:timesteps*X_px.shape[-1]]

    # Flatten the data
    X_px = np.reshape(X_px, [X_px.shape[0], X_px.shape[1]*X_px.shape[2]])
    print('Flattened X_px shape:', X_px.shape)
    X_pa = np.reshape(X_pa, [X_pa.shape[0], X_pa.shape[1]*X_pa.shape[2]*X_pa.shape[3]])
    print('Flattened X_pa shape:', X_pa.shape)
    # Construct input depending on timesteps
    if timesteps == 3:
        X_train = np.concatenate([X_px, X_pa, np.expand_dims(delta_S, 1)], axis=1)
    elif timesteps == 2: 
        X_train = np.concatenate([X_px, X_pa, np.expand_dims(delta_P, 1)], axis=1)
    elif timesteps == 1:
        X_train = np.concatenate([X_px, X_pa], axis=1)
    print('flattened training dataset shape')
    print(X_train.shape)  

    print('Training data labels: %d corn, %d soy, %d other' % (len(np.where(Y==CORN)[0]),
                                                                   len(np.where(Y==SOYBEAN)[0]),
                                                                   len(np.where(Y==OTHER)[0])))

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    clf = RandomForestClassifier(random_state=SEED, n_estimators=100)
    clf.fit(X_train, Y)

    # Load prediction data
    x, valid_mask = utils.load_data(traindir, pred_year, tile_id,
                                    ndvi, bsi, evi, lswi, ndwi, gcvi, 
                                    vv, vh, ia)
    dois = utils.load_dois(phenodir, pred_year, tile_id)
    # Calculate the delta between senescence and greenup
    if timesteps == 3:
        delta_s = dois[...,2]-dois[...,0]
    else:
        delta_s = np.zeros(dois[...,0].shape)
    # Calculate the delta between peak and greenup
    if timesteps > 1: 
        delta_p = dois[...,1]-dois[...,0]
    else:
        delta_p = np.zeros(dois[...,0].shape)
    # Make inputs in the format they go to the model
    x_px, x_pa, y, delta_s, delta_p, inds = utils.make_inputs(x=x, labels=None, delta_s=delta_s, delta_p=delta_p, 
                                                              valid_mask=valid_mask, 
                                                              patchsize=patchdim, 
                                                              dataset='predict', 
                                                              timesteps=timesteps)

    # Limit the timesteps if necessary
    if timesteps < 3 and x_px.shape[-2] != timesteps:
        x_px = x_px[:,:timesteps]
        x_pa = x_pa[...,:timesteps*x_px.shape[-1]]

    # Flatten the data
    x_px = np.reshape(x_px, [x_px.shape[0], x_px.shape[1]*x_px.shape[2]])
    x_pa = np.reshape(x_pa, [x_pa.shape[0], x_pa.shape[1]*x_pa.shape[2]*x_pa.shape[3]])

    if timesteps == 3:
        X_pred = np.concatenate([x_px, x_pa, np.expand_dims(delta_s, 1)], axis=1)
    elif timesteps == 2: 
        X_pred = np.concatenate([x_px, x_pa, np.expand_dims(delta_p, 1)], axis=1)
    elif timesteps == 1:
        X_pred = np.concatenate([x_px, x_pa], axis=1)

    print('flattened prediction dataset shape')
    print(X_pred.shape)

    # Scale the data
    X_pred = scaler.transform(X_pred)

    y_pred = clf.predict(X_pred).astype(np.uint8)
    print(y_pred.shape)

    # Save the predictions
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    np.save(os.path.join(savedir, 'rforest_preds.npy'), y_pred)

    # Save the predicted map as a raster
    filled = (np.ones([3660,3660])*255).astype(np.uint8)
    filled[3:-3,3:-3] = np.reshape(y_pred, [3654, 3654]).astype(np.uint8)
    with rio.open(hlsimg) as src:
        profile = src.profile
        profile['dtype'] = np.uint8
        profile['count'] = 1
        profile['driver'] = 'GTiff'
        profile['nodata'] = 255
        with rio.open(os.path.join(savedir, 'rforest_%s_%s.tif' % (tile_id, pred_year)), 'w', **profile) as dst:
            dst.write(np.expand_dims(filled, 0))

        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, 
                                     description='Train crop type classifiers')
    # Training data info
    parser.add_argument('--traindir', help='directory containing training data (numpy files)')
    parser.add_argument('--labeldir', help='directory containing training labels (numpy files)')
    parser.add_argument('--phenodir', default='/gpfs/data1/cmongp1/hkerner/us-corn/data/phenology', 
                                     help='directory containing phenology data')
    parser.add_argument('--tile_id', help='ID of HLS tile to train on (e.g., 16TBL)')
    parser.add_argument('--start_year', type=int, default=2017, help='start year of data to use for training')
    parser.add_argument('--end_year', type=int, default=2018, help='end year of data to use for training')
    parser.add_argument('--pred_year', type=int, default=2020, help='year of data to use for validation (will only use SW quadrant)')
    parser.add_argument('--homogmask_dir', help='directory containing homogeneity masks for filtering labels')
    parser.add_argument('--timesteps', type=int, default=3, help='number of timesteps to use in prediction')
    # Model info
    parser.add_argument('--patchdim', type=int, default=5, help='size of input patches for CNN')
    parser.add_argument('--savedir', help='directory to store results in')
    parser.add_argument('--hlsimg', help='path to an example HLS image we want to match the CDL raster to')
    # Band indices
    parser.add_argument('--ndvi', action='store_true', help='add NDVI band to input')
    parser.add_argument('--bsi', action='store_true', help='add BSI band to input')
    parser.add_argument('--evi', action='store_true', help='add EVI band to input')
    parser.add_argument('--lswi', action='store_true', help='add LSWI band to input')
    parser.add_argument('--ndwi', action='store_true', help='add NDWI band to input')
    parser.add_argument('--gcvi', action='store_true', help='add GCVI band to input')
    # SAR features
    parser.add_argument('--vv', action='store_true', help='add VV SAR band to input')
    parser.add_argument('--vh', action='store_true', help='add VH SAR band to input')
    parser.add_argument('--ia', action='store_true', help='add incidence angle SAR band to input')
    # other
    parser.add_argument('--use_cached', action='store_true', help='use cached training and validation data')

    args = parser.parse_args()
    main(**vars(args))