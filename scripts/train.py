# Hannah Kerner
# July 6, 2020

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

import models
import utils

# random seeds
# SEEDS = [42, 64, 4, 488, 34]
SEEDS = [488]
# class labels
CORN = 0
SOYBEAN = 1
OTHER = 2
N_CLASSES = 3
# size of training and validation subsets
TRAINSIZE = 100000
VALSIZE = 10000

def split_val(x_px, x_pa, y, ds, dp, inds):
    # Create lists to hold the data
    X_px = {'val': [], 'test': []}
    X_pa = {'val': [], 'test': []}
    Y = {'val': [], 'test': []}
    dS = {'val': [], 'test': []}
    dP = {'val': [], 'test': []}
    indices = {'val': [], 'test': []}
    # Separate based on indices (val=southwest, test=rest)
    for i in range(x_px.shape[0]):
        # southwest quadrant goes in validation
        if inds[i][0] > utils.HEIGHT/2. and inds[i][1] < utils.WIDTH/2.:
            X_px['val'].append(x_px[i])
            X_pa['val'].append(x_pa[i])
            Y['val'].append(y[i])
            dS['val'].append(ds[i])
            dP['val'].append(dp[i])
            indices['val'].append(inds[i])
        # everything else goes in test - TODO if we want to test at the end later
        # else:
        #     X_px['test'].append(x_px[i])
        #     X_pa['test'].append(x_pa[i])
        #     Y['test'].append(y[i])
        #     dS['test'].append(ds[i])
        #     dP['test'].append(dp[i])

    # Turn the lists into numpy arrays
    for k in X_px.keys():
        X_px[k] = np.array(X_px[k])
        X_pa[k] = np.array(X_pa[k])
        Y[k] = np.array(Y[k])
        dS[k] = np.array(dS[k])
        dP[k] = np.array(dP[k])
        indices[k] = np.array(indices[k])

    return X_px['val'], X_pa['val'], Y['val'], dS['val'], dP['val'], indices['val']

# must have shape num samples x ...
def rand_inds(x, n):
    return np.random.randint(0, x.shape[0], size=n)

def balance_classes(X_px, X_pa, y, delta_s, delta_p, inds):
    # for each dataset
    for k in y.keys():
        # Get the minimum number of samples in a class
        n_corn = len(np.where(y[k]==CORN)[0])
        n_soy = len(np.where(y[k]==SOYBEAN)[0])
        n_other = len(np.where(y[k]==OTHER)[0])
        lim = np.min([n_corn, n_soy, n_other])
        # Select random indices
        corn_inds = rand_inds(X_px[k][np.where(y[k]==CORN)], lim)
        soy_inds = rand_inds(X_px[k][np.where(y[k]==SOYBEAN)], lim)
        other_inds = rand_inds(X_px[k][np.where(y[k]==OTHER)], lim)
        # Sub-sample the classes to the same number
        # TODO: this would be faster if we didn't recompute np.where each time
        X_px[k] = np.concatenate([X_px[k][np.where(y[k]==CORN)][corn_inds], X_px[k][np.where(y[k]==SOYBEAN)][soy_inds], X_px[k][np.where(y[k]==OTHER)][other_inds]])
        X_pa[k] = np.concatenate([X_pa[k][np.where(y[k]==CORN)][corn_inds], X_pa[k][np.where(y[k]==SOYBEAN)][soy_inds], X_pa[k][np.where(y[k]==OTHER)][other_inds]])
        delta_s[k] = np.concatenate([delta_s[k][np.where(y[k]==CORN)][corn_inds], delta_s[k][np.where(y[k]==SOYBEAN)][soy_inds], delta_s[k][np.where(y[k]==OTHER)][other_inds]])
        delta_p[k] = np.concatenate([delta_p[k][np.where(y[k]==CORN)][corn_inds], delta_p[k][np.where(y[k]==SOYBEAN)][soy_inds], delta_p[k][np.where(y[k]==OTHER)][other_inds]])
        inds[k] = np.concatenate([inds[k][np.where(y[k]==CORN)][corn_inds], inds[k][np.where(y[k]==SOYBEAN)][soy_inds], inds[k][np.where(y[k]==OTHER)][other_inds]])
        y[k] = np.concatenate([y[k][np.where(y[k]==CORN)][corn_inds], y[k][np.where(y[k]==SOYBEAN)][soy_inds], y[k][np.where(y[k]==OTHER)][other_inds]])
        
    return X_px, X_pa, y, delta_s, delta_p, inds

def shuffle_together(x_px, x_pa, y, ds, dp, inds):
    perm = np.random.permutation(range(x_px.shape[0]))
    return x_px[perm], x_pa[perm], y[perm], ds[perm], dp[perm], inds[perm]

def subset_data(x_px, x_pa, y, ds, dp, inds, n):
    # Shuffle the data first
    x_px, x_pa, y, ds, dp, inds = shuffle_together(x_px, x_pa, y, ds, dp, inds)
    x_px = x_px[:n]
    x_pa = x_pa[:n]
    y = y[:n]
    ds = ds[:n]
    dp = dp[:n]
    inds = inds[:n]
    return x_px, x_pa, y, ds, dp, inds

# Smooth the labels
def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels = labels * (1 - factor)
    labels = labels + (factor / labels.shape[1])
    return labels

# x must be dictionary with keys train, val
# each array must have shape numsamples x ...
def _standardize_array(x, savedir, name):
    # Calculate training data statistics
    mu = np.mean(x['train'], axis=0)
    std = np.std(x['train'], axis=0)
    # Save the training statistics for later
    np.save(os.path.join(savedir, '%s_train_mu.npy' % name), mu)
    np.save(os.path.join(savedir, '%s_train_std.npy' % name), std)
    # Apply standardization
    x['train'] = (x['train'] - mu)/std
    x['val'] = (x['val'] - mu)/std
    return x

def standardize(X_px, X_pa, delta_S, delta_P, savedir, seed):
    # Standardize all the features together
    X_px = _standardize_array(X_px, savedir, 'x_px_seed%d' % seed)
    X_pa = _standardize_array(X_pa, savedir, 'x_pa_seed%d' % seed)
    # Model immediately overfits when deltas scaled
    delta_S = _standardize_array(delta_S, savedir, 'delta_s_seed%d' % seed)
    delta_P = _standardize_array(delta_P, savedir, 'delta_p_seed%d' % seed)
    return X_px, X_pa, delta_S, delta_P

def plot_train_hist(hist, savedir, savename):
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    # Plot loss histories
    ax1.plot(hist.history['loss'], label='train', color='k')
    ax1.plot(hist.history['val_loss'], label='validation', color='k', linestyle='--')
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    # Plot accuracy histories
    ax2.plot(hist.history['accuracy'], label='train', color='k')
    ax2.plot(hist.history['val_accuracy'], label='validation', color='k', linestyle='--')
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    # Add experiment title
    fig.suptitle(savename)
    plt.savefig(os.path.join(savedir, '%s_trainhist.png' % savename))

def main(traindir, labeldir, phenodir, tile_id, start_year, end_year, val_year, homogmask_dir, rawdatapath, timesteps,
         model, savedir, seeds, patchdim, labelsmoothing=True, batchsize=4096, maxepochs=50,
         ndvi=False, bsi=False, evi=False, lswi=False, ndwi=False, gcvi=False,
         vv=False, vh=False, ia=False, alpha=False, entropy=False, anisotropy=False, mchi_b=False, mchi_g=False, mchi_r=False,
         use_cached=False, savemodels=False):
    # Create the save directory if it doesn't exist
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    
    if use_cached:
        # Load saved data
        _X_px = {}
        _X_pa = {}
        _Y = {}
        _delta_S = {}
        _delta_P = {}
        _inds = {}
        for k in ['train', 'val']:
            _X_px[k] = np.load(os.path.join(savedir, '%s_X_px.npy' % k))
            _X_pa[k] = np.load(os.path.join(savedir, '%s_X_pa.npy' % k))
            _Y[k] = np.load(os.path.join(savedir, '%s_Y.npy' % k))
            _delta_S[k] = np.load(os.path.join(savedir, '%s_delta_S.npy' % k))
            _delta_P[k] = np.load(os.path.join(savedir, '%s_delta_P.npy' % k))
            _inds[k] = np.load(os.path.join(savedir, '%s_inds.npy' % k))
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
                                      vv, vh, ia, alpha, entropy, anisotropy, mchi_b, mchi_g, mchi_r)
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
            # x_px, x_pa, y, delta_s, delta_p, inds = utils.make_inputs_l3h(x, y, dois, delta_s, delta_p, year, 
            #                                               homog_mask, valid_mask)
            # x_px, x_pa, y, delta_s, delta_p, inds = utils.make_inputs(x, y, delta_s, delta_p, 
            #                                                           os.path.join(rawdatapath, tile_id, str(year)), dois, 
            #                                                           homog_mask, valid_mask, patchsize=patchdim, 
            #                                                           bands=utils.band_list(ndvi, bsi, evi, lswi, ndwi, gcvi, vv, vh, ia, alpha, 
            #                                                                           entropy, anisotropy, mchi_b, mchi_g, mchi_r))
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

        # Store in a dictionary
        _X_px ={'train': X_px}
        _X_pa = {'train': X_pa}
        _Y = {'train': Y}
        _delta_S = {'train': delta_S}
        _delta_P = {'train': delta_P}
        _inds = {'train': indices}
        
        # Load validation data
        x_val, valid_mask_val = utils.load_data(traindir, val_year, tile_id,
                                      ndvi, bsi, evi, lswi, ndwi, gcvi, 
                                      vv, vh, ia, alpha, entropy, anisotropy, mchi_b, mchi_g, mchi_r)
        y_val = utils.load_labels(labeldir, val_year, tile_id)
        dois_val = utils.load_dois(phenodir, val_year, tile_id)
        if timesteps < 3:
            delta_s_val = np.zeros(dois_val[...,0].shape)
        else:
            delta_s_val = dois_val[...,2]-dois_val[...,0]
        delta_p_val = dois_val[...,1]-dois_val[...,0]
        homog_mask_val = np.load(os.path.join(homogmask_dir, 'cdl_labels_%s_%s_homogmask_5x5.npy' % (tile_id, val_year)))
        # Make inputs in the format they go to the model
        x_px_val, x_pa_val, y_val, delta_s_val, delta_p_val, inds_val = utils.make_inputs(x_val, y_val, delta_s_val, delta_p_val, 
                                                                                          homog_mask_val, valid_mask_val, patchsize=patchdim, timesteps=timesteps)

        print(inds_val.shape)

        # x_px_val, x_pa_val, y_val, delta_s_val, delta_p_val, inds_val = utils.make_inputs_l3h(x_val, y_val, dois_val, delta_s_val, delta_p_val, val_year,
        #                                                                                   homog_mask_val, valid_mask_val)
        # x_px_val, x_pa_val, y_val, delta_s_val, delta_p_val, inds_val = utils.make_inputs(x_val, y_val, delta_s_val, delta_p_val, 
        #                                                                                   os.path.join(rawdatapath, tile_id, str(val_year)), dois_val, 
        #                                                                                   homog_mask_val, valid_mask_val, patchsize=patchdim,
        #                                                                                   bands=utils.band_list(ndvi, bsi, evi, lswi, ndwi, gcvi, vv, vh, ia, alpha, 
        #                                                                                                   entropy, anisotropy, mchi_b, mchi_g, mchi_r))

        # Subset to only the SW quadrant
        _X_px['val'], _X_pa['val'], _Y['val'], _delta_S['val'], _delta_P['val'], _inds['val'] = split_val(x_px_val, x_pa_val, y_val, delta_s_val, delta_p_val, inds_val)
        print(_inds['val'].shape)
        # Cache the data
        for k in _X_px.keys():
            np.save(os.path.join(savedir, '%s_X_px.npy' % k), _X_px[k])
            np.save(os.path.join(savedir, '%s_X_pa.npy' % k), _X_pa[k])
            np.save(os.path.join(savedir, '%s_Y.npy' % k), _Y[k])
            np.save(os.path.join(savedir, '%s_delta_S.npy' % k), _delta_S[k])
            np.save(os.path.join(savedir, '%s_delta_P.npy' % k), _delta_P[k])
            np.save(os.path.join(savedir, '%s_inds.npy' % k), _inds[k])


    # Optionally repeat data selection and model training using N random seeds
    for i in range(seeds):
        # Set the random seed
        np.random.seed(SEEDS[i])
        print('Training with random seed %d' % SEEDS[i])

        # Copy into new variables so they don't change in the subsequent for loop
        X_px = {key: value[:] for key, value in _X_px.items()}
        X_pa = {key: value[:] for key, value in _X_pa.items()}
        Y = {key: value[:] for key, value in _Y.items()}
        delta_S = {key: value[:] for key, value in _delta_S.items()}
        delta_P = {key: value[:] for key, value in _delta_P.items()}
        inds = {key: value[:] for key, value in _inds.items()}
        # Balance the classes in the datasets
        X_px, X_pa, Y, delta_S, delta_P, inds = balance_classes(X_px, X_pa, Y, delta_S, delta_P, inds)
        # Subset the data
        X_px['train'], X_pa['train'], Y['train'], delta_S['train'], delta_P['train'], inds['train'] = subset_data(X_px['train'], 
                                                                                          X_pa['train'], 
                                                                                          Y['train'], 
                                                                                          delta_S['train'], 
                                                                                          delta_P['train'], 
                                                                                          inds['train'], 
                                                                                          n=TRAINSIZE)

        X_px['val'], X_pa['val'], Y['val'], delta_S['val'], delta_P['val'], inds['val'] = subset_data(X_px['val'], 
                                                                                  X_pa['val'], 
                                                                                  Y['val'], 
                                                                                  delta_S['val'], 
                                                                                  delta_P['val'], 
                                                                                  inds['val'], 
                                                                                  n=VALSIZE)

        # Save the training and validation indices used for each seed
        np.save(os.path.join(savedir, 'train_inds_seed%d' % SEEDS[i]), inds['train']) 
        np.save(os.path.join(savedir, 'val_inds_seed%d' % SEEDS[i]), inds['val'])

        # Check labels are balanced
        print('Training data labels: %d corn, %d soy, %d other' % (len(np.where(Y['train']==CORN)[0]),
                                                                   len(np.where(Y['train']==SOYBEAN)[0]),
                                                                   len(np.where(Y['train']==OTHER)[0])))
        print('Validation data labels: %d corn, %d soy, %d other' % (len(np.where(Y['val']==CORN)[0]),
                                                                     len(np.where(Y['val']==SOYBEAN)[0]),
                                                                     len(np.where(Y['val']==OTHER)[0])))

        # Swap temporal and spectral axes (LSTM wants channels time first)
        X_px['train'] = np.swapaxes(X_px['train'], 1, 2)
        X_px['val'] = np.swapaxes(X_px['val'], 1, 2)

        # Turn integer labels into one-hot labels
        Y['train'] = to_categorical(Y['train'], num_classes=N_CLASSES)
        Y['val'] = to_categorical(Y['val'], num_classes=N_CLASSES)
        # Optionally smooth labels
        if labelsmoothing:
            Y['train'] = smooth_labels(Y['train'])
            Y['val'] = smooth_labels(Y['val'])

        # Standardize data
        X_px, X_pa, delta_S, delta_P = standardize(X_px, X_pa, delta_S, delta_P, savedir, SEEDS[i])

        # Limit the timesteps if necessary
        if timesteps < 3 and X_px['train'].shape[-2] != timesteps:
            X_px['train'] = X_px['train'][:,:timesteps]
            X_px['val'] = X_px['val'][:,:timesteps]
            X_pa['train'] = X_pa['train'][...,:timesteps*X_px['train'].shape[-1]]
            X_pa['val'] = X_pa['val'][...,:timesteps*X_px['val'].shape[-1]]

        for k in X_px.keys():
            print(k)
            print(X_px[k].shape)
            print(X_pa[k].shape)
            print(Y[k].shape)
            print(delta_S[k].shape)
            print(delta_P[k].shape)

        # Set up a callback for early stopping
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9)

        # Load the model
        if model == 'lstm_cnn_delta':
            clf = models.lstm_cnn_delta(n_bands=X_px['train'].shape[-1], n_timesteps=X_px['train'].shape[-2], patchdim=patchdim)
            # Compile the model with loss function and optimizer
            clf.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
            # Train the model
            if timesteps == 3:
                hist = clf.fit(x=[X_px['train'], X_pa['train'], delta_S['train']], y=Y['train'], 
                               validation_data=([X_px['val'], X_pa['val'], delta_S['val']], Y['val']),
                               shuffle=True, epochs=maxepochs, batch_size=batchsize, callbacks=[earlystop])
            elif timesteps == 2:
                hist = clf.fit(x=[X_px['train'], X_pa['train'], delta_P['train']], y=Y['train'], 
                               validation_data=([X_px['val'], X_pa['val'], delta_P['val']], Y['val']),
                               shuffle=True, epochs=maxepochs, batch_size=batchsize, callbacks=[earlystop])
            elif timesteps == 1:
                print('Invalid number of timesteps for CNN-LSTM-delta')
        elif model == 'lstm_cnn':
            clf = models.lstm_cnn(n_bands=X_px['train'].shape[-1], n_timesteps=X_px['train'].shape[-2], patchdim=patchdim)
            # Compile the model with loss function and optimizer
            clf.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
            # Train the model
            hist = clf.fit(x=[X_px['train'], X_pa['train']], y=Y['train'], 
                               validation_data=([X_px['val'], X_pa['val']], Y['val']),
                               shuffle=True, epochs=maxepochs, batch_size=batchsize, callbacks=[earlystop])
        elif model == 'cnn':
            clf = models.cnn(n_bands=X_px['train'].shape[-1], n_timesteps=X_px['train'].shape[-2], patchdim=patchdim)
            # Compile the model with loss function and optimizer
            clf.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
            # Train the model
            hist = clf.fit(x=X_pa['train'], y=Y['train'], 
                           validation_data=(X_pa['val'], Y['val']),
                           shuffle=True, epochs=maxepochs, batch_size=batchsize, callbacks=[earlystop])
        elif model == 'lstm':
            clf = models.lstm(n_bands=X_px['train'].shape[-1], n_timesteps=X_px['train'].shape[-2])
            # Compile the model with loss function and optimizer
            clf.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
            # Train the model
            hist = clf.fit(x=X_px['train'], y=Y['train'], 
                           validation_data=(X_px['val'], Y['val']),
                           shuffle=True, epochs=maxepochs, batch_size=batchsize, callbacks=[earlystop])

        if savemodels:
            # Save the model
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            clf.save(os.path.join(savedir, '%s_tile=%s_train=%d-%d_epochs=%d_batchsize=%d_seed=%d.h5' 
                                            % (model, tile_id, start_year, end_year, len(hist.history['val_loss']), batchsize, SEEDS[i])))
            # Save the training and validation plots
            plot_train_hist(hist, savedir=savedir, savename='%s_tile=%s_train=%d-%d_epochs=%d_batchsize=%d_seed=%d' 
                                                             % (model, tile_id, start_year, end_year, len(hist.history['val_loss']), batchsize, SEEDS[i]))

        
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
    parser.add_argument('--val_year', type=int, default=2019, help='year of data to use for validation (will only use SW quadrant)')
    parser.add_argument('--homogmask_dir', help='directory containing homogeneity masks for filtering labels')
    parser.add_argument('--rawdatapath', help='directory containing raw (smooth) time series data',
                                         default='/gpfs/data1/cmongp1/GEOGLAM/Input/field_data/hls/smoothed')
    parser.add_argument('--timesteps', type=int, default=3, help='number of timesteps to use in prediction')
    # Model info
    parser.add_argument('--model', choices=['lstm_cnn_delta', 'cnn', 'lstm_cnn', 'lstm'], help='name of model to train')
    parser.add_argument('--savedir', help='directory to save trained model(s) in')
    parser.add_argument('--seeds', type=int, default=1, help='number of models to train with different random seeds (max=5, default=1)')
    parser.add_argument('--patchdim', type=int, default=5, help='size of input patches for CNN')
    parser.add_argument('--labelsmoothing', action='store_true', help='smooth labels')
    parser.add_argument('--batchsize', type=int, default=4096, help='number of examples in each training batch')
    parser.add_argument('--maxepochs', type=int, default=50, help='max number of epochs to train for (with early stopping)')
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
    parser.add_argument('--alpha', action='store_true', help='add SAR alpha band to input')
    parser.add_argument('--entropy', action='store_true', help='add SAR entropy band to input')
    parser.add_argument('--anisotropy', action='store_true', help='add SAR anisotropy band to input')
    parser.add_argument('--mchi_b', action='store_true', help='add SAR mchi_b band to input')
    parser.add_argument('--mchi_g', action='store_true', help='add SAR mchi_g band to input')
    parser.add_argument('--mchi_r', action='store_true', help='add SAR mchi_r band to input')
    # other
    parser.add_argument('--use_cached', action='store_true', help='use cached training and validation data')
    parser.add_argument('--savemodels', action='store_true', help='save model data')
    args = parser.parse_args()
    main(**vars(args))