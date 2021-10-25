# Hannah Kerner
# July 7, 2020

import numpy as np
from glob import glob
import os 
import sys
import math
import matplotlib.pyplot as plt

import keras
from keras.models import load_model
from keras.utils import to_categorical
from sklearn import metrics

import utils
import rasterio as rio

# class labels
CORN = 0
SOYBEAN = 1
OTHER = 2
N_CLASSES = 3
BATCHSIZE = 500000

def remove_val(arr):
    # SW quadrant was used for validation
    north = arr[:int(utils.HEIGHT/2)].flatten()
    se = arr[int(utils.HEIGHT/2):, int(utils.WIDTH/2):].flatten()
    test = np.concatenate([north, se])
    return test

def split_val(x_px, x_pa, y, ds, dp, inds):
    # Create lists to hold the data
    X_px = {'val': [], 'test': []}
    X_pa = {'val': [], 'test': []}
    Y = {'val': [], 'test': []}
    dS = {'val': [], 'test': []}
    dP = {'val': [], 'test': []}
    # Separate based on indices (val=southwest, test=rest)
    for i in range(x_px.shape[0]):
        # southwest quadrant goes in validation
        if inds[i][0] > utils.HEIGHT/2. and inds[i][1] < utils.WIDTH/2.:
            X_px['val'].append(x_px[i])
            X_pa['val'].append(x_pa[i])
            Y['val'].append(y[i])
            dS['val'].append(ds[i])
            dP['val'].append(dp[i])
        # everything else goes in test - TODO if we want to test at the end later
        else:
            X_px['test'].append(x_px[i])
            X_pa['test'].append(x_pa[i])
            Y['test'].append(y[i])
            dS['test'].append(ds[i])
            dP['test'].append(dp[i])

    # Turn the lists into numpy arrays
    for k in X_px.keys():
        X_px[k] = np.array(X_px[k])
        X_pa[k] = np.array(X_pa[k])
        Y[k] = np.array(Y[k])
        dS[k] = np.array(dS[k])
        dP[k] = np.array(dP[k])

    return X_px['test'], X_pa['test'], Y['test'], dS['test'], dP['test']

# each array must have shape numsamples x ...
def _standardize_array(x, modeldir, name):
    # Calculate training data statistics
    mu = np.load(os.path.join(modeldir, '%s_train_mu.npy' % name))
    std = np.load(os.path.join(modeldir, '%s_train_std.npy' % name))
    # Apply standardization
    return (x - mu)/std

def standardize(X_px, X_pa, delta_S, delta_P, modeldir, seed):
    # Standardize each of the features
    X_px = _standardize_array(X_px, modeldir, 'x_px_seed%s' % seed)
    X_pa = _standardize_array(X_pa, modeldir, 'x_pa_seed%s' % seed)
    delta_S = _standardize_array(delta_S, modeldir, 'delta_s_seed%s' % seed)
    delta_P = _standardize_array(delta_P, modeldir, 'delta_p_seed%s' % seed)
    return X_px, X_pa, delta_S, delta_P

def class_to_rgb(Y):
    Y_rgb = np.ndarray([Y.shape[0], Y.shape[1], 3])
    Y_rgb[np.where(Y==CORN)] = [1, 0.82, 0] # yellow
    Y_rgb[np.where(Y==OTHER)] = [0.8, 0.8, 0.8] # gray
    Y_rgb[np.where(Y==SOYBEAN)] = [0.149, 0.439, 0] # green
    return Y_rgb

def main(testdir, labeldir, phenodir, tile_id, testyear, timesteps, hlsimg,
         modelname, modeldir, n_models, savedir, patchdim, 
         ndvi=False, bsi=False, evi=False, lswi=False, ndwi=False, gcvi=False,
         vv=False, vh=False, ia=False, alpha=False, entropy=False, anisotropy=False, mchi_b=False, mchi_g=False, mchi_r=False):
    # Load the data
    x, valid_mask = utils.load_data(testdir, testyear, tile_id,
                                    ndvi, bsi, evi, lswi, ndwi, gcvi, 
                                    vv, vh, ia, alpha, entropy, anisotropy, mchi_b, mchi_g, mchi_r)
    y = utils.load_labels(labeldir, testyear, tile_id)
    dois = utils.load_dois(phenodir, testyear, tile_id)
    delta_s = dois[...,2]-dois[...,0]
    delta_p = dois[...,2]-dois[...,0]
    # Make inputs in the format they go to the model
    x_px, x_pa, y, delta_s, delta_p, inds = utils.make_inputs(x, y, delta_s, delta_p, 
                                                              valid_mask=valid_mask, 
                                                              patchsize=patchdim, 
                                                              dataset='test')

    # Debug check
    print('test data sizes')
    print(x_px.shape)
    print(x_pa.shape)
    print(y.shape)
    print(delta_s.shape)
    print(delta_p.shape)

    # Check that the number of models asked to run is the same as the number of models found in modeldir
    if len(glob(os.path.join(modeldir, '*seed*.h5'))) != n_models:
        print('Number of models found in modeldir (%d) != number of models requested (%d)' 
              % (len(glob(os.path.join(modeldir, '%s*seed*.h5' % modelname))), n_models))
        sys.exit(0)

    # Swap temporal and spectral axes (LSTM wants channels time first)
    x_px = np.swapaxes(x_px, 1, 2)

    # Turn integer labels into one-hot labels
    y = to_categorical(y, num_classes=N_CLASSES)

    # Create a dict to store results for all runs
    results = {}

    # Iterate through each of the models with different random seeds
    for model in glob(os.path.join(modeldir, '%s*seed*.h5' % modelname)):
        seed = os.path.basename(model).split('_')[-1][5:-3]
        print('Testing model with random seed %s' % seed)

        # Run in batches
        y_pred = np.ndarray([x_px.shape[0],N_CLASSES])
        for ba in range(int(math.ceil(y_pred.shape[0]/BATCHSIZE))):
            # if it's the last batch (partial)
            if ba == int(math.ceil(y_pred.shape[0]/BATCHSIZE))-1:
                x_px_ba = x_px[ba*BATCHSIZE:]
                x_pa_ba = x_pa[ba*BATCHSIZE:]
                delta_s_ba = delta_s[ba*BATCHSIZE:]
                delta_p_ba = delta_p[ba*BATCHSIZE:]
            # if it's not the last batch
            else:
                x_px_ba = x_px[ba*BATCHSIZE:ba*BATCHSIZE+BATCHSIZE]
                x_pa_ba = x_pa[ba*BATCHSIZE:ba*BATCHSIZE+BATCHSIZE]
                delta_s_ba = delta_s[ba*BATCHSIZE:ba*BATCHSIZE+BATCHSIZE]
                delta_p_ba = delta_p[ba*BATCHSIZE:ba*BATCHSIZE+BATCHSIZE]

            # Standardize data
            x_px_std, x_pa_std, delta_s_std, delta_p_std = standardize(x_px_ba, x_pa_ba, delta_s_ba, delta_p_ba, modeldir, seed)

            # Limit the timesteps if necessary
            if timesteps < 3:
                x_px_std = x_px_std[:,:timesteps]            
                x_pa_std = x_pa_std[...,:timesteps*x_px_std.shape[-1]]

            # Load the trained model
            clf = load_model(model)
            # Predict 
            if modelname == 'lstm_cnn_delta':
                if timesteps == 3:
                    y_pred_ba = clf.predict([x_px_std, x_pa_std, delta_s_std])
                elif timesteps == 2:
                    y_pred_ba = clf.predict([x_px_std, x_pa_std, delta_p_std])
            elif modelname == 'lstm_cnn':
                y_pred_ba = clf.predict([x_px_std, x_pa_std])
            elif modelname == 'cnn':
                y_pred_ba = clf.predict([x_pa_std])
            elif modelname == 'lstm':
                y_pred_ba = clf.predict([x_px_std])
                
            # Add it to the total pred array
            # if it's the last batch (partial)
            if ba == int(math.ceil(y_pred.shape[0]/BATCHSIZE))-1:
                y_pred[ba*BATCHSIZE:] = y_pred_ba
            else:
                y_pred[ba*BATCHSIZE:ba*BATCHSIZE+BATCHSIZE] = y_pred_ba

        # Compute border to ignore when computing metrics
        w = int(patchdim/2)+1
        # Convert probs to classes
        y_true_class = np.argmax(y, axis=1)
        y_true_class = np.reshape(y_true_class, [utils.HEIGHT-w*2, utils.WIDTH-w*2])
        y_pred_class = np.argmax(y_pred, axis=1)
        y_pred_class = np.reshape(y_pred_class, [utils.HEIGHT-w*2, utils.WIDTH-w*2])
        
        filled = (np.ones([3660,3660])*255).astype(np.uint8)
        filled[3:-3,3:-3] = y_pred_class
        # Save the predicted map as a raster
        with rio.open(hlsimg) as src:
            profile = src.profile
            profile['dtype'] = np.uint8
            profile['count'] = 1
            profile['driver'] = 'GTiff'
            profile['nodata'] = 255
            with rio.open(os.path.join(savedir, 'map_results_seed%s.tif' % seed), 'w', **profile) as dst:
                dst.write(np.expand_dims(filled, 0))

        # Save the predicted map as an image
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        fig.suptitle(model)
        ax1.set_title('CDL %d' % testyear)
        ax1.imshow(class_to_rgb(y_true_class[500:1000,500:1000]))
        ax1.axis('off')
        ax2.set_title('%s Prediction' % modelname)
        ax2.imshow(class_to_rgb(y_pred_class[500:1000,500:1000]))
        ax2.axis('off')
        fig.savefig(os.path.join(savedir, 'map_results_seed%s.png' % seed))
        # Save the predicted map as an array
        np.save(os.path.join(savedir, 'map_results_seed%s.npy' % seed), y_pred_class)

        print(y_pred_class.shape)
        print(y_true_class.shape)

        # Remove the validation pixels
        # y_true_class = remove_val(y_true_class)
        # y_pred_class = remove_val(y_pred_class)
        # print('Removed validation pixels from test set')
        # print(y_pred_class.shape)
        # print(y_true_class.shape)

        # TODO: Mask out invalid pixels in accuracy assessment

        # Compute performance metrics
        acc = metrics.accuracy_score(y_true_class.flatten(), y_pred_class.flatten())
        print('Accuracy: %f' % acc)
        cfx = metrics.confusion_matrix(y_true_class.flatten(), y_pred_class.flatten())
        print('Confusion matrix:')
        print(cfx)
    #     cr = metrics.classification_report(y_true_class, y_pred_class, target_names=['corn', 'soybean', 'other'])
    #     print('Classification report:')
    #     print(cr)
    #     crdict = metrics.classification_report(y_true_class, y_pred_class, target_names=['corn', 'soybean', 'other'], output_dict=True)
    #     # Store in the results dictionary
    #     results[seed] = {'name': model,
    #                      'acc': acc,
    #                      'confusion_matrix': cfx,
    #                      'classification_report': cr,
    #                      'classification_report_dict': crdict
    #                     }

    # # add average of all seeds
    # results['avg'] = {key: 0 for key in results.keys()}
    # for rseed in results.keys():
    #     if rseed == 'avg':
    #         continue
    #     # Sum up each metric across all seeds
    #     for metric in results[rseed].keys():
    #         if metric == 'classification_report' or 'name':
    #             continue
    #         results['avg'][metric] += results[rseed][metric]
    # # then divide by the number of seeds
    # for metric in results['avg'].keys():
    #     if metric == 'classification_report':
    #         continue
    #     results['avg'][metric] /= len(results.keys())-2
    #     print(type(len(results.keys())-2))
    #     print('Dividing by %d seeds' % len(results.keys())-2)


    # # Write the results dictionary
    # f = open(os.path.join(savedir, 'results.txt'), 'w')
    # f.write(str(results))
    # f.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, 
                                     description='Test crop type classifiers')
    # Test data info
    parser.add_argument('--testdir', help='directory containing test data (numpy files)')
    parser.add_argument('--labeldir', help='directory containing test labels (numpy files)')
    parser.add_argument('--phenodir', default='/gpfs/data1/cmongp1/hkerner/us-corn/data/phenology', 
                                     help='directory containing phenology data')
    parser.add_argument('--tile_id', help='ID of HLS tile to test on (e.g., 16TBL)')
    parser.add_argument('--testyear', type=int, default=2019, help='year of data to use for testing')
    parser.add_argument('--timesteps', type=int, default=3, help='number of timesteps to use in prediction')
    parser.add_argument('--hlsimg', help='path to an example HLS image we want to match the CDL raster to')
    # Model info
    parser.add_argument('--modelname', choices=['lstm_cnn_delta', 'cnn', 'lstm_cnn', 'lstm'], help='name of model to test')
    parser.add_argument('--modeldir', help='directory containing model file(s)')
    parser.add_argument('--n_models', type=int, help='number of models to apply to test data')
    parser.add_argument('--savedir', help='directory to save results in')
    parser.add_argument('--patchdim', type=int, default=5, help='size of input patches for CNN')
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

    args = parser.parse_args()
    main(**vars(args))