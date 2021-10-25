# Hannah Kerner
# February 5, 2021

import numpy as np
import matplotlib.pyplot as plt
import os

import utils
import train

# class labels
CORN = 0
SOYBEAN = 1
OTHER = 2
N_CLASSES = 3
# size of training and validation subsets
TRAINSIZE = 100000
VALSIZE = 10000



def load_samples_for_tile(tile_id, start_year, end_year, 
            traindir, labeldir, phenodir, homogmask_dir,
            timesteps, num_samples, ndvi, bsi, evi, lswi,
            ndwi, gcvi, vv, vh, ia, patchdim):
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
                                  ndvi, bsi, evi, lswi, ndwi, gcvi, vv, vh, ia)
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
        # Add to the dictionary for all data
        X_px[year] = x_px
        X_pa[year] = x_pa
        Y[year] = y
        delta_S[year] = delta_s
        delta_P[year] = delta_p
        indices[year] = inds

    # Balance the classes
    X_px, X_pa, Y, delta_S, delta_P, inds = train.balance_classes(X_px, X_pa, Y, delta_S, delta_P, indices)
    # Combine the inputs for all years
    X_px = np.concatenate([X_px[year] for year in X_px.keys()], axis=0)
    X_pa = np.concatenate([X_pa[year] for year in X_pa.keys()], axis=0)
    Y = np.concatenate([Y[year] for year in Y.keys()], axis=0)
    delta_S = np.concatenate([delta_S[year] for year in delta_S.keys()], axis=0)
    delta_P = np.concatenate([delta_P[year] for year in delta_P.keys()], axis=0)
    indices = np.concatenate([indices[year] for year in indices.keys()], axis=0)
    # Sample from dataset
    return train.subset_data(X_px, X_pa, Y, delta_S, delta_P, indices, n=num_samples)

def main(traindir, labeldir, phenodir, tile_id, start_year, end_year, val_year,
         homogmask_dir, timesteps, savedir, seed, patchdim, ndvi, bsi, evi, lswi,
         ndwi, gcvi, vv, vh, ia):
    # Create the save directory if it doesn't exist
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    # Set the random seed
    np.random.seed(seed)
    # Calculate the split of training samples across adjacent tiles
    n_split = int(TRAINSIZE / float(len(utils.get_adjacent(tile_id))+1))
    print('Loading %d training samples from each tile' % n_split)
    print('Loading %d validation samples from prediction tile' % VALSIZE)
    # Set up a dictionary to store the samples
    px_trn = {}
    pa_trn = {}
    y_trn = {}
    deltaS_trn = {}
    deltaP_trn = {}
    inds_trn = {}
    # First load the samples from the prediction tile
    px_trn[tile_id], pa_trn[tile_id], y_trn[tile_id], deltaS_trn[tile_id], deltaP_trn[tile_id], inds_trn[tile_id] = load_samples_for_tile(
                                                                                                                      tile_id, start_year, end_year, 
                                                                                                                      traindir, labeldir, phenodir, homogmask_dir,
                                                                                                                      timesteps, n_split, ndvi, bsi, evi, lswi,
                                                                                                                      ndwi, gcvi, vv, vh, ia, patchdim)
    px_val, pa_val, y_val, deltaS_val, deltaP_val, inds_val = load_samples_for_tile(
                                                                  tile_id, val_year, val_year, 
                                                                  traindir, labeldir, phenodir, homogmask_dir,
                                                                  timesteps, VALSIZE, ndvi, bsi, evi, lswi,
                                                                  ndwi, gcvi, vv, vh, ia, patchdim)
    print('Loaded validation data with shape', px_val.shape)
    # Write the validation data
    np.save(os.path.join(savedir, 'val_X_px.npy'), px_val)
    np.save(os.path.join(savedir, 'val_X_pa.npy'), pa_val)
    np.save(os.path.join(savedir, 'val_Y.npy'), y_val)
    np.save(os.path.join(savedir, 'val_delta_S.npy'), deltaS_val)
    np.save(os.path.join(savedir, 'val_delta_P.npy'), deltaP_val)
    np.save(os.path.join(savedir, 'val_inds.npy'), inds_val)
    print('Successfully saved validation data in %s' % savedir)
    # Next load the samples from the adjacent tiles
    for tile in utils.get_adjacent(tile_id):
        px_trn[tile], pa_trn[tile], y_trn[tile], deltaS_trn[tile], deltaP_trn[tile], inds_trn[tile] = load_samples_for_tile(
                                                                                                        tile, start_year, end_year, 
                                                                                                        traindir, labeldir, phenodir, homogmask_dir,
                                                                                                        timesteps, n_split, ndvi, bsi, evi, lswi,
                                                                                                        ndwi, gcvi, vv, vh, ia, patchdim)
    # Concatenate them as a single array
    px_trn = np.concatenate([px_trn[tile] for tile in px_trn.keys()], axis=0)
    pa_trn = np.concatenate([pa_trn[tile] for tile in pa_trn.keys()], axis=0)
    y_trn = np.concatenate([y_trn[tile] for tile in y_trn.keys()], axis=0)
    deltaS_trn = np.concatenate([deltaS_trn[tile] for tile in deltaS_trn.keys()], axis=0)
    deltaP_trn = np.concatenate([deltaP_trn[tile] for tile in deltaP_trn.keys()], axis=0)
    inds_trn = np.concatenate([inds_trn[tile] for tile in inds_trn.keys()], axis=0)
    print('Loaded training data with shape', px_trn.shape)
    # Write the training data
    np.save(os.path.join(savedir, 'train_X_px.npy'), px_trn)
    np.save(os.path.join(savedir, 'train_X_pa.npy'), pa_trn)
    np.save(os.path.join(savedir, 'train_Y.npy'), y_trn)
    np.save(os.path.join(savedir, 'train_delta_S.npy'), deltaS_trn)
    np.save(os.path.join(savedir, 'train_delta_P.npy'), deltaP_trn)
    np.save(os.path.join(savedir, 'train_inds.npy'), inds_trn)
    print('Successfully saved training data in %s' % savedir)
        
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
    parser.add_argument('--val_year', type=int, default=2018, help='year of data to use for validation (will only use SW quadrant)')
    parser.add_argument('--homogmask_dir', help='directory containing homogeneity masks for filtering labels')
    parser.add_argument('--timesteps', type=int, default=3, help='number of timesteps to use in prediction')
    parser.add_argument('--savedir', help='directory to save training data in')
    parser.add_argument('--seed', type=int, default=488, help='number of models to train with different random seeds (max=5, default=1)')
    parser.add_argument('--patchdim', type=int, default=5, help='size of input patches for CNN')
    # Band indices
    parser.add_argument('--ndvi', action='store_true', default=False, help='add NDVI band to input')
    parser.add_argument('--bsi', action='store_true', default=False, help='add BSI band to input')
    parser.add_argument('--evi', action='store_true', default=False, help='add EVI band to input')
    parser.add_argument('--lswi', action='store_true', default=False, help='add LSWI band to input')
    parser.add_argument('--ndwi', action='store_true', default=False, help='add NDWI band to input')
    parser.add_argument('--gcvi', action='store_true', default=False, help='add GCVI band to input')
    # SAR features
    parser.add_argument('--vv', action='store_true', default=False, help='add VV SAR band to input')
    parser.add_argument('--vh', action='store_true', default=False, help='add VH SAR band to input')
    parser.add_argument('--ia', action='store_true', default=False, help='add incidence angle SAR band to input')
    # Parse arguments
    args = parser.parse_args()
    main(**vars(args))