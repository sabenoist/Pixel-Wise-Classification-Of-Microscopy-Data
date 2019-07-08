import os
import torch
import numpy as np

from parameters import get_paths, get_patch_parameters
from PatchDataset import PatchDataset
from torch.utils.data import DataLoader
from skimage import io

import warnings
warnings.filterwarnings('ignore')

"""
Generates a validation set by randomly selecting 200 images from 
the training set and isolating them in their own folder
"""

VALIDATION_SET_SIZE = 200

params = get_patch_parameters()

paths = get_paths()
output_path = paths['val_dir']

# creates the directories for the validation files
if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(output_path + '/label')
    os.makedirs(output_path + '/raw')

# selects and loads in 200 patches at a time from the training dataset as batches
print("loading dataset...")
dataset = PatchDataset(paths['out_dir'], torch.device('cpu'), use_wmap=False)
patch_loader = DataLoader(dataset, batch_size=200, shuffle=True, num_workers=0)

# iterates over the over batches containing 200 patches each in the DataLoader
patch_counter = 0
for batch_ndx, batch in enumerate(patch_loader):
    if patch_counter >= VALIDATION_SET_SIZE:
        break

    for i in range(len(batch['patch_name'])):
        if patch_counter >= VALIDATION_SET_SIZE:
            break

        # retrieves from the batch the correct patches that correspond to each other.
        patch_name = batch['patch_name'][i]
        raw = batch['raw'][i].numpy()
        label = batch['label'][i].numpy()

        print(patch_counter + 1, patch_name)

        # saves the retrieved images in the validation_set directory.
        io.imsave('{}/raw/{}'.format(output_path, patch_name), raw.astype(params['img_type']))
        io.imsave('{}/label/{}'.format(output_path, patch_name), label.astype(np.uint8))

        # removes the retrieved images from the training_set directory.
        os.remove("{}/raw/{}".format(paths['out_dir'], patch_name))
        os.remove("{}/label/{}".format(paths['out_dir'], patch_name))

        # if weight-maps were used, the corresponding weight-map also needs to be removed.
        if os.path.exists("{}/wmap/{}".format(paths['out_dir'], patch_name)):
            os.remove("{}/wmap/{}".format(paths['out_dir'], patch_name))

        patch_counter += 1
