import os
import torch
import numpy as np

from parameters import get_paths, get_patch_parameters
from PatchDataset import PatchDataset
from torch.utils.data import DataLoader
from skimage import io

import warnings
warnings.filterwarnings('ignore')

VALIDATION_SET_SIZE = 200

params = get_patch_parameters()

paths = get_paths()
output_path = paths['val_dir']

if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(output_path + '/label')
    os.makedirs(output_path + '/raw')

print("loading dataset...")

dataset = PatchDataset(paths['out_dir'], torch.device('cpu'), use_wmap=False)
patch_loader = DataLoader(dataset, batch_size=200, shuffle=True, num_workers=0)

patch_counter = 0
for batch_ndx, sample in enumerate(patch_loader):
    if patch_counter >= VALIDATION_SET_SIZE:
        break

    for i in range(len(sample['patch_name'])):
        if patch_counter >= VALIDATION_SET_SIZE:
            break

        patch_name = sample['patch_name'][i]
        raw = sample['raw'][i].numpy()
        label = sample['label'][i].numpy()

        print(patch_counter + 1, patch_name)

        io.imsave('{}/raw/{}'.format(output_path, patch_name), raw.astype(params['img_type']))
        io.imsave('{}/label/{}'.format(output_path, patch_name), label.astype(np.uint8))

        patch_counter += 1