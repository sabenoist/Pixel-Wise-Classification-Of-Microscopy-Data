from scripts_training_data.extract_patches import *

from multiprocessing import Pool

import matplotlib.pyplot as plt
import os
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")

dataset = '20190326'

root_dir = 'D:/Bachelor_Project/VU_Bachelor_Project/project/datasets/{}/'.format(dataset)
out_dir = '{}/patches/'.format(root_dir)

label_in_path = '{}/segmented_bordered/'.format(root_dir)
segmented_in_path = '{}/segmented'.format(root_dir)

label_dir = '{}/label/'.format(root_dir)
raw_dir = '{}/raw/'.format(root_dir)

patch_augmentation_parameters = {
    'bin_image': False,
    'scaling': [0.9, 1.1, 'uniform'],
    'transposing': [0, 2, 'randint'],
    'rotating': [0, 4, 'randint'],
    'contrast_shifting': [0.2, 2.0, 'uniform'],
    'noise_mean': [0, 0.1, 'normal'],
    'noise_std': [0,0.05, 'normal'],
    'input_patch_size': [348,348],
    'output_patch_size': [164,164],
    'augmentations_per_image': 10,
    'patches_per_augmentation': 10,
    'label_dir': label_dir,
    'raw_dir': raw_dir,
    'img_type': np.float32,
    'label_class': 2,
    'min_pixels': 200,
    'out_path_raw': '{}/raw/'.format(out_dir),
    'out_path_label': '{}/label/'.format(out_dir),
    'out_path_wmap': '{}/wmap/'.format(out_dir)
}


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    img_list = [a for a in os.listdir(label_dir) if a.endswith('.tif')]

    paramlist = []
    for im in img_list:
        pap = patch_augmentation_parameters.copy()
        pap['frame'] = im
        paramlist.append(pap)

    p = Pool(processes=7)

    p.map(gt_generation, paramlist)








