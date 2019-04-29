from scripts_training_data.extract_patches import *
from parameters import *

from multiprocessing import Pool

import os
import time

import warnings
warnings.filterwarnings("ignore")


paths = get_paths()
patch_augmentation_parameters = get_patch_parameters()


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    img_list = [a for a in os.listdir(paths['label_dir']) if a.endswith('.tif')]

    paramlist = []
    for im in img_list:
        pap = patch_augmentation_parameters.copy()
        pap['frame'] = im
        paramlist.append(pap)

    p = Pool(processes=7)

    p.map(gt_generation, paramlist)







