from scripts_training_data.data_preparation import *
from scripts_training_data.extract_patches import *

from skimage import io

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")

dataset = '20190326'

root_dir = 'D:/Bachelor_Project/code/project/datasets/{}/'.format(dataset)
out_dir = '{}/patches/'.format(root_dir)

label_in_path = '{}/segmented_bordered/'.format(root_dir)
segmented_in_path = '{}/segmented'.format(root_dir)

label_dir = '{}/label/'.format(root_dir)
raw_dir = '{}/raw/'.format(root_dir)


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def show_image(img):
    plt.imshow(img)
    plt.show()


def create_borders(img):
    binary_image = image_to_binary(img)
    borders = get_cell_borders(binary_image)

    result_image = binary_image + 2 * borders

    return result_image


def image_to_binary(img):
	return 1 * (img > 0)


def get_cell_borders(img):
    img_borders = img.copy()

    dilation_strength = 16
    for i in range(dilation_strength):
        img_borders = ndimage.binary_dilation(img_borders).astype(float)

    img_borders -= img

    return img_borders


def process_segmented_images(path):
    files = [a for a in os.listdir(path) if a.endswith('.tif')]

    for file in files:
        img = io.imread("{}/{}".format(path, file))
        bordered_img = create_borders(img)

        io.imsave("{}/{}".format(label_in_path, file), bordered_img)


def classification_to_one_hot_ground_truth(prediction_path, out_path, number_of_classes=None):
    make_dirs(out_path)

    files = [a for a in os.listdir(prediction_path) if a.endswith('.tif')]

    for f in files:
        img = io.imread('{}/{}'.format(prediction_path, f))
        img =img.astype(np.int32)

        if number_of_classes is None:
            noc = np.max(img) + 1
        else:
            noc = number_of_classes

        img_out = np.eye(int(noc))[img]
        
        io.imsave('{}/{}'.format(out_path, f), img_out.astype(np.uint8))




    # for f in files[0:1]:   #TODO remove slice
    #     img = io.imread('{}/{}'.format(prediction_path, f))
    #     img_argmax = np.argmax(img, axis=0)

    #     if number_of_classes is None:
    #         noc = np.max(img_argmax) + 1
    #     else:
    #         noc = number_of_classes

    #     img_out = np.eye(noc)[img_argmax]
    #     io.imsave('{}/{}'.format(out_path, f), img_out.astype(np.uint8))



#process_segmented_images(segmented_in_path)
classification_to_one_hot_ground_truth(label_in_path, label_dir, number_of_classes = None)





