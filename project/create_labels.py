from scripts_training_data.data_preparation import *
from scripts_training_data.extract_patches import *
from parameters import *
from skimage import io

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')


paths = get_paths()


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def show_image(img):
    plt.imshow(img)
    plt.show()


def create_borders(img):
    binary_image = image_to_binary(img)
    borders = get_cell_borders(binary_image)

    # sets the border class to 3 and adds it to the ground-truth
    result_image = binary_image + 3 * borders

    return result_image


def image_to_binary(img):
    """
    Converts the image parameter into a binary image
    where any value above 0 is converted to 1.
    """
    return 1 * (img > 0)


def get_cell_borders(img):
    """
    Generates the border class by dilating the image
    and then subtracting this dilated image from the
    original image, creating an image that contains
    only the border class.
    """
    img_borders = img.copy()

    dilation_strength = 16
    for i in range(dilation_strength):
        img_borders = ndimage.binary_dilation(img_borders).astype(float)

    img_borders -= img

    return img_borders


def process_segmented_images(path):
    """
    Reads in all segmented images and adds to border class to them.
    Afterward, saves the resulting images in the 'segmented_bordered'
    directory.
    """
    files = [a for a in os.listdir(path) if a.endswith('.tif')]

    for file in files:
        img = io.imread('{}/{}'.format(path, file))
        bordered_img = create_borders(img)

        io.imsave('{}/{}'.format(paths['label_in_path'], file), bordered_img)


def classification_to_one_hot_ground_truth(prediction_path, out_path, number_of_classes=None):
    """
    Reads in all the ground-truth images from the 'segmented_bordered'
    folder and converts these to one-hot ground-truth images that are
    saved in the 'label' directory.
    """
    make_dirs(out_path)

    files = [a for a in os.listdir(prediction_path) if a.endswith('.tif')]

    for f in files:
        img = io.imread('{}/{}'.format(prediction_path, f))
        img = img.astype(np.int32)

        if number_of_classes is None:
            noc = np.max(img) + 1
        else:
            noc = number_of_classes

        img_out = np.eye(int(noc))[img]
        
        io.imsave('{}/{}'.format(out_path, f), img_out.astype(np.uint8))


if __name__ == '__main__':
    process_segmented_images(paths['segmented_in_path'])
    classification_to_one_hot_ground_truth(paths['label_in_path'], paths['label_dir'], number_of_classes=5)





