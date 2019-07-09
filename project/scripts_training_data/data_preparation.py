from skimage import io
import numpy as np
import os


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def classification_to_one_hot_ground_truth(prediction_path, out_path, number_of_classes=None):
    """
    Takes the labels from the ground-truth images and
    coverts them to a one-hot endoded ground-truth
    """
    files = [a for a in os.listdir(prediction_path) if a.endswith('.tif')]

    for f in files:
        img = io.imread('{}/{}'.format(prediction_path, f))
        img_argmax = np.argmax(img, axis=0)

        if number_of_classes is None:
            noc = np.max(img_argmax) + 1
        else:
            noc = number_of_classes

        img_out = np.eye(noc)[img_argmax]
        io.imsave('{}/{}'.format(out_path, f), img_out.astype(np.uint8))

    return
