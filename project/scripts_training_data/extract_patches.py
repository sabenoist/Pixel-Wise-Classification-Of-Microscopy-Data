import numpy as np
import os
from skimage import io
from skimage.transform import rescale
from skimage.exposure import rescale_intensity


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_random_numbers(params):
    """
    Returns a dictionary of random values for
    each augmentation parameter based on the
    given distribution in the params dictionary.
    """

    numbers = {}
    for key in params.keys():
        item = params[key]
        if isinstance(item, list):
            if item[-1] == 'uniform':
                numbers[key] = np.random.uniform(item[0], item[1])
            elif item[-1] == 'normal':
                numbers[key] = np.random.normal(item[0], item[1])
            elif item[-1] == 'randint':
                numbers[key] = np.random.randint(item[0], item[1])

    return numbers


def load_files(params, frame):
    """
    Loads in the images from the raw and label
    directories and applies the binning method
    when applicable.
    """

    labels = io.imread('{}/{}'.format(params['label_dir'], frame))
    labels = labels.astype(np.uint8)
    raw = io.imread('{}/{}'.format(params['raw_dir'], frame))
    raw = raw.astype(params['img_type'])

    if params['bin_image']:
        h, w = raw.shape
        labels = labels.reshape(2, h//2, 2, w//2).mean(-1).mean(-1)
        raw = raw.reshape(2, h//2, 2, w//2).mean(-1).mean(-1)

    raw /= 2**16-1

    return raw, labels


def pad(img, params):
    """
    To use patches from the edges of a given training image,
    the image is padded before patch extraction,
    because outputs of U-net are smaller than inputs into it.
    """

    large_patch_size = params['input_patch_size']
    pad_size = int(large_patch_size[0]/2)
    
    if len(img.shape) == 2:
        return np.pad(img, pad_size, 'reflect')
    
    if len(img.shape) == 3:
        # path the image dimensions, but not the classes
        if np.argmin(img.shape) == 0:
            pad_list = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
        else:
            pad_list = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))

        return np.pad(img, pad_list, 'reflect')


def rescale_img(img, R0, params):
    """
    Rescales the image based on the scaling value
    of parameter R0.
    """

    img = img.astype(np.float32)
    img_resc = rescale(img, R0, order=0, preserve_range=True, mode='constant')

    return img_resc.astype(params['img_type'])


def transpose(img, R1):
    """
    Transposes the image based on whether the binary
    parameter R1 is set to True or False.
    """

    if R1 == 0:
        return img
    else:
        if len(img.shape) == 2:
            return img.T
        if len(img.shape) == 3:
            return np.transpose(img, (1, 0, 2))


def rotate(img, R2):
    """
    Rotates the image based on the rotation value
    from parameter R2.
    """

    return np.rot90(img, R2)


def shift_contrast(img, R3, params):
    """
    Shifts the contrast of the image based on the
    contrast shifting value from parameter R3.
    """

    img = np.array((img)**R3).astype(params['img_type'])
    img = rescale_intensity(img, out_range=(0,1)).astype(params['img_type'])

    return img


def add_noise(img, R4, R5):
    """
    This function overlays noise sampled from a normal
    distribution. parameters of the normal distribution
    are random and the same in each image but different
    between images.
    """

    means = np.zeros(img.shape)
    means += R4
    sigma_squareds = np.zeros(img.shape)
    sigma_squareds += R5

    noise = np.random.normal(means, sigma_squareds)

    return img + noise


def augmentation(img, params, numbers, isbf=False):
    """
    Performs the augmentations on the given image by first
    padding it and then rescaling, transposing, and rotating it.
    Based on the binary parameter isbf, contrast shifting and
    noise addition may also be performed.
    """

    img = pad(img, params)
    img = rescale_img(img, numbers['scaling'], params)
    img = transpose(img, numbers['transposing'])
    img = rotate(img, numbers['rotating'])

    if isbf:
        img = shift_contrast(img, numbers['contrast_shifting'], params)
        img = add_noise(img, numbers['noise_mean'], abs(numbers['noise_std']))

    return img


def extract_patches(raw, labels, params):
    """
    Extracts patches from the labels image until a patch
    is found that matches the criteria. Then, the corresponding
    bright-field and fluorescence patches are extracted.
    """

    found = False
    ctr = 0

    large_patch_size = params['input_patch_size']
    small_patch_size = params['output_patch_size']
    
    offset = int((large_patch_size[0] - small_patch_size[0]) / 2)

    while not found:
        y, x, = pick_patch(raw, large_patch_size)
        found, label_patch = check_patch(labels, y, x, offset, params)
        ctr += 1
        if ctr >= 10000:
            found = True
            
    raw_patch = raw[y:y + large_patch_size[0], x:x + large_patch_size[1]]

    return raw_patch, label_patch


def pick_patch(img, large_patch_size):
    """
    Picks a random coordinate from the image for
    the patch extraction.
    """

    y = np.random.randint(0, img.shape[0] - large_patch_size[0])
    x = np.random.randint(0, img.shape[1] - large_patch_size[1])

    return y, x


def check_patch(img, y, x, offset, params):
    """
    Checks if the randomly generated patch has at least
    the specified number (min_pixel) number of pixels from
    the specified (label_class) class

    return: patch and boolean. True if patch is ok, False if patch is not ok.
    """

    small_patch_size = params['output_patch_size']
    label_class = params['label_class']

    patch = img[y + offset: y + offset + small_patch_size[0], x + offset: x + offset + small_patch_size[1]]

    if patch[:, :, label_class].sum() > params['min_pixels']:
        return True, patch
    else: 
        return False, patch


def gt_generation(params):
    """
    Generates the patches by first performing image augmentations
    and then extracting the patches from these augmented images.
    """

    frame = params['frame']
    frame_number = int(frame.split('frame_')[1].split('.tif')[0])

    out_path_raw = params['out_path_raw']
    out_path_label = params['out_path_label']
    
    make_dirs(out_path_raw)
    make_dirs(out_path_label)

    patch_multiplication_factor = params['augmentations_per_image'] * params['patches_per_augmentation']

    raw, labels = load_files(params, frame)

    patch_counter = frame_number * patch_multiplication_factor

    for a in range(params['augmentations_per_image']):
        numbers = make_random_numbers(params)

        raw_aug = augmentation(raw, params, numbers, isbf=True)
        label_aug = augmentation(labels, params, numbers, isbf=False)

        for b in range(params['patches_per_augmentation']):
            raw_patch, label_patch = extract_patches(raw_aug, label_aug, params)

            patchname = str(patch_counter).zfill(6)

            io.imsave('{}/patch_{}.tif'.format(out_path_raw, patchname), raw_patch.astype(params['img_type']))
            io.imsave('{}/patch_{}.tif'.format(out_path_label, patchname), label_patch.astype(np.uint8))

            patch_counter += 1


