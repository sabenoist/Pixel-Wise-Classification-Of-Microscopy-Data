import numpy as np

def get_paths():
    dataset = '20190326'
    root_dir = 'E:/Bachelor_Project/VU_Bachelor_Project/project/datasets/{}/'.format(dataset)

    out_dir = '{}/patches/'.format(root_dir)

    label_in_path = '{}/segmented_bordered/'.format(root_dir)
    segmented_in_path = '{}/segmented'.format(root_dir)

    label_dir = '{}/label/'.format(root_dir)
    raw_dir = '{}/raw/'.format(root_dir)

    model_dir = '{}/models/'.format(root_dir)
    val_dir = '{}/validation/'.format(root_dir)

    paths = {
        'dataset': dataset,
        'root_dir': root_dir,
        'out_dir': out_dir,
        'label_in_path': label_in_path,
        'segmented_in_path': segmented_in_path,
        'label_dir': label_dir,
        'raw_dir': raw_dir,
        'model_dir': model_dir,
        'val_dir': val_dir
    }

    return paths


def get_patch_parameters():
    paths = get_paths()

    patch_augmentation_parameters = {
        'bin_image': False,
        'scaling': [0.9, 1.1, 'uniform'],
        'transposing': [0, 2, 'randint'],
        'rotating': [0, 4, 'randint'],
        'contrast_shifting': [0.2, 2.0, 'uniform'],
        'noise_mean': [0, 0.1, 'normal'],
        'noise_std': [0, 0.05, 'normal'],
        'input_patch_size': [348, 348],
        'output_patch_size': [164, 164],
        'augmentations_per_image': 10,
        'patches_per_augmentation': 10,
        'label_dir': paths['label_dir'],
        'raw_dir': paths['raw_dir'],
        'img_type': np.float32,
        'label_class': 3,   # 0=bg, 1=cells, 2=buds, 3=borders, 4=noise
        'min_pixels': 200,
        'out_path_raw': '{}/raw/'.format(paths['out_dir']),
        'out_path_label': '{}/label/'.format(paths['out_dir']),
        'out_path_wmap': '{}/wmap/'.format(paths['out_dir'])
    }

    return patch_augmentation_parameters

