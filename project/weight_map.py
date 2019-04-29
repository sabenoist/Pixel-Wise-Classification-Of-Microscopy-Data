from scripts_training_data.data_preparation import *
from scripts_training_data.make_weightmaps import *
from scripts_training_data.patch_statistics import *
from multiprocessing import Pool


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


def extract_classes():
    classcounts = count_classes(patch_augmentation_parameters['out_path_label'])
    classcounts /= np.sum(classcounts)

    with open('{}/classcounts.txt'.format(out_dir), 'w') as f:
        f.write('classcounts = {} \n'.format(classcounts))

    return classcounts


def calc_dataset_stats():
    mean, sampleVariance = compute_training_set_statistics(patch_augmentation_parameters['out_path_raw'], )

    with open('{}/patch_mean_var.txt'.format(out_dir), 'w') as f:
        f.write('mean = {} \n'.format(mean))
        f.write('variance = {} \n'.format(sampleVariance))


if __name__ == '__main__':
    classcounts = extract_classes()
    # classcounts = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    calc_dataset_stats()

    make_dirs(patch_augmentation_parameters['out_path_wmap'])

    tuplist = [
        (
            '{}/{}'.format(patch_augmentation_parameters['out_path_label'], a),
            '{}/{}'.format(patch_augmentation_parameters['out_path_wmap'], a),
            np.array(classcounts),
            10        )
        for a in os.listdir(patch_augmentation_parameters['out_path_label']) if a.endswith('.tif')  #TODO remove slice
    ]

    p = Pool(processes=7)

    p.map(make_weightmap, tuplist)