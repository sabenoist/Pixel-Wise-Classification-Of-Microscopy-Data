from scripts_training_data.data_preparation import *
from scripts_training_data.make_weightmaps import *
from scripts_training_data.patch_statistics import *
from parameters import *
from multiprocessing import Pool


import warnings
warnings.filterwarnings('ignore')


paths = get_paths()
patch_augmentation_parameters = get_patch_parameters()


def extract_classes():
    """
    Extracts the percentage of times a class is present
    in the ground-truth dataset and also stores these
    in the classcounts.txt file.
    """

    classcounts = count_classes(patch_augmentation_parameters['out_path_label'])
    classcounts /= np.sum(classcounts)

    with open('{}/classcounts.txt'.format(paths['out_dir']), 'w') as f:
        f.write('classcounts = {} \n'.format(classcounts))

    return classcounts


def calc_dataset_stats():
    """
    Calculates the mean and variance of the training set
    and stores these in the patch_mean_var.txt file.
    """

    mean, sampleVariance = compute_training_set_statistics(patch_augmentation_parameters['out_path_raw'], )

    with open('{}/patch_mean_var.txt'.format(paths['out_dir']), 'w') as f:
        f.write('mean = {} \n'.format(mean))
        f.write('variance = {} \n'.format(sampleVariance))


if __name__ == '__main__':
    """
    Generates the weight maps based on the ground-truth 
    patches in patches/label/ and stores them at
    patches/wmap/.
    """

    classcounts = extract_classes()

    calc_dataset_stats()

    make_dirs(patch_augmentation_parameters['out_path_wmap'])

    tuplist = [
        (
            '{}/{}'.format(patch_augmentation_parameters['out_path_label'], a),
            '{}/{}'.format(patch_augmentation_parameters['out_path_wmap'], a),
            np.array(classcounts),
            10        )
        for a in os.listdir(patch_augmentation_parameters['out_path_label']) if a.endswith('.tif')
    ]

    p = Pool(processes=7)

    p.map(make_weightmap, tuplist)