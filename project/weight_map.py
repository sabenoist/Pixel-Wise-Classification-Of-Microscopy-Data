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
    classcounts = count_classes(patch_augmentation_parameters['out_path_label'])
    classcounts /= np.sum(classcounts)

    with open('{}/classcounts.txt'.format(paths['out_dir']), 'w') as f:
        f.write('classcounts = {} \n'.format(classcounts))

    return classcounts


def calc_dataset_stats():
    mean, sampleVariance = compute_training_set_statistics(patch_augmentation_parameters['out_path_raw'], )

    with open('{}/patch_mean_var.txt'.format(paths['out_dir']), 'w') as f:
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
        for a in os.listdir(patch_augmentation_parameters['out_path_label']) if a.endswith('.tif')
    ]

    p = Pool(processes=7)

    p.map(make_weightmap, tuplist)