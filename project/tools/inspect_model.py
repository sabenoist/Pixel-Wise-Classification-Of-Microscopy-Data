from torch.nn.modules.module import _addindent
from UNet import UNet
from parameters import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from PatchDataset import PatchDataset
from tools.confusion_matrix import plot_confusion_matrix


paths = get_paths()
# select here the desired model to inspect.
path = '{}/Final_25_6_lr-9_stock_epoch_3_patch_30000.pickle'.format(paths['model_dir'])


def torch_summarize(model, show_weights=True, show_parameters=True):
    """
    This function will give an overview of the created neural network
    by printing out all the weights and parameters
    """

    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def inspect_model(raw, label, output):
    """
    This function will give an overview of the prediction output
    by also showing the score maps for each class
    """

    raw = raw.numpy()
    label = label.numpy()
    output = output.detach().numpy()

    # plots the raw image.
    plt.subplot(3, 6, 1)
    plt.title('raw')
    plt.imshow(raw)

    # plots the one-hot encoded ground-truth.
    for i in range(6):
        plt.subplot(3, 6, 7 + i)
        plt.title('ground-truth')
        plt.imshow(label[:, :, i])

    # plots the score maps per class.
    for i in range(6):
        plt.subplot(3, 6, 13 + i)
        plt.title('score map')
        plt.imshow(output[0, i, :, :])

    # plots the predicted segmentation outcome.
    output_classes = np.argmax(output, axis=1)
    plt.subplot(3, 6, 5)
    plt.title('prediction')
    plt.imshow(output_classes[0, ...])

    # plots the expected segmentation.
    label_classes = np.argmax(label, axis=2)
    plt.subplot(3, 6, 3)
    plt.title('expected segmentation')
    plt.imshow(label_classes)

    plt.show()


def read_mean_var():
    """
    Reads the mean and variance from the patch_mean_var.txt file
    """

    file = open('{}/patch_mean_var.txt'.format(paths['val_dir'])).readlines()

    mean = float(file[0].split()[-1])
    var = float(file[1].split()[-1])

    return [mean, var]


def normalize_input(input, mean, var):
    """
    Normalizes the input based on the mean and variance parameters.
    """

    if var <= 0:
        var = 1
    return input.add(-mean).div(var)


def plot_results(raw, label, output):
    """
    Provides a simple plot from the image segmentation for the report
    and also immediately creates a confusion matrix of the prediction.
    """
    raw = raw.numpy()
    label = label.numpy()
    output = output.detach().numpy()

    # plots the raw image
    plt.subplot(1,3,1)
    plt.title('raw')
    plt.imshow(raw)

    # plots the ground truth
    label = np.argmax(label, axis=2)
    plt.subplot(1,3,2)
    plt.title('ground-truth')
    plt.imshow(label)

    # plots the segmentation prediction
    plt.subplot(1,3,3)
    plt.title('prediction')
    output_classes = np.argmax(output, axis=1)[0, ...]
    plt.imshow(output_classes)

    plt.show()

    plot_confusion_matrix(label, output_classes, title='learning rate 10^-9, epoch 4')


# loads in the model and sets it to evaluation mode so that it won't continue training.
model = UNet(in_channel=1, out_channel=6)
model.load_state_dict(torch.load(path))
model.eval()

print(torch_summarize(model))

# loads in the dataset of which images can be selected in the for-loop to inspect.
patches = PatchDataset(paths['val_dir'], torch.device('cpu'))
mean, var = read_mean_var()

for i in range(0, 5):  # any group of images can be selected here for inspection.
    output = model(patches[i]['raw'][None][None])

    print(patches[i]['patch_name'])
    label = patches[i]['label']

    inspect_model(normalize_input(patches[i]['raw'], mean, var), label, output)
    # plot_results(normalize_input(patches[i]['raw'], mean, var), label, output)

