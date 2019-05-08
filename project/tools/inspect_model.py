from torch.nn.modules.module import _addindent
from UNet import UNet
from parameters import *
import torch
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from PatchDataset import PatchDataset
from torch.utils.data import DataLoader


paths = get_paths()
path = '{}/testGPU.pickle'.format(paths['model_dir'])


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
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


def plot_5L_tensors(raw, output):
    raw = raw.numpy()
    output = output.detach().numpy()  # detaches grad from variable

    plt.subplot(2, 3, 1)
    plt.imshow(raw)

    plt.subplot(2, 3, 2)
    plt.imshow(output[0, 0, :, :])

    plt.subplot(2, 3, 3)
    plt.imshow(output[0, 1, :, :])

    plt.subplot(2, 3, 4)
    plt.imshow(output[0, 2, :, :])

    plt.subplot(2, 3, 5)
    plt.imshow(output[0, 3, :, :])

    plt.subplot(2, 3, 6)
    plt.imshow(output[0, 4, :, :])

    plt.show()


def plot_2d_tensors(raw, output):
    raw = raw.numpy()
    output = output.detach().numpy()  # detaches grad from variable

    plt.subplot(1, 2, 1)
    plt.imshow(raw)

    plt.subplot(1, 2, 2)
    plt.imshow(output[0, :, :])

    plt.show()


model = UNet(in_channel=1, out_channel=5)
model.load_state_dict(torch.load(path))
# model.eval()

print(torch_summarize(model))

patches = PatchDataset(paths['out_dir'], torch.device('cpu'))

# output = model(patches[0]['raw'][None][None])
#
# print(output.shape)
# plot_5L_tensors(patches[0]['raw'], output)
#
# predicted, _ = torch.max(output, 1)
# print(predicted.shape)
# plot_2d_tensors(patches[0]['raw'], predicted)

with torch.no_grad():
    for i in range(1):
        output = model(patches[i]['raw'][None][None])
        plot_5L_tensors(patches[i]['raw'], output)

        print(output.shape)

        # predicted, _ = torch.max(output, 1)
        # plot_2d_tensors(patches[i]['raw'], predicted)
