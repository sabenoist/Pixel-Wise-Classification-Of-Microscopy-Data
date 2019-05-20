from torch.nn.modules.module import _addindent
from UNet import UNet
from parameters import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from PatchDataset import PatchDataset


paths = get_paths()
path = '{}/5_epochs_10e-4_lr_0.99_momentum.pickle'.format(paths['model_dir'])


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


def inspect_model(raw, label, output):
    raw = raw.numpy()
    label = label.numpy()
    output = output.detach().numpy()

    plt.subplot(3, 5, 3)
    plt.imshow(raw)

    for i in range(5):
        plt.subplot(3, 5, 6 + i)
        plt.imshow(label[:, :, i])

    for i in range(5):
        plt.subplot(3, 5, 11 + i)
        plt.imshow(output[0, i, :, :])

    plt.show()


model = UNet(in_channel=1, out_channel=5)
model.load_state_dict(torch.load(path))
model.eval()

print(torch_summarize(model))

patches = PatchDataset(paths['out_dir'], torch.device('cpu'))

output = model(patches[0]['raw'][None][None])

for i in range(1,2):
    output = model(patches[i]['raw'][None][None])

    print(patches[i]['patch_name'])
    inspect_model(patches[i]['raw'], patches[i]['label'], output)
