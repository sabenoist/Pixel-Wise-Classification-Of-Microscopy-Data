from torch.nn.modules.module import _addindent
from UNet import UNet
from parameters import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from PatchDataset import PatchDataset
from sklearn.metrics import confusion_matrix


paths = get_paths()
path = '{}/Final_25_6_lr-9_stock_epoch_3_patch_30000.pickle'.format(paths['model_dir'])


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
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
    raw = raw.numpy()
    label = label.numpy()
    output = output.detach().numpy()

    plt.subplot(3, 6, 1)
    plt.imshow(raw)

    for i in range(6):
        plt.subplot(3, 6, 7 + i)
        plt.imshow(label[:, :, i])

    for i in range(6):
        plt.subplot(3, 6, 13 + i)
        plt.imshow(output[0, i, :, :])


    output_classes = np.argmax(output, axis=1)
    plt.subplot(3, 6, 5)
    plt.imshow(output_classes[0,...])

    label_classes = np.argmax(label, axis=2)
    plt.subplot(3, 6, 3)
    plt.imshow(label_classes)

    plt.show()


def read_mean_var():
    file = open('{}/patch_mean_var.txt'.format(paths['out_dir'])).readlines()

    mean = float(file[0].split()[-1])
    var = float(file[1].split()[-1])

    return [mean, var]


def normalize_input(input, mean, var):
    if var <= 0:
        var = 1
    return input.add(-mean).div(var)


def plot_results(raw, label, output):
    raw = raw.numpy()
    label = label.numpy()
    output = output.detach().numpy()

    plt.subplot(1,3,1)
    plt.title('raw')
    plt.imshow(raw)

    label = np.argmax(label, axis=2)
    plt.subplot(1,3,2)
    plt.title('ground truth')
    plt.imshow(label)
    print(label.shape)

    plt.subplot(1,3,3)
    plt.title('prediction')
    output_classes = np.argmax(output, axis=1)[0, ...]
    plt.imshow(output_classes)
    print(output.shape)
    plt.show()

    plot_confusion_matrix(label, output_classes, title='learning rate 10^-9, epoch 4')


def plot_confusion_matrix(y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = [0,1,2,3,4,5]
    np.set_printoptions(precision=2)

    # y_true = y_true.cpu().numpy()
    # y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()


    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.show()



model = UNet(in_channel=1, out_channel=6)
model.load_state_dict(torch.load(path))
model.eval()

print(torch_summarize(model))

# layer = -4
# print(list(model.parameters())[layer].data.shape)
# print(list(model.parameters())[layer].data[:,1,:,:])

patches = PatchDataset(paths['out_dir'], torch.device('cpu'))
mean, var = read_mean_var()

for i in range(31024,31025): # was 1,2 and 31024,31025
    output = model(patches[i]['raw'][None][None])

    print(patches[i]['patch_name'])
    label = patches[i]['label']
    # inspect_model(normalize_input(patches[i]['raw'], mean, var), label, output)
    plot_results(normalize_input(patches[i]['raw'], mean, var), label, output)

