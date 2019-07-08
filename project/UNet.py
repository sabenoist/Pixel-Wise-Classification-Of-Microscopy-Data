import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import select_device
from parameters import get_paths
from PatchDataset import PatchDataset
from WeightedCrossEntropyLoss import WeightedCrossEntropyLoss
from SpatialWeightedSGD import SpatialWeightedSGD
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# UNet layer sizes
layer1_size = 64
layer2_size = 128
layer3_size = 256
layer4_size = 512
layer5_size = 1024


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        # Downwards encoding part
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=layer1_size)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv_encode2 = self.contracting_block(in_channels=layer1_size, out_channels=layer2_size)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv_encode3 = self.contracting_block(in_channels=layer2_size, out_channels=layer3_size)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv_encode4 = self.contracting_block(in_channels=layer3_size, out_channels=layer4_size)
        self.conv_maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=layer4_size, out_channels=layer5_size),
                            nn.ReLU(),
                            nn.BatchNorm2d(layer5_size),
                            nn.Conv2d(kernel_size=3, in_channels=layer5_size, out_channels=layer5_size),
                            nn.ReLU(),
                            nn.BatchNorm2d(layer5_size),
                            nn.ConvTranspose2d(in_channels=layer5_size, out_channels=layer4_size, kernel_size=3, stride=2,
                                               padding=1, output_padding=1)
                                              )

        # Upwards decoding part
        self.conv_decode4 = self.expansive_block(layer5_size, layer4_size, layer3_size)
        self.conv_decode3 = self.expansive_block(layer4_size, layer3_size, layer2_size)
        self.conv_decode2 = self.expansive_block(layer3_size, layer2_size, layer1_size)
        self.final_layer = self.final_block(layer2_size, layer1_size , out_channel)


    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return block


    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block


    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return block


    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)


    def init_weights(self, m):
        # Can be applied to convolution layers to initiate custom weights
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, input_img):
        # Encode
        encode_block1 = self.conv_encode1(input_img)
        encode_pool1 = self.conv_maxpool1(encode_block1)

        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)

        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_maxpool4(encode_block4)

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool4)

        # Decode
        decode_block4 = self.crop_and_concat(bottleneck1, encode_block4, crop=True)
        cat_layer3 = self.conv_decode4(decode_block4)

        decode_block3 = self.crop_and_concat(cat_layer3, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)

        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)

        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)

        return final_layer


def gradClamp(parameters, clip=5):
    for p in parameters:
        p.grad.data.clamp_(min=-clip, max=clip)


def train_UNet(device, unet, dataset, validation_set, width_out, height_out, epochs=10):
    model_name = 'Final_25_6_lr-9_stock'

    # criterion = WeightedCrossEntropyLoss(device).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(unet.parameters(), lr=10e-9, momentum=0.99)
    optimizer.zero_grad() # sets the gradient to accumulate instead of replace.

    loss_info = list()
    validation_info = list()

    mean, var = read_mean_var()

    for epoch in range(epochs):
        batch_size = 200
        patch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        patches_amount = len(dataset)
        patch_counter = 0
        for batch_ndx, sample in enumerate(patch_loader):
            # if patch_counter >= 1:
            #     return

            for i in range(len(sample['patch_name'])):
                # if patch_counter >= 1:
                #     return

                # Forward part
                patch_name = sample['patch_name'][i]
                raw = normalize_input(sample['raw'][i], mean, var)
                label = Variable(sample['label'][i])
                wmap = sample['wmap'][i]

                output = unet(raw[None][None])  # None will add the missing dimensions at the front, the Unet requires a 4d input for the weights.

                # Backwards part
                output = output.permute(0, 2, 3, 1)  # permute such that number of desired segments would be on 4th dimension
                label = label.unsqueeze(0)  # sets label to have the same dimensions as output
                m = output.shape[0]

                # Resizing the outputs and label to calculate pixel wise softmax loss
                output = output.resize(m * width_out * height_out, 6)
                label = label.resize(m * width_out * height_out, 6)
                wmap = wmap.resize(m * width_out * height_out, 1)

                # loss = criterion(output, label, wmap=wmap)
                loss = criterion(output, torch.argmax(label, 1))
                loss.backward()
                gradClamp(unet.parameters(), clip=5)

                optimizer.step()

                # save loss info per 100 images
                if patch_counter % 99 == 0:
                    loss_info.append([epoch * patches_amount, patch_counter, loss.item()])

                # perform validation per 2000 images
                if patch_counter % 1999 == 0:
                    validation_info.append([epoch * patches_amount, patch_counter, run_validation(device, unet, validation_set, width_out, height_out)])

                # save model every 10000 images
                if patch_counter % 10000 == 0 and patch_counter != 0:
                    save_model(unet, paths['model_dir'], model_name + '_epoch_' + str(epoch) + '_patch_' + str(patch_counter) + '.pickle')

                print('{}. [{}/{}] - {} loss: {}'.format(epoch + 1, patch_counter + 1, patches_amount, patch_name, loss))

                # save confusion matrix every end of the epoch
                if patch_counter == patches_amount - 1:
                    plot_confusion_matrix(torch.argmax(label, 1), torch.argmax(output, 1), epoch, title='learning rate 10^-9, epoch {}'.format(epoch + 1))

                patch_counter += 1

    save_model(unet, paths['model_dir'], model_name + '.pickle')
    save_loss_info(loss_info, paths['model_dir'], model_name + '_loss.txt')
    save_loss_info(validation_info, paths['model_dir'], model_name + '_validation.txt')


def run_validation(device, unet, validation_set, width_out, height_out):
    print("running validation test")

    # validation_criterion = WeightedCrossEntropyLoss(device).to(device)
    validation_criterion = nn.CrossEntropyLoss().to(device)

    batch_size = 200
    patch_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)

    losses = list()

    mean, var = read_mean_var(training=False)

    validation_counter = 0
    validation_amount = len(validation_set)

    unet.eval()  # sets the network to stop learning

    for batch_ndx, sample in enumerate(patch_loader):
        for i in range(batch_size):
            patch_name = sample['patch_name'][i]
            raw = normalize_input(sample['raw'][i], mean, var)
            # label = Variable(sample['label'][i])
            label = Variable(sample['label'][i])

            output = unet(raw[None][None])  # None will add the missing dimensions at the front, the Unet requires a 4d input for the weights.

            output = output.permute(0, 2, 3, 1)  # permute such that number of desired segments would be on 4th dimension

            m = output.shape[0]
            label = label.unsqueeze(0)

            # Resizing the outputs and label to calculate pixel wise softmax loss
            output = output.resize(m * width_out * height_out, 6)
            label = label.resize(m * width_out * height_out, 6)

            # loss = validation_criterion(output, label)
            # loss = validation_criterion(output, torch.max(label, 1)[1])
            loss = validation_criterion(output, torch.argmax(label, 1))

            print('validation. [{}/{}] {} - loss: {}'.format(validation_counter + 1, validation_amount, patch_name, loss.item()))

            losses.append(loss.item())

            validation_counter += 1

    unet.train()  # sets the network to start learning again

    return sum(losses) / len(losses)


def save_model(unet, path, name):
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(unet.state_dict(), path + name)


def save_loss_info(loss_info, path, name):
    if not os.path.exists(path):
        os.makedirs(path)

    file = open(path + name, 'w+')
    file.write('patch_counter loss\n')

    for info in loss_info:
        file.write('{} {} \n'.format(info[0] + info[1] + 1, info[2]))

    file.close()


def read_mean_var(training=True):
    if training:
        file = open('{}/patch_mean_var.txt'.format(paths['out_dir'])).readlines()
    else:
        file = open('{}/patch_mean_var.txt'.format(paths['val_dir'])).readlines()

    mean = float(file[0].split()[-1])
    var = float(file[1].split()[-1])

    return [mean, var]


def normalize_input(input, mean, var):
    if var <= 0:
        var = 1
    return input.add(-mean).div(var)


def remove_buds(label):
    label[label==2] = 1
    return label


def plot_confusion_matrix(y_true, y_pred, epoch, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = [0,1,2,3,4,5]
    np.set_printoptions(precision=2)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

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
    # return ax

    # plt.show()

    fig.savefig('confusion_matrices/' + str(epoch) + '.png')



if __name__ == '__main__':
    device = select_device(force_cpu=False)

    unet = UNet(in_channel=1, out_channel=6)  # out_channel represents number of segments desired
    unet = unet.to(device)

    paths = get_paths()

    training_set = PatchDataset(paths['out_dir'], device)
    validation_set = PatchDataset(paths['val_dir'], device, use_wmap=False)

    train_UNet(device, unet, training_set, validation_set, width_out=164, height_out=164)
