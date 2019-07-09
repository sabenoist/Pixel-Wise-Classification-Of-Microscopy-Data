import os
import torch
import torch.nn as nn
import sys, getopt

from device import select_device
from parameters import get_paths
from PatchDataset import PatchDataset
from WeightedCrossEntropyLoss import WeightedCrossEntropyLoss
from torch.utils.data import DataLoader
from torch.autograd import Variable
from UNet import UNet


import warnings
warnings.filterwarnings('ignore')


def train_UNet(model_name, device, unet, training_set, validation_set, width_out, height_out, epochs=1, lr=10e-9):
    """
    The training algorithm that is used to train the models.
    In its current state it uses the CrossEntropyLoss class
    from Pytorch to compute the loss values and updates the
    gradients through Stochastic Gradient Descent. Weight
    mapping has not been implemented yet and for this the
    WeightedCrossEntropyLoss class should be used.
    """

    criterion = nn.CrossEntropyLoss().to(device)

    # the Stochastic Gradient Descent optimizer from Pytorch.
    optimizer = torch.optim.SGD(unet.parameters(), lr=lr, momentum=0.99)
    optimizer.zero_grad()  # sets the gradient to accumulate instead of replace.

    loss_info = list()
    validation_info = list()

    mean, var = read_mean_var()

    for epoch in range(epochs):
        batch_size = 200
        patch_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)

        patches_amount = len(training_set)
        patch_counter = 0
        for batch_ndx, sample in enumerate(patch_loader):
            for i in range(len(sample['patch_name'])):
                # Forward part
                patch_name = sample['patch_name'][i]
                raw = normalize_input(sample['raw'][i], mean, var)
                label = Variable(sample['label'][i])
                # wmap = sample['wmap'][i]  TODO: uncomment when wmap support has been implemented.

                output = unet(raw[None][None])  # None will add the missing dimensions at the front, the Unet requires a 4d input for the weights.

                # Backwards part
                output = output.permute(0, 2, 3, 1)  # permute such that number of desired segments would be on 4th dimension
                label = label.unsqueeze(0)  # sets label to have the same dimensions as output
                m = output.shape[0]

                # Resizing the outputs and label to calculate pixel wise softmax loss
                output = output.resize(m * width_out * height_out, 6)
                label = label.resize(m * width_out * height_out, 6)
                # wmap = wmap.resize(m * width_out * height_out, 1)  TODO: uncomment when wmap support has been implemented.

                loss = criterion(output, torch.argmax(label, 1))  # Pytorch Cross-Entropy Loss function does not accept one-hot encoded ground-truths
                loss.backward()  # performs the back-propagation of the loss value
                grad_Clamp(unet.parameters(), clip=5)

                optimizer.step() # performs the weight updates.

                # save loss info per 100 images
                if patch_counter + 1 % 100 == 0:
                    loss_info.append([epoch * patches_amount, patch_counter, loss.item()])

                # perform validation per 2000 images
                if patch_counter + 1 % 2000 == 0:
                    validation_info.append([epoch * patches_amount, patch_counter, run_validation(device, unet, validation_set, width_out, height_out)])

                # save model every 10000 images
                if patch_counter + 1 % 10000 == 0 and patch_counter != 0:
                    save_model(unet, paths['model_dir'], model_name + '_epoch_' + str(epoch) + '_patch_' + str(patch_counter) + '.pickle')

                print('{}. [{}/{}] - {} loss: {}'.format(epoch + 1, patch_counter + 1, patches_amount, patch_name, loss))

                patch_counter += 1

    save_model(unet, paths['model_dir'], model_name + '.pickle')
    save_loss_info(loss_info, paths['model_dir'], model_name + '_loss.txt')
    save_loss_info(validation_info, paths['model_dir'], model_name + '_validation.txt')


def run_validation(device, unet, validation_set, width_out, height_out):
    """
    This function will perform the validation test with the
    images from the validation_set. Its cross-entropy loss
    values are collected, averaged and then returned. The
    loss values are computed by using the built-in
    CrossEntropyLoss class from Pytorch.
    """

    print("running validation test")

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
            label = Variable(sample['label'][i])

            output = unet(raw[None][None])  # None will add the missing dimensions at the front, the Unet requires a 4d input for the weights.

            output = output.permute(0, 2, 3, 1)  # permute such that number of desired segments would be on 4th dimension

            m = output.shape[0]
            label = label.unsqueeze(0)  # sets label to have the same dimensions as output

            # Resizing the outputs and label to calculate pixel wise softmax loss
            output = output.resize(m * width_out * height_out, 6)
            label = label.resize(m * width_out * height_out, 6)

            loss = validation_criterion(output, torch.argmax(label, 1))  # Pytorch Cross-Entropy Loss function does not accept one-hot encoded ground-truths

            print('validation. [{}/{}] {} - loss: {}'.format(validation_counter + 1, validation_amount, patch_name, loss.item()))

            losses.append(loss.item())

            validation_counter += 1

    unet.train()  # sets the network to start learning again

    return sum(losses) / len(losses)


def save_model(unet, path, name):
    """
    Saves the model in a pickle file so that it later can be used for
    classification or further training.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(unet.state_dict(), path + name)


def save_loss_info(loss_info, path, name):
    """
    Saves the collected loss value in a file for later inspection and plotting.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    file = open(path + name, 'w+')
    file.write('patch_counter loss\n')

    for info in loss_info:
        file.write('{} {} \n'.format(info[0] + info[1] + 1, info[2]))

    file.close()


def read_mean_var(training=True):
    """
    Reads the mean and variance from the patch_mean_var.txt file
    """

    if training:
        file = open('{}/patch_mean_var.txt'.format(paths['out_dir'])).readlines()
    else:
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


def grad_Clamp(parameters, clip=5):
    """
    Puts clamps on the gradient to prevent it from exploding.
    """

    for p in parameters:
        p.grad.data.clamp_(min=-clip, max=clip)


if __name__ == '__main__':
    device = select_device(force_cpu=False)

    unet = UNet(in_channel=1, out_channel=6)  # out_channel represents number of segments desired
    unet = unet.to(device)

    paths = get_paths()

    training_set = PatchDataset(paths['out_dir'], device, use_wmap=False)  # TODO: use_wmap=True when implemented.
    validation_set = PatchDataset(paths['val_dir'], device, use_wmap=False)

    train_UNet(model_name, device, unet, training_set, validation_set, width_out=164, height_out=164, epochs=epochs, lr=learning_rate)