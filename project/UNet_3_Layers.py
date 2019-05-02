import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from device import select_device
from parameters import get_paths
from PatchDataset import PatchDataset
from torch.utils.data import DataLoader


# UNet layer sizes
layer1_size = 64
layer2_size = 128
layer3_size = 256


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        # Downwards encoding part
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=layer1_size)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv_encode2 = self.contracting_block(in_channels=layer1_size, out_channels=layer2_size)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=layer2_size, out_channels=layer3_size),
                            nn.ReLU(),
                            nn.BatchNorm2d(layer3_size),
                            nn.Conv2d(kernel_size=3, in_channels=layer3_size, out_channels=layer3_size),
                            nn.ReLU(),
                            nn.BatchNorm2d(layer3_size),
                            nn.ConvTranspose2d(in_channels=layer3_size, out_channels=layer2_size, kernel_size=3, stride=2,
                                               padding=1, output_padding=1)
                                              )

        # Upwards decoding part
        self.conv_decode2 = self.expansive_block(layer3_size, layer2_size, layer1_size)
        self.final_layer = self.final_block(layer2_size, layer1_size, out_channel)


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
        print(self.conv_encode1[0].weight.shape)

        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool2)

        # Decode
        decode_block2 = self.crop_and_concat(bottleneck1, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)

        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)

        return final_layer


def train_UNet(device, unet, dataset, width_out, height_out, epochs=1):
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(unet.parameters(), lr=0.01, momentum=0.99)
    optimizer.zero_grad()

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for epoch in range(epochs):
        # for batch_ndx, sample in enumerate(loader):
        #     print(batch_ndx, sample['patch_name'])
        #
        #     raw = sample['raw']
        #     label = sample['label']
        #

        for i in range(1):#(len(dataset)):  # TODO: remove slice
            print(dataset[i]['patch_name'])

            raw = dataset[i]['raw']
            labels = dataset[i]['label']

            outputs = unet(raw[None][None])  # None will add the missing dimensions at the front

            # print('\nraw.shape: {}\n'.format(raw.shape))
            # print('outputs.shape: {}'.format(outputs.shape))
            # print('outputs.shape[0]: {}\n'.format(outputs[0].shape))

            plot_tensors(raw, outputs)

            # permute such that number of desired segments would be on 4th dimension
            outputs = outputs.permute(0, 2, 3, 1)
            m = outputs.shape[0]



            # Resizing the outputs and label to caculate pixel wise softmax loss
            # outputs = outputs.resize(m * width_out * height_out, 2)
            # labels = labels.resize(m * width_out * height_out)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()


def plot_tensors(raw, output):
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


if __name__ == '__main__':
    device = select_device(force_cpu=True)

    unet = UNet(in_channel=1, out_channel=5)  # out_channel represents number of segments desired
    unet = unet.to(device)

    paths = get_paths()
    patches = PatchDataset(paths['out_dir'], device)

    train_UNet(device, unet, patches, 348, 348, 1)
