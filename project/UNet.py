import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from parameters import *



class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        #Encoding part
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(in_channels=in_channel, out_channels=128)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)

        #Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
                            nn.ReLU(),
                            nn.BatchNorm2d(512),
                            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
                            nn.ReLU(),
                            nn.BatchNorm2d(512),
                            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
                                              )

        #Decoding part
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)


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


    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)

        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)

        final_layer = self.final_layer(decode_block1)

        return  final_layer


def train_UNet(unet, inputs, width_out, height_out, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99)
    optimizer.zero_grad()

    for epoch in range(epochs):
        for input in inputs:
            output = unet(input)

            # permute such that number of desired segments would be on 4th dimension
            output = output.permute(0, 2, 3, 1)
            m = output.shape[0]

            # Resizing the outputs and label to calculate pixel wise softmax loss
            output = output.resize(m * width_out * height_out, 2)
            labels = labels.resize(m * width_out * height_out)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()


def select_device():
    if torch.cuda.is_available():
        try:
            gpu_test = torch.empty(5, 5)
            gpu_test.to(torch.device('cuda'))

            return torch.device('cuda')  # GPU
        except:
            print('Warning: GPU too old to be used or CUDA is corrupt. Using CPU instead.')

            return torch.device('cpu')  # CPU
    else:
        print('Warning: Incompatible GPU found. Using CPU instead.')

        return torch.device('cpu')  # CPU


if __name__ == '__main__':
    paths = get_paths()
    training_data = [img for img in os.listdir(paths['label_dir']) if img.endswith('.tif')]

    unet = UNet(in_channel=5, out_channel=2)  # out_channel represents number of segments desired
    unet.to(device=select_device())

    train_UNet(unet, training_data, 164, 164, 1)
