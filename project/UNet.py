import torch
import torch.nn as nn
import torch.nn.functional as F

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



