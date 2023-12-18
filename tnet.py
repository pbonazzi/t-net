import pdb
import torch
import torch.nn as nn


class Conv_1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_1x1, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv_1x1(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, return_residual=False):
        super(ConvBlock, self).__init__()
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.batch_conv_1x1 = Conv_1x1(out_channels, out_channels)
        self.group_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels
        )
        self.batch_norm_3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.return_residual = return_residual

    def forward(self, x):
        x_1 = self.relu(self.batch_norm_1(self.conv_3x3(x)))
        x_2 = self.batch_conv_1x1(x_1)
        x_3 = self.relu(self.batch_norm_3(self.group_conv(x_2)))

        if self.return_residual:
            return x_1, x_3
        else:
            return x_3


class TNet(nn.Module):
    def __init__(self, num_filters=[16, 32, 64]):
        super(TNet, self).__init__()

        # Encoder blocks
        self.enc_block1 = ConvBlock(3, num_filters[0], return_residual=True)

        self.enc_block2 = ConvBlock(
            num_filters[0], num_filters[1], return_residual=True
        )
        self.enc_1x1_block2 = Conv_1x1(num_filters[0], num_filters[1])

        self.enc_block3 = ConvBlock(
            num_filters[1], num_filters[2], return_residual=True
        )
        self.enc_1x1_block3 = Conv_1x1(num_filters[1], num_filters[2])

        # Intermediate block
        self.intermediate_block = ConvBlock(num_filters[2], num_filters[2])
        self.intermediate_1x1_block = Conv_1x1(num_filters[2], num_filters[2])

        # Decoder blocks
        self.dec_block1 = ConvBlock(num_filters[2], num_filters[1])
        self.dec_1x1_block1 = Conv_1x1(num_filters[2], num_filters[1])

        self.dec_block2 = ConvBlock(num_filters[1], num_filters[0])
        self.dec_1x1_block2 = Conv_1x1(num_filters[1], num_filters[0])

        self.dec_block3 = ConvBlock(num_filters[0], num_filters[0])
        self.dec_1x1_block3 = Conv_1x1(num_filters[0], num_filters[0])

        self.output = nn.Conv2d(num_filters[0], 1, kernel_size=3, padding=1)

        # Max pooling and unpooling layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x1_res, x1 = self.enc_block1(x)
        x1, x1_ind = self.maxpool(x1)

        skip = self.enc_1x1_block2(x1)
        x2_res, x2 = self.enc_block2(x1)
        x2 = x2 + skip
        x2, x2_ind = self.maxpool(x2)

        skip = self.enc_1x1_block3(x2)
        x3_res, x3 = self.enc_block3(x2)
        x3 = x3 + skip
        x3, x3_ind = self.maxpool(x3)

        # Intermediate
        skip = self.intermediate_1x1_block(x3)
        x_intermediate = self.intermediate_block(x3)
        x_intermediate = x_intermediate + skip

        # Decoder
        x_dec1 = self.unpool(x_intermediate, x3_ind)
        skip = self.dec_1x1_block1(x_dec1)
        x_dec1 = x_dec1 + x3_res
        x_dec1 = self.dec_block1(x_dec1)
        x_dec1 = x_dec1 + skip

        x_dec2 = self.unpool(x_dec1, x2_ind)
        skip = self.dec_1x1_block2(x_dec2)
        x_dec2 = x_dec2 + x2_res
        x_dec2 = self.dec_block2(x_dec2)
        x_dec2 = x_dec2 + skip

        x_dec3 = self.unpool(x_dec2, x1_ind)
        skip = self.dec_1x1_block3(x_dec3)
        x_dec3 = x_dec3 + x1_res
        x_dec3 = self.dec_block3(x_dec3)
        x_dec3 = x_dec3 + skip

        pred = self.output(x_dec3)

        return pred


if __name__ == "__main__":
    model = TNet()
    pred = model(torch.randn(1, 3, 512, 512))
    print(pred.shape)
