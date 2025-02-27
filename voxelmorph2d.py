import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from skimage import img_as_float32
import skimage.io as io
import cv2
import math
import matplotlib
from unet_parts import *

matplotlib.use('Agg')
from matplotlib import pyplot as plt


import numpy as np


torch.manual_seed(42)
torch.cuda.manual_seed(42)
use_gpu = torch.cuda.is_available()


threeorgan_save_quiver_path = './results/grid/'
#save_grid_path = './new results_2/grid/'
# threeorgan_save_quiver_path = './predict/grid/'
# Bladder_save_quiver_path = './predict/Bladder_grid/'
# Cervical_save_quiver_path = './predict/Cervical_grid/'
# Rectum_save_quiver_path = './predict/Rectum_grid/'

# import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
# import numpy as np
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# use_gpu = torch.cuda.is_available()

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, n_convnum=64 ,bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, n_convnum)
        self.down1 = Down(n_convnum, n_convnum*2)
        self.down2 = Down(n_convnum*2, n_convnum*4)
        self.down3 = Down(n_convnum*4, n_convnum*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_convnum*8, n_convnum*16 // factor)
        self.dropout = nn.Dropout(p=0.5)
        self.up1 = Up(n_convnum*16, n_convnum*8 // factor, bilinear)
        self.up2 = Up(n_convnum*8, n_convnum*4 // factor, bilinear)
        self.up3 = Up(n_convnum*4, n_convnum*2 // factor, bilinear)
        self.up4 = Up(n_convnum*2, n_convnum, bilinear)
        self.outc = OutConv(n_convnum, n_classes)

    def forward(self, x,y):
        x = torch.cat([x, y], dim=3).permute(0,3,1,2)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits.permute(0, 2, 3, 1)

class laplacian(nn.Module):
    def __init__(self):
        super(laplacian, self).__init__()
        kernel = [[0.0, 1.0, 0.0],
                  [1.0, -4.0, 1.0],
                  [0.0, 1.0, 0.0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, x):
        x1 = x[:,:1,:,:]
        x2 = x[:,1:,:,:]
        x1 = F.conv2d(x1, self.weight, padding=1)
        x2 = F.conv2d(x2, self.weight, padding=1)
        x = torch.cat([x1, x2], dim=1)
        return x


class VFF(nn.Module):
    def __init__(self, mode, outchannels=64):
        super(VFF, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = 8, out_features = 16, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = 16, out_features = 24, bias = False)
            #nn.Linear(in_features=16, out_features=6, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.lap = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, stride=1, padding=2)
        #  self.lap = laplacian()
        self.reverseconv = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3,stride=1, padding=1)
        self.out_layers = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=outchannels, kernel_size=1, stride=1, padding=0)
        )
        # self.out_layers = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=16, kernel_size=1, stride=1, padding=0),
        #     # nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=outchannels, kernel_size=1, stride=1, padding=0)
        # )
        self.conv = nn.Conv2d(in_channels=8, out_channels=outchannels, kernel_size=1, stride=1, padding=0)
    def forward(self, x1,x2,x3,sup):
        #b, _, _, _ = x1.shape
        # supcontrast = torch.zeros(sup.size()).cuda()
        v = torch.cat([x1, x2, x3,sup], dim=3).permute(0,3,1,2)
        x1 = x1.permute(0,3,1,2)
        x2 = x2.permute(0,3,1,2)
        x3 = x3.permute(0,3,1,2)
        b, _, _, _ = x1.shape
        # v = sup
        v = self.global_pooling(v).view(b, 8)
        # v = self.fc_layers(v).view(b, 6, 1, 1)
        v = self.fc_layers(v).view(b, 24, 1, 1)
        v = self.sigmoid(v)

        v1,v2,v3 = v.split([8,8,8],1)
        # v1, v2, v3 = v.split([2, 2, 2], 1)

        x1 = self.lap(x1)
        x2 = self.lap(x2)
        x3 = self.lap(x3)

        # a = v1+v2+v3
        # v1 = v1/a
        # v2 = v2/a
        # v3 = v3/a

        res = v1*x1+v2*x2+v3*x3
        # res = self.reverseconv(res)
        res = self.out_layers(res)
        return res



class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).contiguous().view(b, c, -1)
        # [N, H * W, C/2]
        y_theta = self.conv_theta(y).contiguous().view(b, c, -1).permute(0, 2, 1)
        x_g = self.conv_g(x).contiguous().view(b, c, -1).permute(0, 2, 1)
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(y_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


def gridshow():
    gridelment = torch.ones([192, 320]) * 255
    for i in range(0, 192, 10):
        gridelment[i, :] = 0

    for j in range(0, 320, 10):
        gridelment[:, j] = 0

    gridelment = gridelment.cuda()
    return gridelment


def plot_grid(ax, gridx,gridy, **kwargs):
    for i in range(gridx.shape[0]):
        ax.plot(gridx[i,:], gridy[i,:], **kwargs)
    for i in range(gridx.shape[1]):
        ax.plot(gridx[:,i], gridy[:,i], **kwargs)


def draw_grid(im, grid_size):
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(1,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(1,))


class deformation_mix(nn.Module):
    def block(self, in_channels, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.LeakyReLU(0.2),
                )
        return block
    def __init__(self, in_channel, out_channel):
        super(deformation_mix, self).__init__()
        self.layer = self.block(in_channel, out_channel)

    def forward(self, x):
        final_layer = self.layer(x)
        return final_layer



class deformation_mix(nn.Module):
    def block(self, in_channels, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.LeakyReLU(0.2)
                )
        return block
    def __init__(self, in_channel, out_channel):
        super(deformation_mix, self).__init__()
        self.layer = self.block(in_channel, out_channel)

    def forward(self, x):
        final_layer = self.layer(x)
        return final_layer


class noMMoE(nn.Module):
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.LeakyReLU(0.2),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.LeakyReLU(0.2),
                )
        return block

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    # feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)
    def __init__(self):
        super(noMMoE, self).__init__()

        '''Tower1'''
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, 2)
        '''Tower2'''
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, 2)
        '''Tower3'''
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, 2)

    def forward(self, x, encode_block1, encode_block2, encode_block3):
        decode_block3 = self.crop_and_concat(x, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer_1 = self.final_layer(decode_block1)

        decode_block3 = self.crop_and_concat(x, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer_2 = self.final_layer(decode_block1)

        decode_block3 = self.crop_and_concat(x, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer_3 = self.final_layer(decode_block1)

    #else:
        #out1 = self.tower1(towers)
        #out2 = self.tower2(towers)

        return final_layer_1.permute(0, 2, 3, 1), final_layer_2.permute(0, 2, 3, 1), final_layer_3.permute(0, 2, 3, 1)


# Model = MMoE(feature_dim=112, expert_dim=32, n_expert=4, n_task=2, use_gate=True)
#
# nParams = sum([p.nelement() for p in Model.parameters()])
# print('* number of parameters: %d' % nParams)




class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
        )
        return block



    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=16)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(16, 32)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(32, 64)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.cross_model = NonLocalBlock(channel=64)
        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm2d(mid_channel * 2),
                                torch.nn.LeakyReLU(0.2),
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.LeakyReLU(0.2),
                                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.LeakyReLU(0.2),
                            )
        # Decode


    def forward(self, moving_image, fixed_image):
        # # Encode
        # encode_block1 = self.conv_encode1(x)
        # encode_pool1 = self.conv_maxpool1(encode_block1)
        # encode_block2 = self.conv_encode2(encode_pool1)
        # encode_pool2 = self.conv_maxpool2(encode_block2)
        # encode_block3 = self.conv_encode3(encode_pool2)
        # encode_pool3 = self.conv_maxpool3(encode_block3)
        x = moving_image.permute(0,3,1,2)
        y = fixed_image.permute(0,3,1,2)
        encode_block1_x = self.conv_encode1(x)
        encode_pool1_x = self.conv_maxpool1(encode_block1_x)

        encode_block2_x = self.conv_encode2(encode_pool1_x)
        encode_pool2_x = self.conv_maxpool2(encode_block2_x)
        encode_block3_x = self.conv_encode3(encode_pool2_x)
        encode_pool3_x = self.conv_maxpool3(encode_block3_x)

        encode_block1_y = self.conv_encode1(y)
        encode_pool1_y = self.conv_maxpool1(encode_block1_y)
        encode_block2_y = self.conv_encode2(encode_pool1_y)
        encode_pool2_y = self.conv_maxpool2(encode_block2_y)
        encode_block3_y = self.conv_encode3(encode_pool2_y)
        encode_pool3_y = self.conv_maxpool3(encode_block3_y)

        # TransformerEncoderLayer(embed_dim=embed_dim,
        #                    num_heads=self.num_heads,
        #                    layers=max(self.layers, layers),
        #                    attn_dropout=attn_dropout,
        #                    relu_dropout=self.relu_dropout,
        #                    res_dropout=self.res_dropout,
        #                    embed_dropout=self.embed_dropout,
        #                    attn_mask=self.attn_mask)

        crossmoedl_x = self.cross_model(encode_pool3_x, encode_pool3_y)
        crossmoedl_x_1 = self.cross_model(crossmoedl_x, encode_pool3_y)
        crossmoedl_x_2 = self.cross_model(crossmoedl_x_1, encode_pool3_y)
        crossmoedl_x_3 = self.cross_model(crossmoedl_x_2, encode_pool3_y)
        crossmoedl_x_4 = self.cross_model(crossmoedl_x_3, encode_pool3_y)

        crossmoedl_y = self.cross_model(encode_pool3_y, encode_pool3_x)
        crossmoedl_y_1 = self.cross_model(crossmoedl_y, encode_pool3_x)
        crossmoedl_y_2 = self.cross_model(crossmoedl_y_1, encode_pool3_x)
        crossmoedl_y_3 = self.cross_model(crossmoedl_y_2, encode_pool3_x)
        crossmoedl_y_4 = self.cross_model(crossmoedl_y_3, encode_pool3_x)

        # # contact
        encode_block1 = torch.cat([encode_block1_x, encode_block1_y], dim=1)
        encode_block2 = torch.cat([encode_block2_x, encode_block2_y], dim=1)
        encode_block3 = torch.cat([encode_block3_x, encode_block3_y], dim=1)
        encode_pool3 = torch.cat([crossmoedl_x_4,crossmoedl_y_4],dim=1)
        # # Bottleneck

        bottleneck1 = self.bottleneck(encode_pool3)

        # Decode

        return bottleneck1, encode_block1, encode_block2, encode_block3

class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu==True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda()
        return torch.squeeze(torch.reshape(x, (-1, 1)))


    def interpolate(self, im, x, y):

        im = F.pad(im, (0,0,1,1,1,1,0,0))

        batch_size, height, width, channels = im.shape

        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float().cuda()
        dim, _ = idx_a.transpose(1,0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1, 0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1, 0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1, 0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1, 0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1,0)
        wb = (dx * (1-dy)).transpose(1,0)
        wc = ((1-dx) * dy).transpose(1,0)
        wd = ((1-dx) * (1-dy)).transpose(1,0)

        output = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)


class VoxelMorph2d(nn.Module):
    def __init__(self, in_channels, use_gpu=False):
        super(VoxelMorph2d, self).__init__()
        self.unet = UNet(in_channels, 2)
        self.spatial_transform = SpatialTransformation(use_gpu)
        self.vff = VFF(mode="max")
        self.output = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.deformation_mix = deformation_mix(6, 2)
        self.field_sup = UNet2(2, 2, 32)
        self.noMMoe = noMMoE()
        if use_gpu:
            self.unet = self.unet.cuda()
            self.vff = self.vff.cuda()
            self.output = self.output.cuda()
            self.field_sup = self.field_sup.cuda()
            self.deformation_mix = self.deformation_mix.cuda()
            self.spatial_transform = self.spatial_transform.cuda()
            self.noMMoe = self.noMMoe.cuda()


    def forward(self, id, moving_image, fixed_image,Bladder_fixed_image_label, Cervical_fixed_image_label, Rectum_fixed_image_label, Bladder_moving_image_label, Cervical_moving_image_label, Rectum_moving_image_label):

        # x = torch.cat([moving_image, fixed_image], dim=3).permute(0,3,1,2)

        bottleneck1, encode_block1, encode_block2, encode_block3 = self.unet(moving_image,fixed_image)
        #.permute(0,2,3,1)
        deformation_matrix_1, deformation_matrix_2, deformation_matrix_3 = self.noMMoe(bottleneck1, encode_block1, encode_block2, encode_block3)

        # deformation_matrix = deformation_matrix_1+deformation_matrix_2+deformation_matrix_3
        sup = self.field_sup(Bladder_moving_image_label + Cervical_moving_image_label + Rectum_moving_image_label,
                             Bladder_fixed_image_label + Cervical_fixed_image_label + Rectum_fixed_image_label)
        ss = self.vff(deformation_matrix_1, deformation_matrix_2, deformation_matrix_3, sup)
        deformation_matrix = self.output(ss).permute(0, 2, 3, 1)

        rand_field_norm = deformation_matrix[0, :, :, :]
        fig, ax = plt.subplots()
        grid_x, grid_y = np.meshgrid(np.linspace(0, 190, 54), np.linspace(0, 320, 32))  # 生产网格大小
        plot_grid(ax, grid_x, grid_y, color="lightgrey")
        distx = torch.from_numpy(grid_x).cuda() + rand_field_norm[0::6, 0::6, 0]  # 间隔取变形场
        disty = torch.from_numpy(grid_y).cuda() + rand_field_norm[0::6, 0::6, 1]
        plot_grid(ax, distx.cpu().detach().numpy(), disty.cpu().detach().numpy(), color="C0")
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        #plt.show()
        plt.savefig(threeorgan_save_quiver_path + id[0] + '.png')  # 保存图像

        plt.cla()
        plt.close("all")

        # deformation_matrix_cat = torch.cat([deformation_matrix_1, deformation_matrix_2, deformation_matrix_3],
        #                                    dim=3).permute(0, 3, 1, 2)
        # deformation_matrix = self.deformation_mix(deformation_matrix_cat).permute(0, 2, 3, 1)

        registered_image = self.spatial_transform(moving_image, deformation_matrix)
        return registered_image, deformation_matrix


def correlation(imgA, imgB):
    cols = imgA.shape[1]
    rows = imgA.shape[2]
    cols1 = imgB.shape[0]
    rows1 = imgB.shape[1]
    # print(cols,rows)
    # print(cols1,rows1)
    # print("imgage:",imgA[0,0])
    A = 0
    B = 0
    C = 0
    for i in torch.arange(0, cols).reshape(-1):
        for j in torch.arange(0, rows).reshape(-1):
            C = C + (imgA[0, i, j, 0] * imgB[0, i, j, 0])
            A = A + (imgA[0, i, j, 0]) ** 2
            B = B + (imgB[0, i, j, 0]) ** 2

    cc = C / (math.sqrt(A) * math.sqrt(B))
    return cc


def cosine_similarity(img1, img2):

    sizes = np.prod(img1.shape[1:])
    flatten1 = img1.view( -1, sizes)
    flatten2 = img2.view(-1, sizes)


    mean1 = torch.mean(flatten1).view(-1, 1)
    mean2 = torch.mean(flatten2).view(-1, 1)
    var1 = torch.mean(torch.square(flatten1 - mean1))
    var2 = torch.mean(torch.square(flatten2 - mean2))
    cov12 = torch.mean(
        (flatten1 - mean1) * (flatten2 - mean2))
    pearson_r = cov12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))

    raw_loss = 1 - pearson_r
    raw_loss = torch.mean(raw_loss)
    # print(raw_loss)
    return raw_loss*5

def cross_correlation_loss(I, J, n):
    I = I.permute(0, 3, 1, 2)
    J = J.permute(0, 3, 1, 2)
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = torch.ones((1, channels, n, n))
    if use_gpu:
        sum_filter = sum_filter.cuda()
    I_sum = torch.conv2d(I, sum_filter, padding=1, stride=(1,1))
    J_sum = torch.conv2d(J, sum_filter,  padding=1 ,stride=(1,1))
    I2_sum = torch.conv2d(I2, sum_filter, padding=1, stride=(1,1))
    J2_sum = torch.conv2d(J2, sum_filter, padding=1, stride=(1,1))
    IJ_sum = torch.conv2d(IJ, sum_filter, padding=1, stride=(1,1))
    win_size = n**2
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
    cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
    return torch.mean(cc)


def hxx(m, f):
    x = m
    y = f
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    size = x.shape[-1]
    px = np.histogram(x, 256, (0, 255))[0] / size
    py = np.histogram(y, 256, (0, 255))[0] / size
    hx = - np.sum(px * np.log(px + 1e-8))
    hy = - np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(x, y, 256, [[0, 255], [0, 255]])[0]
    hxy /= (1.0 * size)
    hxy = - np.sum(hxy * np.log(hxy + 1e-8))

    r = hx + hy - hxy
    return r


def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d/2.0

def vox_morph_loss(y, ytrue, n=2, lamda=0.01):
    cc = cross_correlation_loss(y, ytrue, n)
    # mi = hxx(y, ytrue)
    # ncc = cosine_similarity(y, ytrue)
    sm = smooothing_loss(y)
    #print("CC Loss", cc, "Gradient Loss", sm)
    loss = -1.0 * cc + lamda * sm
    return loss


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def dice_score(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    target = target.cuda()
    top = 2 * torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    #print("Dice score", dice)
    return dice

def dice_score_2(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    pred[pred > 0] = 1
    target[target > 0] = 1
    top = 2 * torch.sum(pred * target)
    union = np.sum(pred + target)
    #eps = np.ones_like(union) * 1e-5
    #bottom = np.max(union, eps)
    dice = np.mean(top / union)
    #print("Dice score", dice)
    return dice

def Voxelmorph(input_dims):
    model = VoxelMorph2d(input_dims[0] * 2)
    print('using VoxelMorph2d')
    return model