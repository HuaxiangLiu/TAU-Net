from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from nnunet.network_architecture.neural_network import SegmentationNetwork
# import numpy as np
# from .resnet_model import *
from torchvision import models


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class Multiscaleconv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(Multiscaleconv_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv3(x)+ self.conv5(x)+self.conv1(x)
        return x

class Multiscaleconv_block_v0(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(Multiscaleconv_block_v0, self).__init__()

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv3(x)+ self.conv5(x)   #+self.conv1(x)
        return x

class MultiscaleFuse(nn.Module):
    """
    Convolution Block
    """

    def __init__(self):
        super(MultiscaleFuse, self).__init__()

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x1 = self.down(x)
        x  = self.up(x)

        x1 = self.up(x1)
        x1 = self.up(x1)
        return x+x1

class MultiscaleFuse1(nn.Module):
    """
    Convolution Block
    """

    def __init__(self):
        super(MultiscaleFuse1, self).__init__()

        self.up =  nn.Upsample(scale_factor=2)
        # self.down = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        # x1 = self.down(x)
        x  = self.up(x)

        # x1 = self.up(x1)
        # x1 = self.up(x1)
        return x  #+x1

class Resconv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(Resconv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=nn.ReLU(inplace=True)) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation is not None:
            layers.append(activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class Multiscaleconv_small(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(Multiscaleconv_small, self).__init__()

        self.conv1 = BN_Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.conv3 = nn.Sequential(
            BN_Conv2d(in_ch, out_ch//2, 3, 1, 1, bias=False),
            BN_Conv2d(out_ch//2, out_ch, 1, 1, 0, bias=False),
        )
        self.conv5 = nn.Sequential(
            BN_Conv2d(in_ch, out_ch//2, 3, 1, 1, bias=False),
            BN_Conv2d(out_ch//2, out_ch, 3, 1, 1, bias=False),
            BN_Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        )
    def forward(self, x):
        x = self.conv3(x)+ self.conv5(x)+self.conv1(x)
        return x

# Reduce the number of channel
class TAU_Net_small(SegmentationNetwork):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=3):
        super(TAU_Net_small, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16, 32, 64, 128, 256, 512]

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv0 = Multiscaleconv_small(in_ch, filters[0])
        self.MConv1 = Multiscaleconv_small(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_small(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_small(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_small(filters[3], filters[4])
        # self.MConv5 = Multiscaleconv_small(filters[4], filters[5])

        # self.dconv1 = conv_block(filters[0], filters[1])
        # self.dconv2 = conv_block(filters[1], filters[2])
        # self.dconv3 = conv_block(filters[2], filters[3])
        # self.dconv4 = conv_block(filters[3], filters[4])
        # self.dconv5 = conv_block(filters[4], filters[5])

        self.inconv0 = conv_block(in_ch, filters[0])
        self.inconv1 = Resconv_block(filters[0], filters[0])
        self.inconv2 = Resconv_block(filters[1], filters[1])
        self.inconv3 = Resconv_block(filters[2], filters[2])
        self.inconv4 = Resconv_block(filters[3], filters[3])
        # self.inconv5 = Resconv_block(filters[4], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.mup = MultiscaleFuse()

        self.Up5 = up_conv(filters[4], filters[3], 2)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2], 2)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], 2)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], 2)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        # e1 = self.inconv0(x)   #16*512*512
        e1 = self.MConv0(x)
        # rese1 = self.resconv1(e1)  #32*512*512
        e2 = self.Maxpool1(e1)    #16*256*256
        # e2 = self.MConv1(e2)
        m2 = self.mup(e2)   #16*512*512
        m2 = m2 + e1
        m2 =self.inconv1(m2)#16*512*512

        # e2 = self.dconv1(e2)#32*256*256
        e2 = self.MConv1(e2)
        # rese2 = self.resconv2(e2)
        e3 = self.Maxpool2(e2)
        # e3 = self.MConv2(e3)
        m3 = self.mup(e3)
        m3 = m3 + e2
        m3 = self.inconv2(m3)

        # e3 = self.dconv2(e3)#64*128*128
        e3 = self.MConv2(e3)
        # rese3 = self.resconv3(e3)
        e4 = self.Maxpool3(e3)   #64*64*64
        # e4 = self.MConv3(e4)
        m4 = self.mup(e4)
        m4 = m4 + e3
        m4 = self.inconv3(m4)

        # e4 = self.dconv3(e4)#128*64*64
        e4 = self.MConv3(e4)
        # rese4 = self.resconv4(e4)
        e5 = self.Maxpool4(e4)
        # e5 = self.MConv4(e5)
        m5 = self.mup(e5)
        m5 = m5 +e4
        m5 = self.inconv4(m5)

        e6 = self.MConv4(e5)  # 256*32*32

        d5 = self.Up5(e6)
        d5 = torch.cat((m5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((m4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((m3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((m2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class TAU_Net_Fusion(SegmentationNetwork):
    """
    # Fusing the multiscale features 
  
    """

    def __init__(self, in_ch=1, out_ch=3):
        super(TAU_Net_Fusion, self).__init__()

        n = 2

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16*n, 32*n, 64*n, 128*n, 256*n, 512*n]

        # self.up =  nn.Upsample(scale_factor=2)
        # self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv0 = Multiscaleconv_block(in_ch, filters[0])
        self.MConv1 = Multiscaleconv_block(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_block(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_block(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_block(filters[3], filters[4])
        # self.MConv5 = Multiscaleconv_block(filters[4], filters[5])

        # self.dconv1 = conv_block(filters[0], filters[1])
        # self.dconv2 = conv_block(filters[1], filters[2])
        # self.dconv3 = conv_block(filters[2], filters[3])
        # self.dconv4 = conv_block(filters[3], filters[4])
        # self.dconv5 = conv_block(filters[4], filters[5])

        # self.inconv0 = conv_block(in_ch, filters[0])
        self.inconv1 = Resconv_block(filters[0], filters[0])
        self.inconv2 = Resconv_block(filters[1], filters[1])
        self.inconv3 = Resconv_block(filters[2], filters[2])
        self.inconv4 = Resconv_block(filters[3], filters[3])
        self.inconv5 = Resconv_block(filters[4], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.mup = MultiscaleFuse()

        self.Up5 = up_conv(filters[4], filters[3], 2)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2], 2)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], 2)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], 2)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        # e1 = self.inconv0(x)   #16*512*512
        e1 = self.MConv0(x)
        # rese1 = self.resconv1(e1)  #32*512*512
        e2 = self.Maxpool1(e1)    #16*256*256
        # e2 = self.MConv1(e2)
        m2 = self.mup(e2)   #16*512*512
        m2 = m2 + e1
        m2 =self.inconv1(m2)#16*512*512

        # e2 = self.dconv1(e2)#32*256*256
        e2 = self.MConv1(e2)
        # rese2 = self.resconv2(e2)
        e3 = self.Maxpool2(e2)
        # e3 = self.MConv2(e3)
        m3 = self.mup(e3)
        m3 = m3 + e2
        m3 = self.inconv2(m3)

        # e3 = self.dconv2(e3)#64*128*128
        e3 = self.MConv2(e3)
        # rese3 = self.resconv3(e3)
        e4 = self.Maxpool3(e3)   #64*64*64
        # e4 = self.MConv3(e4)
        m4 = self.mup(e4)
        m4 = m4 + e3
        m4 = self.inconv3(m4)

        # e4 = self.dconv3(e4)#128*64*64
        e4 = self.MConv3(e4)
        # rese4 = self.resconv4(e4)
        e5 = self.Maxpool4(e4)
        # e5 = self.MConv4(e5)
        m5 = self.mup(e5)
        m5 = m5 +e4
        m5 = self.inconv4(m5)

        e6 = self.MConv4(e5)  # 256*32*32

        d5 = self.Up5(e6)
        d5 = torch.cat((m5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((m4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((m3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((m2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class inception_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(inception_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch//2),
            nn.Conv2d(out_ch//2, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch) )
        self.avgpool = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        conv3 = self.conv3(x)
        conv13=self.conv1_3(x)
        conv1 = self.conv1(x)
        pool = self.avgpool(x)
        x = self.relu(self.conv3(x)+ self.conv1_3(x)+self.conv1(x)+self.avgpool(x))
        return x


class DoubleConv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(DoubleConv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class FeatureAggregrationPlus(nn.Module):
    def __init__(self, top_ch, left_ch, down_ch):

        super(FeatureAggregrationPlus, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, top_ch // 2)

        self.conv1 = nn.Conv2d(top_ch, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, top_ch, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, top_ch, kernel_size=1, stride=1, padding=0)

        self.down_conv = nn.Sequential(
            nn.Conv2d(down_ch, top_ch, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(top_ch),
            nn.ReLU(inplace=True))

        self.conv = DoubleConv_block(top_ch + left_ch, left_ch)

    def forward(self, top, left, down):
        down = self.down_conv(down)
        down = F.interpolate(down, top.size()[2:], mode='bilinear', align_corners=True)

        n, c, h, w = down.size()
        x_h = self.pool_h(down)
        x_w = self.pool_w(down).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        top = top * a_w * a_h + top

        # left = F.interpolate(left, top.size()[2:], mode='bilinear', align_corners=True)

        out = torch.cat((top, left), dim=1)
        out = self.conv(out)

        return out

from torch.nn import Conv2d, Parameter, Softmax

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class TAU_Net_PLUS(SegmentationNetwork):
    """
    # refering the structure of UNet++
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(TAU_Net_PLUS, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16, 32, 64, 128, 256, 512]

        # self.up =  nn.Upsample(scale_factor=2)
        # self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resconv1 = Resconv_block(filters[0], filters[1])
        self.resconv2 = Resconv_block(filters[1], filters[2])
        self.resconv3 = Resconv_block(filters[2], filters[3])
        self.resconv4 = Resconv_block(filters[3], filters[4])
        self.resconv5 = Resconv_block(filters[4], filters[5])

        self.inconv0 = conv_block(in_ch, filters[0])
        self.inconv1 = Resconv_block(filters[1], filters[1])
        self.inconv2 = Resconv_block(filters[2], filters[2])
        self.inconv3 = Resconv_block(filters[3], filters[3])
        self.inconv4 = Resconv_block(filters[4], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv1 = Multiscaleconv_block_v0(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_block_v0(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_block_v0(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_block_v0(filters[3], filters[4])
        self.MConv5 = Multiscaleconv_block_v0(filters[4], filters[5])

        self.mup = MultiscaleFuse()
        self.cam = CAM_Module(filters[5])
        self.pam = PAM_Module(filters[5])

        self.fa0 = FeatureAggregrationPlus(filters[4], filters[4], filters[5])
        self.fa1 = FeatureAggregrationPlus(filters[3], filters[3], filters[5])
        self.fa2 = FeatureAggregrationPlus(filters[2], filters[2], filters[5])
        self.fa3 = FeatureAggregrationPlus(filters[1], filters[1], filters[5])

        self.Up5 = up_conv(filters[5], filters[4])
        # self.Up_conv5 = conv_block(filters[5], filters[4])

        self.Up4 = up_conv(filters[4], filters[3])
        # self.Up_conv4 = conv_block(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
        # self.Up_conv3 = conv_block(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        # self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Conv = nn.Conv2d(filters[1], out_ch, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.inconv0(x)  #512

        rese1 = self.resconv1(e1)
        e2 = self.Maxpool1(e1)
        e2 = self.MConv1(e2)
        m2 = self.mup(e2)
        m2 = m2 + rese1
        m2 =self.inconv1(m2) #512

        rese2 = self.resconv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.MConv2(e3)
        m3 = self.mup(e3)
        m3 = m3 + rese2
        m3 = self.inconv2(m3) #256

        rese3 = self.resconv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.MConv3(e4)
        m4 = self.mup(e4)
        m4 = m4 + rese3
        m4 = self.inconv3(m4) #128

        rese4 = self.resconv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.MConv4(e5)
        m5 = self.mup(e5)
        m5 = m5 +rese4
        m5 = self.inconv4(m5) #64

        rese5 = self.resconv5(e5)
        e6 = self.Maxpool5(e5)
        e6 = self.MConv5(e6)
        me6 = self.mup(e6)
        m6 = me6+rese5    #32

        pam_feat = self.pam(m6)
        cam_feat = self.cam(m6)

        e6 = pam_feat + cam_feat
        d5 = self.Up5(e6)
        out_decoder4 = self.fa0(m5, d5, e6)

        d4 = self.Up4(out_decoder4)
        out_decoder3 = self.fa1(m4, d4, e6)
        # d4 = torch.cat((e3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        d3 = self.Up3(out_decoder3)
        out_decoder2 = self.fa2(m3, d3, e6)
        # d3 = torch.cat((e2, d3), dim=1)
        # d3 = self.Up_conv3(d3)

        d2 = self.Up2(out_decoder2)
        out_decoder1 = self.fa3(m2, d2, e6)

        # d5 = self.Up5(m6)
        # d5 = torch.cat((m5, d5), dim=1)
        # d5 = self.Up_conv5(d5)
        #
        # d4 = self.Up4(d5)
        # d4 = torch.cat((m4, d4), dim=1)
        # d4 = self.Up_conv4(d4)
        #
        # d3 = self.Up3(d4)
        # d3 = torch.cat((m3, d3), dim=1)
        # d3 = self.Up_conv3(d3)
        #
        # d2 = self.Up2(d3)
        # d2 = torch.cat((m2, d2), dim=1)
        # d2 = self.Up_conv2(d2)

        out = self.Conv(out_decoder1)

        # d1 = self.active(out)

        # return tuple(out)

        return out

class TAU_Net(SegmentationNetwork):
    """
    UNet - Basic Implementation

    """

    def __init__(self, in_ch=3, out_ch=1):
        super(TAU_Net, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16, 32, 64, 128, 256, 512]

        # self.up =  nn.Upsample(scale_factor=2)
        # self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resconv1 = Resconv_block(filters[0], filters[1])
        self.resconv2 = Resconv_block(filters[1], filters[2])
        self.resconv3 = Resconv_block(filters[2], filters[3])
        self.resconv4 = Resconv_block(filters[3], filters[4])
        self.resconv5 = Resconv_block(filters[4], filters[5])

        self.inconv0 = conv_block(in_ch, filters[0])
        self.inconv1 = Resconv_block(filters[1], filters[1])
        self.inconv2 = Resconv_block(filters[2], filters[2])
        self.inconv3 = Resconv_block(filters[3], filters[3])
        self.inconv4 = Resconv_block(filters[4], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv1 = Multiscaleconv_block_v0(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_block_v0(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_block_v0(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_block_v0(filters[3], filters[4])
        self.MConv5 = Multiscaleconv_block_v0(filters[4], filters[5])

        self.mup = MultiscaleFuse1()

        self.Up5 = up_conv(filters[5], filters[4])
        self.Up_conv5 = conv_block(filters[5], filters[4])

        self.Up4 = up_conv(filters[4], filters[3])
        self.Up_conv4 = conv_block(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
        self.Up_conv3 = conv_block(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Conv = nn.Conv2d(filters[1], out_ch, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.inconv0(x)

        rese1 = self.resconv1(e1)
        e2 = self.Maxpool1(e1)
        e2 = self.MConv1(e2)
        m2 = self.mup(e2)
        m2 = m2 + rese1
        m2 =self.inconv1(m2)

        rese2 = self.resconv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.MConv2(e3)
        m3 = self.mup(e3)
        m3 = m3 + rese2
        m3 = self.inconv2(m3)

        rese3 = self.resconv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.MConv3(e4)
        m4 = self.mup(e4)
        m4 = m4 + rese3
        m4 = self.inconv3(m4)

        rese4 = self.resconv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.MConv4(e5)
        m5 = self.mup(e5)
        m5 = m5 +rese4
        m5 = self.inconv4(m5)

        rese5 = self.resconv5(e5)
        e6 = self.Maxpool5(e5)
        e6 = self.MConv5(e6)
        me6 = self.mup(e6)
        m6 = me6+rese5

        d5 = self.Up5(m6)
        d5 = torch.cat((m5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((m4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((m3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((m2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        # return tuple(out)

        return out

class TAU_Net_incep(SegmentationNetwork):
    """
    UNet with the inception module   
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(TAU_Net_incep, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16, 32, 64, 128, 256, 512]

        # self.up =  nn.Upsample(scale_factor=2)
        # self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resconv1 = Resconv_block(filters[0], filters[1])
        self.resconv2 = Resconv_block(filters[1], filters[2])
        self.resconv3 = Resconv_block(filters[2], filters[3])
        self.resconv4 = Resconv_block(filters[3], filters[4])
        self.resconv5 = Resconv_block(filters[4], filters[5])

        self.inconv0 = conv_block(in_ch, filters[0])
        self.inconv1 = Resconv_block(filters[1], filters[1])
        self.inconv2 = Resconv_block(filters[2], filters[2])
        self.inconv3 = Resconv_block(filters[3], filters[3])
        self.inconv4 = Resconv_block(filters[4], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv1 = inception_block(filters[0], filters[1])
        self.MConv2 = inception_block(filters[1], filters[2])
        self.MConv3 = inception_block(filters[2], filters[3])
        self.MConv4 = inception_block(filters[3], filters[4])
        self.MConv5 = inception_block(filters[4], filters[5])

        self.mup = MultiscaleFuse()

        self.Up5 = up_conv(filters[5], filters[4])
        self.Up_conv5 = conv_block(filters[5], filters[4])

        self.Up4 = up_conv(filters[4], filters[3])
        self.Up_conv4 = conv_block(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
        self.Up_conv3 = conv_block(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Conv = nn.Conv2d(filters[1], out_ch, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.inconv0(x)

        rese1 = self.resconv1(e1)
        e2 = self.Maxpool1(e1)
        e2 = self.MConv1(e2)
        m2 = self.mup(e2)
        m2 = m2 + rese1
        m2 =self.inconv1(m2)

        rese2 = self.resconv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.MConv2(e3)
        m3 = self.mup(e3)
        m3 = m3 + rese2
        m3 = self.inconv2(m3)

        rese3 = self.resconv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.MConv3(e4)
        m4 = self.mup(e4)
        m4 = m4 + rese3
        m4 = self.inconv3(m4)

        rese4 = self.resconv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.MConv4(e5)
        m5 = self.mup(e5)
        m5 = m5 +rese4
        m5 = self.inconv4(m5)

        rese5 = self.resconv5(e5)
        e6 = self.Maxpool5(e5)
        e6 = self.MConv5(e6)
        me6 = self.mup(e6)
        m6 = me6+rese5

        d5 = self.Up5(m6)
        d5 = torch.cat((m5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((m4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((m3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((m2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        # return tuple(out)

        return out

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, scfactor=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scfactor, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)

def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

net = model_unet(BAU_Net_v0,1,3)
net.apply(init)

# # 输出数据维度检�?# net = net.cuda()
# data = torch.randn((1, 1, 512, 512)).cuda()#B*C*W*H
# res = net(data)
# for item in res:
#     print(item.size())

# 计算网络参数
print('net total parameters:', sum(param.numel() for param in net.parameters()))
print('print net parameters finish!')
