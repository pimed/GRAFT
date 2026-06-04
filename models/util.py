from __future__ import print_function
from numpy import append
from numpy.core.fromnumeric import transpose

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import operator
from functools import reduce


def cal_param_size(model):
    return sum([i.numel() for i in model.parameters()])


count_ops = 0
def measure_layer(layer, x, multi_add=1):
    delta_ops = 0
    type_name = str(layer)[:str(layer).find('(')].strip()

    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) //
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) //
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w // layer.groups * multi_add

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = 0
        delta_ops = weight_ops + bias_ops

    global count_ops
    count_ops += delta_ops
    return


def is_leaf(module):
    return sum(1 for x in module.children()) == 0


def should_measure(module):
    if str(module).startswith('Sequential'):
        return False
    if is_leaf(module):
        return True
    return False


def cal_multi_adds(model, shape=(2,3,32,32)):
    global count_ops
    count_ops = 0
    data = torch.zeros(shape)

    def new_forward(m):
        def lambda_forward(x):
            measure_layer(m, x)
            return m.old_forward(x)
        return lambda_forward

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops


class ConvReg(nn.Module):
    """Convolutional regression for FitNet (feature map layer)"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        self.s_H = s_H
        self.t_H = t_H
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.s_H * 4 == self.t_H:
            x = F.interpolate(x, size=(self.t_H, self.t_H), mode='bilinear')
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)
        

class Regress(nn.Module):
    """Simple Linear Regression for FitNet (feature vector layer)"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x
        
class CalWeight(nn.Module):
    def __init__(self, feat_s, feat_t_list, opt):
        super(CalWeight, self).__init__()

        self.opt = opt
        # student和teacher都用最后一层
        s_channel = feat_s.shape[1]
        print('s_channel', s_channel)
        for i in range(len(feat_t_list)):
            t_channel = feat_t_list[i].shape[1]
            print('t_channel', t_channel)
            setattr(self, 'embed'+str(i), Embed(s_channel, t_channel, self.opt.factor, self.opt.convs))


    def forward(self, feat_s, feat_t_list, model_t_list=None):
        tmp_model = [model_t.distill_seq() for model_t in model_t_list]
        trans_feat_s_list = []
        output_feat_t_list = []
        s_H = feat_s.shape[2]
        for i, mid_feat_t in enumerate(feat_t_list):
            t_H = mid_feat_t.shape[2]
            if s_H >= t_H:
                feat_s = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            else:
                feat_s = F.interpolate(feat_s, size=(t_H, t_H), mode='bilinear')
            trans_feat_s = getattr(self, 'embed'+str(i))(feat_s)
            trans_feat_s_list.append(trans_feat_s)
            output_feat_t = tmp_model[i][-1](trans_feat_s)
            output_feat_t_list.append(output_feat_t)
        return trans_feat_s_list, output_feat_t_list

class TransFeat(nn.Module):
    def __init__(self, feat_s_size, feat_t_list_size, use_orthogonal=False):
        super(TransFeat, self).__init__()
        # student和teacher都用最后一层
        s_channel = feat_s_size[1]
        self.feat_t_list_size = feat_t_list_size
        self.use_orthogonal = use_orthogonal
        
        # Auto-detect 2D vs 3D based on feature dimensions
        # 2D: (B, C, H, W) -> len=4, 3D: (B, C, D, H, W) -> len=5
        self.is_3d = len(feat_s_size) == 5
        
        for i in range(len(feat_t_list_size)):
            t_channel = feat_t_list_size[i][1]
            if self.is_3d:
                if use_orthogonal:
                    setattr(self, 'embed'+str(i), Embed3D_Orthogonal(s_channel, t_channel))
                else:
                    setattr(self, 'embed'+str(i), Embed3D(s_channel, t_channel))
            else:
                setattr(self, 'embed'+str(i), Embed(s_channel, t_channel))

    def forward(self, feat_s):
        trans_feat_s_list = []
        
        if self.is_3d:
            # 3D feature handling: [B, C, D, H, W]
            s_D, s_H, s_W = feat_s.shape[2], feat_s.shape[3], feat_s.shape[4]
            
            for i, mid_feat_t in enumerate(self.feat_t_list_size):
                t_D, t_H, t_W = mid_feat_t[2], mid_feat_t[3], mid_feat_t[4]
                
                # Resize spatial dimensions to match teacher
                if s_D >= t_D and s_H >= t_H and s_W >= t_W:
                    # Downsample using adaptive pooling (memory efficient)
                    feat_s_resized = F.adaptive_avg_pool3d(feat_s, (t_D, t_H, t_W))
                else:
                    # Upsample using trilinear interpolation
                    feat_s_resized = F.interpolate(feat_s, size=(t_D, t_H, t_W), mode='trilinear', align_corners=True)
                
                # Apply embedding and immediately delete resized features
                trans_feat_s = getattr(self, 'embed'+str(i))(feat_s_resized)
                del feat_s_resized  # Explicitly free memory
                trans_feat_s_list.append(trans_feat_s)
        else:
            # 2D feature handling: [B, C, H, W] (original code)
            s_H = feat_s.shape[2]
            for i, mid_feat_t in enumerate(self.feat_t_list_size):
                t_H = mid_feat_t[2]
                if s_H >= t_H:
                    feat_s_resized = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
                else:
                    feat_s_resized = F.interpolate(feat_s, size=(t_H, t_H), mode='bilinear')
                trans_feat_s = getattr(self, 'embed'+str(i))(feat_s_resized)
                del feat_s_resized  # Explicitly free memory
                trans_feat_s_list.append(trans_feat_s)
                
        return trans_feat_s_list
    
    
class AAEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(AAEmbed, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.regressor = nn.Sequential(
            # conv1x1(num_input_channels, self.num_mid_channel),
            # nn.BatchNorm2d(self.num_mid_channel),
            # nn.ReLU(inplace=True),
            # conv3x3(self.num_mid_channel, self.num_mid_channel),
            # nn.BatchNorm2d(self.num_mid_channel),
            # nn.ReLU(inplace=True),
            # conv1x1(self.num_mid_channel, num_target_channels),
            conv1x1(num_input_channels, num_target_channels),
            nn.BatchNorm2d(num_target_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
        
class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128, factor=2, convs=False):
        super(Embed, self).__init__()
        self.convs = convs
        if self.convs:
            self.transfer = nn.Sequential(
                nn.Conv2d(dim_in, dim_in//factor, kernel_size=1),
                nn.BatchNorm2d(dim_in//factor),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in//factor, dim_in//factor, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_in//factor),
                nn.ReLU(inplace=True), 
                nn.Conv2d(dim_in//factor, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True)              
            )
        else:
            self.transfer = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True) 
            )


    def forward(self, x):
        x = self.transfer(x)
        return x


class Embed3D(nn.Module):
    """3D Embedding module for 3D feature transformation
    
    Note: Using InstanceNorm3d instead of BatchNorm3d for better performance with small batch sizes.
    This avoids train/eval mismatch issues caused by poorly-estimated running statistics.
    """
    def __init__(self, dim_in=1024, dim_out=128, factor=2, convs=False):
        super(Embed3D, self).__init__()
        self.convs = convs
        if self.convs:
            self.transfer = nn.Sequential(
                nn.Conv3d(dim_in, dim_in//factor, kernel_size=1),
                nn.InstanceNorm3d(dim_in//factor, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(dim_in//factor, dim_in//factor, kernel_size=3, padding=1),
                nn.InstanceNorm3d(dim_in//factor, affine=True),
                nn.ReLU(inplace=True), 
                nn.Conv3d(dim_in//factor, dim_out, kernel_size=1),
                nn.InstanceNorm3d(dim_out, affine=True),
                nn.ReLU(inplace=True)              
            )
        else:
            self.transfer = nn.Sequential(
                nn.Conv3d(dim_in, dim_out, kernel_size=1),
                nn.InstanceNorm3d(dim_out, affine=True),
                nn.ReLU(inplace=True) 
            )

    def forward(self, x):
        x = self.transfer(x)
        return x


class Embed3D_Orthogonal(nn.Module):
    """3D Embedding with orthogonal projection constraint (VkD-style)
    
    Uses orthogonal parametrization on Conv3d to constrain weights W such that W^T @ W = I.
    This preserves feature norms (||Wx|| = ||x||) and provides stable gradients.
    
    No ReLU/normalization to preserve orthogonality property.
    Learnable per-channel scale allows the network to adjust feature magnitudes.
    """
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed3D_Orthogonal, self).__init__()
        # Orthogonal 1x1x1 conv (no bias to maintain pure orthogonal transform)
        self.transfer = torch.nn.utils.parametrizations.orthogonal(
            nn.Conv3d(dim_in, dim_out, kernel_size=1, bias=False)
        )
        # Learnable per-channel scale (like VkD)
        self.scale = nn.Parameter(torch.ones(dim_out))
        
    def forward(self, x):
        x = self.transfer(x)
        # Apply learnable scale per channel
        x = x * self.scale.view(1, -1, 1, 1, 1)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""
    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super().__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(nChannels*pool_size*pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)


if __name__ == '__main__':
    import torch

    g_s = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    g_t = [
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 128, 4, 4),
    ]
    s_shapes = [s.shape for s in g_s]
    t_shapes = [t.shape for t in g_t]

    net = ConnectorV2(s_shapes, t_shapes)
    out = net(g_s)
    for f in out:
        print(f.shape)
