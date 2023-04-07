import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19, 
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase_part1(nn.Module):
    def __init__(self, res_name):
        super(ResBase_part1, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class ResBase_part2(nn.Module):
    def __init__(self, res_name):
        super(ResBase_part2, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class DS_module(nn.Module):
    def __init__(self, res_name):
        super(DS_module, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.di_layer = model_resnet.layer4[0]
        self.conv_a = nn.Conv2d(2048, 256, 1, stride=1, bias=False)
        self.bn_a = nn.BatchNorm2d(256, affine=True)
        self.conv_b = nn.Conv2d(256, 128, 1, stride=1, bias=False)
        self.bn_b = nn.BatchNorm2d(128, affine=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.in_features = 128

    def forward(self, x):
        x = self.di_layer(x)
        x = self.conv_a(x)
        x1 = self.bn_a(x)
        x = self.conv_b(x1)
        x2 = self.bn_b(x)
        x = self.avgpool(x2)
        x = x.view(x.size(0), -1)
        return x, x1, x2


class DS_module_two(nn.Module):
    def __init__(self, res_name):
        super(DS_module_two, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.di_layer = model_resnet.layer4
        self.conv_a = nn.Conv2d(2048, 512, 1, stride=1, bias=False)
        self.bn_a = nn.BatchNorm2d(512, affine=True)
        self.conv_b = nn.Conv2d(512, 512, 3, stride=1, padding=(1, 1), bias=False)
        self.bn_b = nn.BatchNorm2d(512, affine=True)
        self.conv_c = nn.Conv2d(512, 128, 1, stride=1, bias=False)
        self.bn_c = nn.BatchNorm2d(128, affine=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.in_features = 128

    def forward(self, x):
        x = self.di_layer(x)
        x1 = self.conv_a(x)
        x = self.bn_a(x1)
        print("x1 size = "+str(torch.mean(x1, dim=(0,2,3)).size()))
        x2 = self.conv_b(x)
        x = self.bn_b(x2)
        print("x2 size = " + str(x2.size()))
        x3 = self.conv_c(x)
        x = self.bn_c(x3)
        print("x3 size = " + str(x3.size()))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x, x1, x2, x3
    # def forward(self, x):
    #     x = self.di_layer(x)
    #     x = self.conv_a(x)
    #     x1 = self.bn_a(x)
    #     # print("x1 size = "+str(torch.mean(x1, dim=(0,2,3)).size()))
    #     x = self.conv_b(x1)
    #     x2 = self.bn_b(x)
    #     # print("x2 size = " + str(x2.size()))
    #     x = self.conv_c(x2)
    #     x3 = self.bn_c(x)
    #     # print("x3 size = " + str(x3.size()))
    #     x = self.avgpool(x3)
    #     x = x.view(x.size(0), -1)
    #     return x, x1, x2, x3


class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y

""""
import import import import import import import import import import
import import import import import import import import import import
import import import import import import import import import import
import import import import import import import import import import
lmport lmport lmport lmport 111111 lmport lmport lmport lmport lmport
lmport lmport lmport lm 111 111111 111 po lmport lmport lmport lmport
lmport lmport lmport lm 111 111111 111 po port lmport lmport lmportad
lmport lmport lmport lmport 111111 lmport lmport lmport lmport lmport
'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor'
'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor'
'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor'
'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor' 'mpor'
"""