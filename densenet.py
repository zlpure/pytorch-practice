#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:26:29 2017
ref: https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
@author: zengliang
"""
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Bottleneck(nn.Module):
    def __init__(self,nChannels,growthRate):
        super(Bottleneck,self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels,interChannels,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm1d(growthRate)
        self.conv2 = nn.Conv2d(interChannels,growthRate,kernel_size=1,padding=1,bias=False)
    
    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(x)))
        out = torch.cat((x,out),1)
        return out
    
class SingerLayer(nn.Module):
    def __init__(self,nChannels,growthRate):
        super(SingerLayer,self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels,growthRate,kernel_size=3,padding=1,bias=False)
        
    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x,out),1)
        return out
        
class Transtion(nn.Module):
    def __init__(self,nChannels,nOutChannels):
        super(Transtion,self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels,nOutChannels,kernel_size=1,bias=False)
    
    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out,2)
        return out
    
class DenseNet(nn.Module):
    def __init__(self,growthRate,depth,reduction,nClasses,bottleneck):
        super(DenseNet,self).__init__()
        nDenseBlocks = (depth-4) //3
        if bottleneck:
            nDenseBlocks = depth //2
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3,nChannels,kernel_size=3,padding=1,bias=False)
        self.dense1 = self._make_dense(nChannels,growthRate,nDenseBlocks,bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(np.floor(nChannels*reduction))
        self.trans1 = Transtion(nChannels,nOutChannels)
        
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels,growthRate,nDenseBlocks,bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(np.floor(nChannels*reduction))
        self.trans1 = Transtion(nChannels,nOutChannels)
        
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels,growthRate,nDenseBlocks,bottleneck)
        nChannels += nDenseBlocks*growthRate
        
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels,nClasses)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight.data)
        
        def _make_dense(self,nChannels, growthRate, nDenseBlocks, bottleneck):
            layers = []
            for i in range(int(nDenseBlocks)):
                if bottleneck:
                    layers.append(Bottleneck(nChannels,growthRate))
                else:
                    layers.append(SingerLayer(nChannels,growthRate))
                nChannels += growthRate
            return nn.Sequential(*layers)
        
        def forward(self,x):
            out = self.conv1(x)
            out = self.trans1(self.dense1(out))
            out = self.trans2(self.dense2(out))
            out = self.dense3(out)
            out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
            out = F.log_softmax(self.fc(out))
            return out
            
            
            