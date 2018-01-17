#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:40:02 2017
Capsule Network with pytorch
@author: zengliang
"""
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BATCH_SIZE = 128
NUM_EPOCHS = 30
NUM_ROUNTING_ITERATIONS = 3

caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6
caps1_n_dims = 8
caps2_n_caps = 10
caps2_n_dims = 16

train_dataset = dsets.MNIST(root='../data/',train=True,
                            transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='../data/',train=False,
                            transform=transforms.ToTensor(),download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)


def _squash(x, dim=-1, epsilon=1e-7):
    squared_norm = (x**2).sum(dim=dim,keepdim=True)
    safe_norm = torch.sqrt(squared_norm+epsilon)
    squash_factor = squared_norm / (1.0+squared_norm)
    unit_vector = x / safe_norm
    return squash_factor * unit_vector

def _margin_loss(out, target):
    # target is provided with shape = [batch_size, n_classes], i.e. one-hot code.
    m_pos = 0.9
    m_neg = 0.1
    lamda = 0.5
    pos = target * F.relu(m_pos-torch.norm(input=out,p=2.0,dim=-1))**2
    neg = lamda * (1-target) * F.relu(torch.norm(input=out,p=2.0,dim=-1)-m_neg)**2
    return torch.mean(torch.sum(pos+neg, dim=-1))

class _all_loss(nn.Module):
    def __init__(self):
        super(_all_loss,self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(caps2_n_caps*caps2_n_dims,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,784),
            nn.Sigmoid()
            )
 
    def forward(self, x, out, y):
        alpha = 0.0005
        loss1 = _margin_loss(out,y)
        loss2 = torch.sum((x.view(x.size(0),-1)-self.decoder(out.view(out.size(0),-1)))**2,dim=-1)
        return torch.mean(loss1+alpha*loss2) 

class primary_capsule(nn.Module):
    def __init__(self,in_capsules,in_dims,out_capsules,out_dims):
        #in_capsules:256
        #in_dims:20
        #out_capsules:32*6*6
        #out_dims:8
        super(primary_capsule,self).__init__()
        self.capsules = nn.ModuleList(
                [nn.Conv2d(in_capsules,caps1_n_maps,9,2,0) for _ in range(out_dims)])
    
    def forward(self,x):
        output = [torch.unsqeeze(capsules(x),dim=-1) for capsules in self.capsules]
        output = torch.cat(output,-1)
        output = output.view(output.size(0),-1,output.size(-1))
        output = _squash(output,dim=-1)
        return output
    
class digit_capsule(nn.Module):
    def __init__(self,in_capsules,in_dims,out_capsules,out_dims):
        #in_capsules:32*6*6=1152
        #in_dims:8
        #out_capsules:10
        #out_dims:16
        super(digit_capsule,self).__init__()
        self.out_dims = out_dims
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.W = nn.Parameter(
                torch.randn(in_capsules,out_capsules,out_dims,in_dims))
        
    def forward(self,x):
        b = Variable(torch.zeros(x.size(0),self.in_capsules,self.out_capsules,1,1)).cuda()
        x = torch.unsqueeze(x.unsqueeze_(dim=-2).repeat(1,1,self.out_capsules,1),dim=-1)
        w = torch.unsqueeze(self.W,dim=0).repeat(x.size(0),1,1,1,1)
	u = torch.matmul(w, x)
        for i in range(NUM_ROUNTING_ITERATIONS):
            c = F.softmax(b,dim=2)
            c_expand = c.repeat(1,1,1,self.out_dims,1)
            s = torch.sum(c_expand * u,dim=1,keepdim=True)
            v = _squash(s,dim=2)
            if i != NUM_ROUNTING_ITERATIONS-1:
                v_expand = v.repeat(1,self.in_capsules,1,1,1)
                b = b + torch.matmul(u.transpose(-1,-2),v_expand)
        return v.squeeze_()
                
    
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,256,9,1)
        self.primary_capsule = primary_capsule(256,20,caps1_n_caps,caps1_n_dims)
        self.digit_capsule = digit_capsule(caps1_n_caps,caps1_n_dims,caps2_n_caps,caps2_n_dims)
        
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.primary_capsule(out)
        out = self.digit_capsule(out)
        return out
        

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


if __name__ == "__main__":
    net = Net()
    criterion = _all_loss()
    print net
    if use_cuda:
        net.cuda()
        criterion.cuda()
 
    optim = optim.Adam(net.parameters())
    for epoch in range(NUM_EPOCHS):
        for i, (images,labels) in enumerate(train_loader):
            images = Variable(images).type(Tensor)
            labels_one_hot = torch.zeros(labels.size(0),caps2_n_caps)
            labels_one_hot.scatter_(1,labels.squeeze_(-1),1.0)
            labels_one_hot = Variable(labels_one_hot).type(Tensor)
            optim.zero_grad()
            output = net(images)
            loss = criterion(images,output,labels_one_hot)
            loss.backward()
            optim.step()
            
            if (i+1) % 100 == 0:
                print 'Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'%(epoch+1,
                             NUM_EPOCHS, i, len(train_dataset)//BATCH_SIZE,loss.data[0])
    
    total, correct = 0, 0           
    for images,labels in test_loader:
        images = Variable(images).type(Tensor)
        output = net(images)
        predicted = torch.max(torch.norm(output,2,-1),dim=-1)[1]
        total += labels.size(0)
        correct += (predicted.data==labels).sum()
    
    print 'Test Accuracy %.4f'%(correct/total)
