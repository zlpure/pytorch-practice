#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:11:15 2017

@author: zengliang
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        #flatten perform, view equares reshape
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
net = Net()
print net

params = list(net.parameters())
print len(params)
print params[0].size()

input = Variable(torch.randn(1,1,32,32))
out = net(input)
print out
#net.zero_grad() #Sets gradients of all model parameters to zero.
#out.backward(torch.randn(1,10))
#note:torch.nn only supports mini-batches
#If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.

target = Variable(torch.arange(1,11))
criterion = nn.BCELoss()
optim = optim.Adam(net.parameters())
loss = criterion(out,target)
print loss.data[0]
print loss.grad_fn
print loss.grad_fn.next_functions[0][0]

net.zero_grad()
print net.conv1.bias.grad

loss.backward()
print net.conv1.bias.grad

#lr = 0.01
#for f in net.parameters():
#    f.data.sub_(f.grad.data*lr)
optimizer = optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output,target)
loss.backward()
optimizer.step() #auto perform
