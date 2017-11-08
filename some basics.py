#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:35:52 2017
some basics in pytorch
@author: zengliang
"""
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
#---------------------------
x = torch.rand(5,3)
print x
print x.size() #torch.Size is in fact a tuple.

print x.t_() #Any operation that mutates a tensor in-place is post-fixed with an _
#torch.add(x, 1, out=x)

print x[:,1] #numpy-like slicing

#if torch.cuda.is_available(): #Tensors can be moved onto GPU using the .cuda function.
#    x = x.cuda()
#    y = y.cuda()
#    x + y

#---------------------------------
#autograd: Variable and Function
#Each variable has a .grad_fn attribute that references a Function that has created the Variable
x = Variable(torch.ones(2,2), requires_grad=True)
y = x+1
print y.grad_fn

x = torch.rand(3)
x = Variable(x, requires_grad=True)
y = x*2
gradients = torch.FloatTensor([0.1,1,0.001])
y.backward(gradients)
print x.grad #mutiply different scale of gradients

x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

y = w*x+b
y.backward()

print x.grad, w.grad, b.grad
#-------------------------------
x = Variable(torch.randn(5,3))
y = Variable(torch.randn(5,2))

linear = nn.Linear(3,2,bias=True)
print 'w: ',linear.weight
print 'b: ',linear.bias

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(),lr=0.01)

pred = linear(x)
loss = criterion(pred,y)
print 'loss: ',loss.data[0]

loss.backward()
print 'dl/dw: ',linear.weight.grad
print 'dl/db: ',linear.bias.grad

optimizer.step()
#linear.weight.data.sub_(0.01*linear.weight.grad.data)
#linear.bias.data.sub_(0.01*linear.bias.grad.data)

pred = linear(x)
loss = criterion(pred,y)
print 'loss after optimization',loss

#-------------------------------------
a = np.array([[1,2],[3,4]])
b = torch.from_numpy(a) #numpy to torch tensor
c = b.numpy() #torch tensor to numpy
print b
print c
#---------------------------------
dtype = torch.FloatTensor
w1 = torch.randn(D_in, H).type(dtype)
