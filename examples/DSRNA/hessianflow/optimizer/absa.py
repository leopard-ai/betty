#*
# @file ABSA training driver based on arxiv:1810.01021 
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of HessianFlow library.
#
# HessianFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HessianFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HessianFlow.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from .progressbar import progress_bar
from .optm_utils import fgsm, exp_lr_scheduler, test

import hessianflow
from hessianflow.utils import get_params_grad, group_add
from hessianflow.eigen import get_eigen
from copy import deepcopy


def get_lr(opt):
    """
    get the learning rate 
    """
    for param_group in opt.param_groups:
        return param_group['lr']

def copy_update(opt, grad):
    """
    used for optimizer update
    """
    for group in opt.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for i,p in enumerate(group['params']):
            d_p = grad[i]
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = opt.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            p.data.add_(-group['lr'], d_p)

def absa(model, train_loader, hessian_loader, test_loader, criterion, optimizer, epochs, lr_decay_epoch, lr_decay_ratio, batch_size = 128,
        max_large_ratio = 1, adv_ratio = 0., eps = 0., duration = True, cuda = True, print_flag = False):
    """
    adaptive batch size with adversarial training
    """
    
    # initilization 
    large_grad = []
    inner_loop = 0
    large_ratio = 1
    max_eig = None
    decay_ratio = 2
    flag = True
    if max_large_ratio == 1:
        flag = False
    
    data_eigen = None
    target_eigen = None
    flag_data = True
    if duration == True: 
        duration = 10
    else:
        duration = None

    cur_duration = 0
    num_updates = 0
    initial_lr = get_lr(optimizer)
    
    
    for epoch in range(1, epochs + 1):
        print('\nCurrent Epoch: %d' % epoch)
        print('\nTraining')
        train_loss = 0.
        total_num = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if data.size()[0] < batch_size:
                continue
            # gather input and target for large batch training        
            inner_loop += 1
            
            # save the data for eigen-computation
            if flag_data:
                data_eigen = data
                target_eigen = target
                #flag_data = False
            # get small model update
            # use adversarial training
            if adv_ratio > 1. / batch_size:
                adv_r = max(int(batch_size * adv_ratio), 1)
                model.eval() # set flag so that Batch Norm statistics would not be polluted with fgsm
                adv_data = fgsm(model, data[:adv_r], target[:adv_r], eps, cuda)
                model.train() # set flag to train for Batch Norm
                adv_data = torch.cat([adv_data, data[adv_r:]])
            else:
                model.train()
                adv_data = data

            optimizer.zero_grad()
            if cuda:
                adv_data, target = adv_data.cuda(), target.cuda()

            output = model(adv_data)
            loss = criterion(output, target) / large_ratio
            total_num +=target.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            
            train_loss += loss.item() * target.size(0) * float(large_ratio)
            loss.backward()
            _, small_grad= get_params_grad(model)
            if not large_grad:
                large_grad = deepcopy(small_grad) #[small_grad_ + 0. for small_grad_ in small_grad]
            else:
                large_grad = group_add(large_grad, small_grad)


            if inner_loop % large_ratio  == 0:
                num_updates += 1
                copy_update(optimizer, large_grad) # todo: see if we can use deep copy to set optimizer.grad = large_grad
                large_grad = []
                inner_loop = 0
                optimizer.zero_grad()
                
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
             % (train_loss / total_num,
                100. * correct/total_num, correct, total_num))
                
        ## compute eigenvalues and update large_ratio, adv_ratio etc
        if flag:
            for data, target in hessian_loader:
                data_eigen = data
                target_eigen = target
                break
            eig, _ = get_eigen(model, data_eigen, target_eigen, criterion, cuda = True, maxIter = 10, tol = 1e-2)
            cur_duration += 1

            if max_eig == None:
                max_eig = eig
            else:
                if eig <= max_eig/decay_ratio:
                    # ensure the learning rate is not too crazy, espeacially for model without batch normalization
                    max_eig = eig
                    prev_ratio = large_ratio
                    large_ratio = int(large_ratio*decay_ratio)
                    adv_ratio /= decay_ratio
                    if large_ratio  >= max_large_ratio:
                        large_ratio = max_large_ratio
                        adv_ratio = 0.
                        flag = False
                    cur_duration = 0
                    optimizer = exp_lr_scheduler(optimizer, decay_ratio = large_ratio/prev_ratio)
        if duration != None: # if it is around a quadratic bowl, increase batch size
            # ensure the learning rate is not too crazy, espeacially for model without batch normalization
            if cur_duration - duration > -0.5:
                prev_ratio = large_ratio
                large_ratio = int(large_ratio*decay_ratio)
                adv_ratio /= decay_ratio
                if large_ratio  >= max_large_ratio:
                    large_ratio = max_large_ratio
                    adv_ratio = 0.
                    flag = False
                cur_duration = 0
                optimizer = exp_lr_scheduler(optimizer, decay_ratio = large_ratio/prev_ratio)


        if epoch in lr_decay_epoch:
            optimizer = exp_lr_scheduler(optimizer, decay_ratio = lr_decay_ratio)
            
        if epoch >= epochs // 2:
            adv_ratio = 0.
        
        if print_flag:
            #print('\n Batch size %d' % (batch_size*large_ratio))
            print('\n Eig %f Max Eig %f Batch size %d' % (eig, max_eig, batch_size * large_ratio))
            
        test(model, test_loader)
        
    return model, num_updates
