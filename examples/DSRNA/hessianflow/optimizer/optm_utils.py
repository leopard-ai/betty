
#*
# @file optm_utils.py different utility functions
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


def fgsm(model, data, target, eps, cuda = True):
    """Generate an adversarial pertubation using the fast gradient sign method.

    Args:
        data: input image to perturb
    """
    model.eval()
    if cuda:
        data, target = data.cuda(), target.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward(create_graph = False)
    pertubation = eps * torch.sign(data.grad.data)
    x_fgsm = data.data + pertubation
    X_adv = torch.clamp(x_fgsm, torch.min(data.data), torch.max(data.data))

    return X_adv.cpu()

def exp_lr_scheduler(optimizer, decay_ratio = 0.1):
    """
    Decay learning rate by a factor of lr_decay 
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_ratio
    return optimizer

    
def test(model, test_loader):
    """
    Evaluation the performance of model on test_loader
    """
    print('\nTesting')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Acc: %.3f%% (%d/%d)'
                         % (100. * correct/total, correct, total))

    return correct * 100 / total
