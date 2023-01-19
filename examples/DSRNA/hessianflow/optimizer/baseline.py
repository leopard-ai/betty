from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from .progressbar import progress_bar
from .optm_utils import exp_lr_scheduler, test

# import hessianflow

def baseline(model, train_loader, test_loader, criterion, optimizer, epochs, lr_decay_epoch,
        lr_decay_ratio, batch_size = 128, max_large_ratio = 1, cuda = True):
    """
    baseline method training, i,e, vanilla training schedule
    """
    
    inner_loop = 0
    num_updates = 0
    large_ratio = max_large_ratio 
    # assert that shuffle is set for train_loader
    # assert and explain large ratio 
    # assert that the train_loader is always set with a small batch size if not print error/warning telling
    # the user to instead use large_ratio
    for epoch in range(1, epochs + 1):
        print('\nCurrent Epoch: ', epoch)
        print('\nTraining')
        train_loss = 0.
        total_num = 0.
        correct = 0.

        for batch_idx, (data, target) in enumerate(train_loader):
            if target.size(0) < 128:
                continue
            model.train()
            # gather input and target for large batch training        
            inner_loop += 1
            # get small model update
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)/float(large_ratio)
            loss.backward()
            train_loss += loss.item()*target.size(0)*float(large_ratio)
            total_num += target.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

            if inner_loop % large_ratio  == 0:
                num_updates += 1
                optimizer.step()
                inner_loop = 0
                optimizer.zero_grad()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / total_num,
                            100. * correct / total_num, correct, total_num))

        if epoch in lr_decay_epoch:
            exp_lr_scheduler(optimizer, decay_ratio=lr_decay_ratio)
        
        test(model, test_loader)     
    return model, num_updates
