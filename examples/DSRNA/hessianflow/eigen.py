import torch
import math
from torch.autograd import Variable
import numpy as np

from .utils import *


def get_eigen(model, inputs, targets, criterion, cuda = True, maxIter = 50, tol = 1e-3):
    """
    compute the top eigenvalues of model parameters and 
    the corresponding eigenvectors.
    """
    if cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    # change the model to evaluation mode, otherwise the batch Normalization Layer will change.
    # If you call this functino during training, remember to change the mode back to training mode.
    model.eval()

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward(create_graph = True)

    params, gradsH = get_params_grad(model)
    v = [torch.randn(p.size()).to(device) for p in params]
    v = normalization(v)

    eigenvalue = None

    for i in range(maxIter):
        model.zero_grad()
        Hv = hessian_vector_product(gradsH, params, v)
        eigenvalue_tmp = group_product(Hv, v).cpu().item()
        v = normalization(Hv)
        if eigenvalue == None:
            eigenvalue = eigenvalue_tmp
        else:
            if abs(eigenvalue-eigenvalue_tmp)/abs(eigenvalue) < tol:
                return eigenvalue_tmp, v
            else:
                eigenvalue = eigenvalue_tmp
    return eigenvalue, v

def get_eigen_full_dataset(model, dataloader, criterion, cuda = True, maxIter = 50, tol = 1e-3):
    """
    compute the top eigenvalues of model parameters and 
    the corresponding eigenvectors with a full dataset. 
    Notice, this is very expensive.
    """
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    # change the model to evaluation mode, otherwise the batch Normalization Layer will change.
    # If you call this functino during training, remember to change the mode back to training mode.
    model.eval()

    
    params,_ = get_params_grad(model)
    v = [torch.randn(p.size()).to(device) for p in params]
    v = normalization(v)

    batch_size = None
    eigenvalue = None

    for i in range(maxIter):
        THv = [torch.zeros(p.size()).to(device) for p in params]
        counter = 0
        for inputs, targets in dataloader:
            
            if batch_size == None:
                batch_size = targets.size(0)
               
            if targets.size(0) < batch_size:
                continue
            
            model.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)

            params, gradsH = get_params_grad(model)
            Hv = torch.autograd.grad(gradsH, params, grad_outputs = v, only_inputs = True, retain_graph = False)

            THv = [THv1 + Hv1 + 0. for THv1, Hv1 in zip(THv, Hv)]
            counter += 1

        eigenvalue_tmp =group_product(THv,v).cpu().item() / float(counter)
        v = normalization(THv)
        
        if eigenvalue == None:
            eigenvalue = eigenvalue_tmp
        else:
            if abs(eigenvalue-eigenvalue_tmp)/abs(eigenvalue) < tol:
                return eigenvalue_tmp, v
            else:
                eigenvalue = eigenvalue_tmp

    return eigenvalue, v
