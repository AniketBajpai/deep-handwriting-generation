import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def positional_loss(parameters, targets, weights):
    # parameters of bivariate distribution
    mu1, mu2, sigma1, sigma2, rho = parameters
    x1, x2 = targets
    z = ((x1 - mu1) / sigma1)**2 + ((x2 - mu2) / sigma2)**2 - \
        2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
    rho_dash = (1 - rho**2)
    bivariate_gaussian_exp = z / (2 * rho_dash)
    n = torch.exp(-bivariate_gaussian_exp) / \
        (2 * np.pi * sigma1 * sigma2 * torch.sqrt(rho_dash))

    # loss function for element at time t
    eps = np.finfo(float).eps
    elementwise_loss = - \
        torch.log(torch.sum(weights * n, dim=1) + eps)   # (seq_length-1)
    total_loss = torch.sum(elementwise_loss, dim=0)
    return total_loss

def eos_loss(eos_prob, eos_target):
    eps = np.finfo(float).eps
    elementwise_loss = - (torch.log(eos_prob + eps) * eos_target +
                            torch.log(1 - eos_prob + eps) * (1 - eos_target))
    total_loss = torch.sum(elementwise_loss, dim=0)
    return total_loss
