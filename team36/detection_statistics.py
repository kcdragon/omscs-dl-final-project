import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

def calculateStats(X):
    return PCA(X)

# def calculateMMD(x1, x2, device, kernel="guassian"):
#     """
#     Args:
#     x1: first sample, distribution P
#     x2: second sample, distribution Q
#     kernel: kernel type - default is guassian
#     """
#     xt1 = x1 @ x1.T
#     xt2 = x2 @ x2.T
#     combined = x1 @ x2.T
#
#     rx1 = (xt1.diag().unsqueeze(0).expand_as(xt1))
#     rx2 = (xt2.diag().unsqueeze(0).expand_as(xt2))
#
#     dx1 = rx1.T + rx1 - 2. * xt1
#     dx2 = rx2.T + rx2 - 2. * xt2
#     dx = rx1.t() + rx2 - 2. * combined
#
#     X1 = torch.zeros(xt1.shape).to(device)
#     X2 = torch.zeros(xt1.shape).to(device)
#     COMB = torch.zeros(xt1.shape).to(device)
#
#     if kernel == "guassian":
#         bandwidth_range = [10, 15, 20, 50]
#         for a in bandwidth_range:
#             X1 += torch.exp(-0.5 * dx1 / a)
#             X2 += torch.exp(-0.5 * dx2 / a)
#             COMB += torch.exp(-0.5 * dx / a)
#     else:
#         print("Only guassian kernels supported at this time.")
#
#     return torch.mean(X1 + X2 - 2. * COMB)

def calculatePCA(X):
    pca = PCA()
    pca.fit(X)

    return pca.score()