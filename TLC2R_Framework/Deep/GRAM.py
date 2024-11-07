# Reference: https://github.com/ismailnejjar/DARE-GRAM

import torch
import torch.nn as nn


def DARE_GRAM_LOSS(H1, H2, device,treshold = 0.9,tradeoff_angle = 0.05,tradeoff_scale = 0.001):
    b, p = H1.shape

    A = torch.cat((torch.ones(b, 1).to(device), H1), 1)
    B = torch.cat((torch.ones(b, 1).to(device), H2), 1)

    cov_A = (A.t() @ A)
    cov_B = (B.t() @ B)

    _, L_A, _ = torch.linalg.svd(cov_A)
    _, L_B, _ = torch.linalg.svd(cov_B)

    eigen_A = torch.cumsum(L_A.detach(), dim=0) / L_A.sum()
    eigen_B = torch.cumsum(L_B.detach(), dim=0) / L_B.sum()

    if (eigen_A[1] > treshold):
        T = eigen_A[1].detach()
    else:
        T = treshold

    index_A = torch.argwhere(eigen_A.detach() <= T)[-1]

    if (eigen_B[1] > treshold):
        T = eigen_B[1].detach()
    else:
        T = treshold

    index_B = torch.argwhere(eigen_B.detach() <= T)[-1]

    k = max(index_A, index_B)[0]

    A = torch.linalg.pinv(cov_A, rtol=(L_A[k] / L_A[0]).detach())
    B = torch.linalg.pinv(cov_B, rtol=(L_B[k] / L_B[0]).detach())

    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos = torch.dist(torch.ones((p + 1)).to(device), (cos_sim(A, B)), p=1) / (p + 1)
    return tradeoff_angle * (cos) + tradeoff_scale * torch.dist((L_A[:k]), (L_B[:k])) / k