import torch
import torch.nn.functional as F

# --------------------------------------
# pooling
# --------------------------------------

def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))


def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

# --------------------------------------
# normalization
# --------------------------------------

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

def powerlaw(x, eps=1e-6):
    x = x + eps
    return x.abs().sqrt().mul(x.sign())

# --------------------------------------
# loss
# --------------------------------------
def activations(x, eps=1e-6):
    # # PMATA
    # x = x.pow(2).sum(dim=1)
    # x = x / (torch.norm(x, p=2) + eps)
    # return x
    return x.permute((2, 3, 0, 1)) / (torch.norm(x, p=2, dim=[2, 3]) + eps)
    # return (x.permute((2, 3, 1, 0)) / (torch.norm(x, p=2, dim=[2, 3, 1]) + eps)).permute(0, 1, 3, 2)

def identity(x):
    return x

def mse_loss(x1, x2, eps=1e-6):
    dif = x1 - x2
    D = torch.pow(dif + eps, 2)
    return D.sum(0).fill_diagonal_(0).sum() / (D.shape[1] * (D.shape[2]-1))


def regression_loss(x1, x2, eps=1e-6):
    dif = 1 - (x1 * x2).sum(-1)
    D = torch.pow(dif + eps, 2)
    return torch.sum(D.mean(-1))


def triplet_loss(x, p, n, margin=0.1):
    # D x nnum x batch_size
    pnum = p.shape[1]
    nnum = n.shape[1]

    dist_pos = torch.sum(torch.pow(x - p, 2), dim=-1)
    dist_neg = torch.sum(torch.pow(x - n, 2), dim=-1)

    loss = dist_pos.repeat_interleave(nnum, 1) - dist_neg.repeat(1, pnum) + margin
    loss = torch.clamp(loss, min=0)
    return loss.mean(dim=-1).sum()