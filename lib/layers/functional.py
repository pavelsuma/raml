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


def triplet_loss(x, label, margin=0.1):
    dim = x.size(0)  # D
    nq = torch.sum(label.data == -1).item()  # number of tuples
    S = x.size(1) // nq  # number of images per tuple including query: 1+1+n
    P = sum(label[:S] == 1)  # number of positives per tuple
    N = S-P-1 # number of negatives per tuple
    R = N*P*nq  # number of final embeddings for all tuples
    
    xa = x[:, label.data == -1].permute(1, 0).repeat(1, N*P).view(R, dim).permute(1, 0)
    xp = x[:, label.data == 1].permute(1, 0).repeat(1, N).view(R, dim).permute(1, 0)
    xn = x[:, label.data == 0].permute(1, 0).reshape(nq, -1).repeat(1, P).view(R, dim).permute(1, 0)

    dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=0)
    dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=0)

    loss = dist_pos - dist_neg + margin
    loss = torch.clamp(loss, min=0)
    loss = torch.sum(torch.mean(loss.view(N, P*nq), dim=0))
    return loss
