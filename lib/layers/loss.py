import torch
import torch.nn as nn
import lib.layers.functional as LF


class RegressionLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RegressionLoss, self).__init__()
        self.eps = eps

    def forward(self, x1, x2):
        return LF.regression_loss(x1, x2, eps=self.eps)


class SecondOrderLoss(nn.Module):
    def __init__(self, lam_1, lam_2, eps=1e-6):
        super(SecondOrderLoss, self).__init__()
        self.eps = eps
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.rel = lam_1 or lam_2

    def forward(self, s_vecs, t_vecs):
        # abs
        loss = LF.regression_loss(s_vecs, t_vecs, eps=self.eps)

        if self.rel:
            t_sim = torch.bmm(t_vecs, t_vecs.permute(0, 2, 1))
            # rel_ts
            if self.lam_1:
                s_t_sim = torch.bmm(s_vecs, t_vecs.permute(0, 2, 1))
                loss += self.lam_1 * LF.mse_loss(s_t_sim, t_sim)

            # rel_ss
            if self.lam_2:
                s_sim = torch.bmm(s_vecs, s_vecs.permute(0, 2, 1))
                loss += self.lam_2 * LF.mse_loss(s_sim, t_sim)

        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x, pos, neg):
        loss = LF.triplet_loss(x, pos, neg, margin=self.margin)
        return loss


    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'