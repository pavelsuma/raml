import numpy as np
import torch
import torch.utils.data as data
from lib.datasets.datahelpers import default_loader


def mine_dist(vecs, qvecs, cls, qcls, nnum):
    dim = qvecs.shape[0]
    prod = torch.mm(vecs.t(), qvecs)
    distances = (2 - 2 * prod).clamp(min=0).sqrt().clamp(min=0.5)
    sel_d = distances.shape[0]

    nidxs = []
    for q in range(qvecs.shape[1]):
        q_d_inv = inverse_sphere_distances(dim, distances[:, q], cls, qcls[q])
        nidxs.append(np.random.choice(sel_d, size=nnum, p=q_d_inv, replace=False))

    return torch.tensor(nidxs)

# Function taken from https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch by Karsten Roth
def inverse_sphere_distances(dim, anchor_to_all_dists, labels, anchor_label):
    dists = anchor_to_all_dists

    # negated log-distribution of distances of unit sphere in dimension <dim>
    log_q_d_inv = ((2.0 - float(dim)) * torch.log(dists) - (float(dim - 3) / 2) * torch.log(
        1.0 - 0.25 * (dists.pow(2))))
    log_q_d_inv[np.where(labels == anchor_label)[0]] = 0

    q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv))
    q_d_inv[np.where(labels == anchor_label)[0]] = 0

    q_d_inv += 1e-6
    q_d_inv = q_d_inv / q_d_inv.sum()
    return q_d_inv.detach().cpu().numpy()

def mine_hardest(scores, cls, qcls, num, negative=True):
    ranks = torch.argsort(scores, descending=True, dim=0)[1:]
    mask = (cls[ranks] - qcls == 0).to(torch.int)
    ids = torch.gather(ranks, 0, torch.topk(mask, largest=negative ^ True, k=num, dim=0)[1])
    return ids.t()


class RetrievalDataset(data.Dataset):
    """
    Args:
        name (string): dataset name
        imsize (int, Default: None): Defines the maximum size of longer image side
        mean (list): contains mean value for each channel of input RGB images
        std (list): contains std value for each channel of input RGB images
        loader (callable, optional): A function to load an image given its path.
        nnum (int, Default:5): Number of negatives for a query image in a training tuple
        qsize (int, Default:1000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        poolsize (int, Default:10000): Pool size for negative images re-mining
        batch_size (int): number of images processed in a single network pass-through

     Attributes:
        images (list): List of full filenames for each image
        clusters (list): List of clusterID per image
        qpool (list): List of all query image indexes
        ppool (list): List of positive image indexes, each corresponding to query at the same position in qpool

        qidxs (list): List of qsize query image indexes to be processed in an epoch
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs

        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method,
            ie new q-p pairs are picked and negative images are remined
    """
    def __init__(self, name, imsize, mean, std, nnum=5, pnum=1, qsize=2000, poolsize=20000,
                 batch_size=20, loader=default_loader, num_workers=8):

        # initializing tuples dataset
        self.name = name
        self.imsize = imsize
        self.mean = mean
        self.std = std

        # size of training subset for an epoch
        self.nnum = nnum
        self.pnum = pnum
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None
        self.qsize = qsize
        self.poolsize = poolsize

        self.loader = loader
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.print_freq = 100

    def __len__(self):
        return self.qsize

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name: {}\n'.format(self.name)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_vecs(self, net, idxs):
        self.dataloader.batch_sampler.sampler = idxs
        vecs = torch.zeros(net.meta['outputdim'], len(idxs)).cuda()

        for i, input in enumerate(self.dataloader):
            x = net(input.cuda())
            vecs[:, i * self.batch_size:(i + 1) * self.batch_size] = x.data

            if (i + 1) % self.print_freq == 0 or (i + 1) == len(self.dataloader):
                print('\r>>>> {}/{} done...'.format(i + 1, len(self.dataloader)), end='')
        print('')
        return vecs
