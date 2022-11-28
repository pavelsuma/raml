import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import timm

from lib.layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC
from lib.layers.normalization import L2N
from lib.datasets.genericdataset import ImagesFromList

# possible global pooling layers
POOLING = {
    'mac'    : MAC,
    'spoc'   : SPoC,
    'gem'    : GeM,
    'gemmp'  : GeMmp,
    'rmac'   : RMAC,
    'no_pool': nn.Identity
}

class ImageRetrievalNet(nn.Module):
    
    def __init__(self, backbone, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.backbone = backbone
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        o = self.backbone(x)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1, 0)

    @property
    def device(self):
        tmp = self.backbone if type(self.backbone) == nn.Sequential else self.backbone.blocks
        return tmp[0].weight.device

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(params):
    architecture = params.get('architecture', 'resnet50')
    pooling = params.get('pooling', 'gem')
    whitening = params.get('whitening', 0)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # load network from torchvision or timm and remove any classifier/pooling
    if hasattr(torchvision.models, architecture):
        backbone = getattr(torchvision.models, architecture)(pretrained=pretrained)
        if hasattr(backbone, 'features'):
            dim = backbone.classifier[-1].in_features
            backbone = backbone.features
        else:
            backbone = list(backbone.children())
            dim = backbone[-1].in_features
            while any(x in type(backbone[-1]).__name__.lower() for x in ('pool', 'linear')):
                backbone.pop()
            backbone = nn.Sequential(*backbone)

    elif architecture in timm.list_models():
        backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0, global_pool=None)
        mean = backbone.pretrained_cfg['mean']
        std = backbone.pretrained_cfg['std']
        dim = backbone.num_features
    else:
        raise ValueError('Architecture not found in torchvision neither timm!')

    # initialize pooling
    if pooling == 'gemmp':
        pool = POOLING[pooling](mp=dim)
    else:
        pool = POOLING[pooling]()

    # initialize whitening layer
    if whitening:
        whiten = nn.Linear(dim, whitening, bias=True)
        dim = whitening
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture,
        'pooling' : pooling,
        'whitening' : whitening,
        'mean' : mean,
        'std' : std,
        'outputdim' : dim,
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(backbone, pool, whiten, meta)
    return net


def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10, workers=8, batch_size=1):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False
    )

    # extracting vectors
    with torch.no_grad():
        vecs = torch.zeros(net.meta['outputdim'], len(images))
        for i, input in enumerate(loader):
            input = input.cuda()

            if len(ms) == 1 and ms[0] == 1:
                vecs[:, i * batch_size:(i + 1) * batch_size] = extract_ss(net, input)
            else:
                vecs[:, i * batch_size:(i + 1) * batch_size] = extract_ms(net, input, ms, msp)

            if (i+1) % print_freq == 0 or (i+1) == len(loader):
                print('\r>>>> {}/{} done...'.format((i+1), len(loader)), end='')
        print('')

    return vecs

def extract_ss(net, input):
    return net(input).cpu().data

def extract_ms(net, input, ms, msp):
    
    v = torch.zeros(net.meta['outputdim'])
    
    for s in ms: 
        if s == 1:
            input_t = input.clone()
        else:    
            input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(input_t).pow(msp).cpu().data.squeeze()
        
    v /= len(ms)
    v = v.pow(1./msp)
    v /= v.norm()

    return v