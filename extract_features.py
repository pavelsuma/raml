import argparse
import itertools as it
import os
import time

import numpy as np
import torch
from PIL import Image, ImageFile

from lib.datasets.datahelpers import get_dataset_config, keydefaultdict, fg_eval_transform
from lib.networks.imageretrievalnet import extract_vectors
from lib.utils.general import htime
from modelhelpers import load_model

datasets_names = ['cub', 'cub-val', 'cub-test', 'cars', 'cars-val', 'cars-test', 'sop', 'sop-val', 'sop-test']

parser = argparse.ArgumentParser(description='Feature extractor for a given model and dataset.')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help="pretrained network or network path (destination where network is saved)")
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help="off-the-shelf network, in the format 'ARCHITECTURE-POOLING' or 'ARCHITECTURE-POOLING-{reg-lwhiten-whiten}'," + 
                        " examples: 'resnet101-gem' | 'resnet101-gem-reg' | 'resnet101-gem-whiten' | 'resnet101-gem-lwhiten' | 'resnet101-gem-reg-whiten'")
parser.add_argument('--data-root', metavar='DATA_DIR',
                    help='destination where the datasets are located')
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='cub,cub-val,cub-test',
                    help="comma separated list of test datasets: " + 
                        " | ".join(datasets_names) + 
                        " (default: 'cub, cub-val, cub-test')")
parser.add_argument('--image-size', '-imsize', default=448, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]', 
                    help="use multiscale vectors for testing, " + 
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")


def pil_loader(path):
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning 
    #(https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def main():
    args = parser.parse_args()

    # check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))
    
    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    net = load_model(args.data_root, args, None, args.network_path)

    # setting up the multiscale parameters
    ms = list(eval(args.multiscale))
    if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    feat_dir = os.path.join(args.data_root, 'features')
    # evaluate on test datasets
    datasets = args.datasets.split(',')
    cfgs = keydefaultdict(lambda x: get_dataset_config(args.data_root, x, val_ratio=1/2))
    for dataset in datasets:
        vecs = extract_features(dataset, cfgs[dataset], args.image_size, net, ms, msp)
        out_path = f'{feat_dir}/{dataset}_{"ms" if len(ms) > 1 else "ss"}{args.image_size}r_' \
                   f'{args.network_path.split("/")[-1][:-4]}'
        np.save(out_path, vecs)


def load_features(datasets, args, cfgs, ms=(1,), msp=1):
    feats = {}
    model = None

    for dataset in datasets:
        feat_path = os.path.join(args.data_root, 'features', f'{dataset}_{"ms" if len(ms) > 1 else "ss"}'
                                                             f'{args.teacher_image_size}r_{args.teacher}.npy')
        if os.path.exists(feat_path):
            vecs = torch.from_numpy(np.load(feat_path))
        else:
            if model is None:
                model = load_model(args.data_root, args, args.teacher, args.teacher_path)
            vecs = extract_features(dataset, cfgs[dataset], args.teacher_image_size, model, ms, msp)
        feats[dataset] = vecs

    return feats

def extract_features(dataset, cfg, image_size, net, ms, msp):
    start = time.time()
    print('>> {}: Extracting...'.format(dataset))

    d = dataset.split('-')
    mode = d[1] if len(d) > 1 else 'train'
    transform = fg_eval_transform(image_size, net.meta['mean'], net.meta['std'])
    vecs = extract_vectors(net, list(it.chain(*cfg[mode])), None, transform, ms=ms, msp=msp, batch_size=50)

    print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))

    return vecs

if __name__ == '__main__':
    main()