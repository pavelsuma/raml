import math
import os
import torch
from torch.utils.model_zoo import load_url

from lib.networks.imageretrievalnet import init_network

PRETRAINED = {
    # teacher models pretrained with triplet loss
    "cub-448r-resnet50-512-gem-w"               : 'http://ptak.felk.cvut.cz/personal/sumapave/public/cub-448r-resnet50-512-gem-w.pth',
    "cars-448r-resnet50-512-gem-w"              : 'http://ptak.felk.cvut.cz/personal/sumapave/public/cars-448r-resnet50-512-gem-w.pth',
    "sop-448r-resnet50-512-gem-w"               : 'http://ptak.felk.cvut.cz/personal/sumapave/public/sop-448r-resnet50-512-gem-w.pth',
    "cub-224r-resnet50-512-gem-w"               : 'http://ptak.felk.cvut.cz/personal/sumapave/public/cub-224r-resnet50-512-gem-w.pth',
    "cars-224r-resnet50-512-gem-w"              : 'http://ptak.felk.cvut.cz/personal/sumapave/public/cars-224r-resnet50-512-gem-w.pth',
    "sop-224r-resnet50-512-gem-w"               : 'http://ptak.felk.cvut.cz/personal/sumapave/public/sop-224r-resnet50-512-gem-w.pth',
}

def load_model(data_root, args=None, arch=None, path=None):
    if path is not None:
        model_path = os.path.join(data_root, 'networks')
        print(">> Loading network:\n>>>> '{}'".format(path))
        if path in PRETRAINED:
            # pretrained networks (downloaded automatically)
            state = load_url(PRETRAINED[path], model_dir=model_path)
        elif os.path.exists(os.path.join(model_path, path + '.pth')):
            # model saved in the networks directory
            state = torch.load(os.path.join(model_path, path + '.pth'))
        else:
            # fine-tuned network from path
            state = torch.load(path)

        if 'max_map' in state:
            print(f"Loaded model {path} epoch {state['epoch']} with val mAP: {state['max_map']}'")

        # parsing net params from meta - architecture, pooling, mean, std required
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['whitening'] = state['meta']['whitening']
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False

        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])

        print(">>>> loaded network: ")
        print(net.meta_repr())

    # Build ImageNet pretrained network based on arguments
    elif arch is not None:
        net_params = {
            'architecture': arch,
            'pooling': args.pool,
            'whitening': args.whitening,
            'pretrained': args.pretrained
        }
        if args.pretrained:
            print(">> Using pre-trained model '{}'".format(net_params['architecture']))
        else:
            print(">> Using model from scratch (random weights) '{}'".format(net_params['architecture']))

        net = init_network(net_params)
    else:
        raise ValueError('Unsupported or unknown architecture!')

    return net


def resume_model(model, optimizer, scheduler, path, directory=None):
    if not os.path.isfile(path) and directory is not None:
        print(">> Finding the last checkpoint")
        all_file = os.listdir('/'.join(path.split('/')[:-1]))
        last_ckpt = 0
        ckpt_iter = 0
        for f in all_file:
            if f.startswith('model_epoch'):
                ckpt_temp = int(all_file[ckpt_iter].split('.')[0].split('model_epoch')[1])
                if ckpt_temp > last_ckpt:
                    last_ckpt = ckpt_temp
            ckpt_iter += 1
        path = os.path.join(directory, 'model_epoch' + str(last_ckpt) + '.pth.tar')
        if os.path.isfile(path):
            print(">> No checkpoint found at '{}'".format(path))
            return model, optimizer, 0

    # load checkpoint weights and update model and optimizer
    print(">> Loading checkpoint:\n>> '{}'".format(path))
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
          .format(path, checkpoint['epoch']))

    return model, optimizer, scheduler, start_epoch


def get_model_optimizer(args, model, dataset):
    # parameters split into features, pool, whitening
    # IMPORTANT: no weight decay for pooling parameter p in GeM or regional-GeM
    parameters = []

    # add feature parameters
    parameters.append({'params': model.backbone.parameters()})
    # global, only pooling parameter p weight decay should be 0
    if args.pool == 'gem':
        parameters.append({'params': model.pool.parameters(), 'lr': args.lr * 10, 'weight_decay': 0})
    elif args.pool == 'gemmp':
        parameters.append({'params': model.pool.parameters(), 'lr': args.lr * 100, 'weight_decay': 0})

    # add final whitening if exists
    if model.whiten is not None:
        parameters.append({'params': model.whiten.parameters()})

    # define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(parameters, args.lr, weight_decay=args.weight_decay)

    # define scheduler
    if args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, div_factor=10,
                                                        steps_per_epoch=math.ceil(len(dataset) / args.batch_size / args.update_every),
                                                        epochs=args.epochs)
    elif args.scheduler == 'cyclic':
        step_size_up = int(args.epochs * len(dataset) / args.batch_size / args.update_every / 6)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=args.lr, base_lr=args.lr/10,
                                                      step_size_up=step_size_up, mode='triangular2', cycle_momentum=False)
    elif args.scheduler == 'exponential':
        exp_decay = math.exp(-0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
    else:
        scheduler = None

    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, val_map, epoch, directory):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'max_map': val_map,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
    }
    filename = os.path.join(directory, 'model_best.pth.tar')
    torch.save(state, filename)