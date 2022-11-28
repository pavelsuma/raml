import os
import random
from collections import defaultdict

import numpy as np
from scipy import io
from PIL import Image, ImageFile

import torch
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

def imresize(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img

def imrescale(img, imsize, side=max):
    ratio = imsize / side(img.size)
    img = img.resize((round(ratio * img.size[0]), round(ratio * img.size[1])), Image.ANTIALIAS)
    return img

def get_random_aug(img, transform, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return transform(img)

class Rescale(object):
    def __init__(self, imsize, side=max):
        self.imsize = imsize
        self.side = side

    def __call__(self, img):
        return imrescale(img, self.imsize, self.side)

def fg_eval_transform(imsize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        Rescale(imsize + (imsize/7), min),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_dataset_config(data_root, dataset, val_ratio=1/2):
    if dataset.startswith('cub'):
            cfg = give_cub_datasets(data_root, val_ratio=val_ratio)
    elif dataset.startswith('cars'):
            cfg = give_cars196_datasets(data_root, val_ratio=val_ratio)
    elif dataset.startswith('sop'):
            cfg = give_sop_datasets(data_root, val_ratio=val_ratio)
    else:
        raise ValueError(f'Train/test split config not implemented for dataset {dataset}!')
    return cfg


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret

# Functions to load fine-grained datasets adapted from https://github.com/yash0307/RecallatK_surrogate by Yash Patel
def give_cars196_datasets(data_root, val_ratio):
    dataset_root = os.path.join(data_root, 'cars')
    all_image_dict = defaultdict(list)
    data = io.loadmat(os.path.join(dataset_root, 'cars_annos.mat'))['annotations'][0]
    for entry in data:
        data_set = entry[6][0][0]
        im_path = os.path.join(dataset_root, entry[0][0])
        class_id = entry[5][0][0]
        all_image_dict[class_id].append(im_path)

    split = {'train': list(all_image_dict.values())[:int(98 * (1 - val_ratio))],
             'val': list(all_image_dict.values())[int(98 * (1 - val_ratio)):98],
             'test': list(all_image_dict.values())[98:]}

    return split


def give_cub_datasets(data_root, val_ratio):
    dataset_root = os.path.join(data_root, 'cub')
    all_image_dict = defaultdict(list)
    images_data = open(os.path.join(dataset_root, 'images.txt'), 'r').read().splitlines()
    split_data = open(os.path.join(dataset_root, 'train_test_split.txt'), 'r').read().splitlines()
    data_dict = {}
    for given_sample in images_data:
        key = given_sample.split(' ')[0]
        path = given_sample.split(' ')[1]
        data_dict[key] = [path]
    for given_sample in split_data:
        key = given_sample.split(' ')[0]
        split = int(given_sample.split(' ')[1])
        data_dict[key].append(split)
    for entry in data_dict.keys():
        given_sample = data_dict[entry]
        class_id = int(given_sample[0].split('.')[0])
        im_path = os.path.join(dataset_root, 'images', given_sample[0])
        split = given_sample[1]
        all_image_dict[class_id].append(im_path)

    split = {'train': list(all_image_dict.values())[:int(100 * (1 - val_ratio))],
             'val': list(all_image_dict.values())[int(100 * (1 - val_ratio)):100],
             'test': list(all_image_dict.values())[100:]}

    return split


def give_sop_datasets(data_root, val_ratio):
    dataset_root = os.path.join(data_root, 'sop')
    train_image_dict, test_image_dict = defaultdict(list), defaultdict(list)
    train_data = open(os.path.join(dataset_root, 'Ebay_train.txt'), 'r').read().splitlines()[1:]
    test_data = open(os.path.join(dataset_root, 'Ebay_test.txt'), 'r').read().splitlines()[1:]
    for entry in train_data:
        info = entry.split(' ')
        class_id = info[1]
        im_path = os.path.join(dataset_root, info[3])
        train_image_dict[class_id].append(im_path)
    for entry in test_data:
        info = entry.split(' ')
        class_id = info[1]
        im_path = os.path.join(dataset_root, info[3])
        test_image_dict[class_id].append(im_path)

    train_split_size = int(len(train_image_dict.keys()) * (1 - val_ratio))
    split = {'train': list(train_image_dict.values())[:train_split_size],
             'val': list(train_image_dict.values())[train_split_size:],
             'test': list(test_image_dict.values())}
    return split