import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from lib.datasets.datahelpers import fg_eval_transform, get_random_aug
from lib.datasets.genericdataset import ImagesFromList
from lib.datasets.traindataset import RetrievalDataset, mine_dist


class FineGrainDataset(RetrievalDataset):
    def __init__(self, cfg, *args, spc=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples_per_class = spc
        self.cfg, self.clusters, self.images = [], [], []
        self.min_class = 1e6

        for class_id, grp in enumerate(cfg['train']):
            grp_ids = np.arange(len(grp)) + len(self.images)
            if len(grp_ids) >= self.samples_per_class:
                self.images.extend(grp)
                self.clusters.extend([class_id] * len(grp))
                self.cfg.append(grp_ids)
                self.min_class = min(self.min_class, len(grp_ids))

        self.clusters = np.asarray(self.clusters)
        self.qsize = min(self.qsize, len(self.images))

    def __len__(self):
        return self.qsize

    def __getitem__(self, index):
        pass


class TuplesFineGrainDataset(FineGrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.imsize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.qsize = min(self.qsize, self.min_class // self.samples_per_class * self.samples_per_class * len(self.cfg))

        self.dataloader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=self.images, imsize=None,
                           transform=fg_eval_transform(self.imsize, self.mean, self.std)),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=False
        )

    def __getitem__(self, index):
        if self.__len__() == 0:
            raise (RuntimeError(
                "List batch is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []

        output.append(self.loader(self.images[self.qidxs[index]]))
        for j in range(len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index, j]]))

        if self.transform is not None:
            output = [self.transform(img) for img in output]

        return output[0], self.clusters[self.qidxs[index]], torch.stack(output[1:])

    def create_epoch_tuples(self, net):
        print('>> Creating tuples for an epoch of {}...'.format(self.name))

        # prepare network
        net.cuda()
        net.eval()

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------

        classes_no = len(self.cfg)
        ims_per_cls = self.min_class // self.samples_per_class * self.samples_per_class
        classes = np.asarray(list(map(lambda x: np.random.permutation(x)[:ims_per_cls], self.cfg)))
        classes = np.random.permutation(classes)

        self.qidxs = classes.reshape(classes_no, -1, self.samples_per_class).transpose(2, 1, 0)\
            .reshape(self.samples_per_class, -1).T.flatten()[:self.qsize]

        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.batch))]
        else:
            with torch.no_grad():
                idxs2images = torch.randperm(len(self.images))[:self.poolsize]
                print('>> Extracting descriptors for negative pool...')
                if (self.poolsize + self.qsize) >= 1.1*len(self.images):
                    qpool = np.arange(len(self.images))
                    qvecs = self.get_vecs(net, qpool)
                    poolvecs = qvecs[:, idxs2images]
                    qvecs = qvecs[:, self.qidxs]
                else:
                    qpool = self.qidxs
                    qvecs = self.get_vecs(net, qpool)
                    poolvecs = self.get_vecs(net, idxs2images)

                nidxs = mine_dist(poolvecs, qvecs, self.clusters[idxs2images], self.clusters[qpool], self.nnum)
                self.nidxs = idxs2images[nidxs]


class FineGrainRegressionTS(FineGrainDataset):
    def __init__(self, *args, teacher_feat, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat = teacher_feat
        self.transform = fg_eval_transform(self.imsize)

    def __getitem__(self, index):
        img = self.loader(self.images[self.qidxs[index]])
        img = self.transform(img)

        return img.unsqueeze(0), self.feat[:, self.qidxs[index]].unsqueeze(0)

    def create_epoch_tuples(self, net):
        self.qidxs = torch.randperm(len(self.images))[:self.qsize]
        return 0


class AugFineGrainTS(FineGrainDataset):

    def __init__(self, *args, teacher_imsize, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_imsize = teacher_imsize
        base_transform = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=.4, hue=.1, saturation=.4, contrast=.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]
        self.s_transform = transforms.Compose([transforms.RandomResizedCrop(self.imsize)] + base_transform)
        self.t_transform = transforms.Compose([transforms.RandomResizedCrop(self.teacher_imsize)] + base_transform)

    def __getitem__(self, index):
        if self.__len__() == 0:
            raise (RuntimeError(
                "List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        img = self.loader(self.images[self.qidxs[index]])
        img_1 = []
        img_2 = []
        for i in range(self.pnum):
            seed = np.random.randint(2147483647)
            i1 = get_random_aug(img, self.s_transform, seed)
            i2 = get_random_aug(img, self.t_transform, seed)

            # mixup
            rnd_id = np.random.randint(len(self.images))
            mixup_img = self.loader(self.images[rnd_id])
            mixup_i1 = get_random_aug(mixup_img, self.s_transform, seed)
            mixup_i2 = get_random_aug(mixup_img, self.t_transform, seed)
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            img_1.append(lam * i1 + (1 - lam) * mixup_i1)
            img_2.append(lam * i2 + (1 - lam) * mixup_i2)

        return torch.stack(img_1), torch.stack(img_2)

    def create_epoch_tuples(self, net):
        self.qidxs = torch.randperm(len(self.images))[:self.qsize]
        return 0
