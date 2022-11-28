from lib.datasets.datahelpers import fg_eval_transform
from lib.networks.imageretrievalnet import extract_vectors

class TestDataset:
    def __init__(self, name, teacher_feat, num_workers=8):
        self.name = name
        self.feat = teacher_feat
        self.num_workers = num_workers

    def extract_db(self, model, ms, msp, imsize):
        pass

    def extract_query(self, model, ms, msp, imsize):
        pass

    def load_or_extract_db(self, model, ms, msp, imsize):
        pass


class FineGrainTestset(TestDataset):
    def __init__(self, cfg, name, **kwargs):
        super().__init__(name, **kwargs)
        self.cfg, self.images = [], []
        for class_id, grp in enumerate(cfg):
            self.images.extend(grp)
            grp = list(range(len(self.images) - len(grp), len(self.images)))
            for k in grp:
                grp_copy = grp.copy()
                grp_copy.remove(k)
                item = {'ok': grp_copy, 'junk': [k]}
                self.cfg.append(item)
        self.batch_size = 50

    def extract_db(self, model, ms, msp, imsize):
        return extract_vectors(model, self.images, None, fg_eval_transform(imsize),
                               ms=ms, msp=msp, workers=self.num_workers, batch_size=self.batch_size)

    def extract_query(self, model, ms, msp, imsize):
        return self.extract_db(model, ms, msp, imsize)

    def extract_all(self, model, ms, msp, imsize):
        return self.extract_db(model, ms, msp, imsize)

    def load_or_extract_db(self, *args):
        return self.feat if self.feat is not None else self.extract_db(*args)


def get_testsets(datasets, args, cfgs, feats):
    for dataset in datasets:
       yield FineGrainTestset(
           cfg=cfgs[dataset]['test'],
           name=dataset,
           teacher_feat=feats.get(dataset),
           num_workers=args.workers
       )
