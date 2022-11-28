import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from lib.datasets.genericdataset import ImagesFromList
from lib.layers.functional import activations

class Logger:
    def __init__(self, exp_name):
        self.exp_name = exp_name

    def log_metadata(self, args):
        print(json.dumps(vars(args)))

    def set_eval(self):
        pass

    def set_epoch(self, epoch):
        pass

    def log_text(self, texts):
        pass

    def log_scalars(self, scalars):
        pass


class TBLogger(Logger):
    def __init__(self, directory, exp_name):
        super().__init__(exp_name)
        from tensorboardX import SummaryWriter
        self.tb_writer = SummaryWriter(directory)
        self.epoch = 0
        self.eval = False

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_eval(self):
        self.eval = True

    def log_metadata(self, args):
        self.tb_writer.add_text('args', json.dumps(vars(args)))

    def log_text(self, texts):
        for entry_name, entry in texts.items():
            self.tb_writer.add_text(entry_name, str(entry))

    def log_scalars(self, scalars):
        self.tb_writer.flush()
        for scalar_name, scalar in scalars.items():
            self.tb_writer.add_scalar(scalar_name, scalar, self.epoch)

    def log_histogram(self, name, hist):
        self.tb_writer.add_histogram(name, hist, self.epoch)

    def save_img(self, net, images, imsize=1024, transform=None):
        dataloader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=images, imsize=imsize, transform=transform),
            batch_size=1, shuffle=False, num_workers=8, drop_last=False
        )

        for i, input in enumerate(dataloader):
            input = input.cuda()
            learned_img = net.blocks[0][0](input).squeeze_(0)
            self.tb_writer.add_image(f'maps_{i}', learned_img, 0)
            _, maps = net.process_backbone(input)
            for k, m in enumerate(maps):
                act = activations(m)
                act *= 1.0 / act.max()
                self.tb_writer.add_image(f'maps_{i}', act.squeeze_(0), k+1)


def log_eval(logger, log_res, str_res):
    if not hasattr(logger, 'eval') or not logger.eval:
        logger.log_scalars(log_res)
    else:
        logger.log_text({f'eval': str_res})


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it in CHW"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img).transpose((2, 0, 1))

def get_logger(log_type, directory, exp_name):
    if log_type == 'tensorboard':
        logger = TBLogger(Path(directory).parent / 'tensorboard' / exp_name, exp_name)
    else:
        print("No logger or unknown type selected.")
        logger = Logger(exp_name)
    return logger