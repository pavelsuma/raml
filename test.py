import argparse
import os
import time
import numpy as np
import torch

from lib.cli import pool_names
from lib.datasets.datahelpers import get_dataset_config, keydefaultdict
from lib.datasets.testdataset import get_testsets
from lib.utils.evaluate import compute_map_and_print
from lib.utils.general import htime
from logger import Logger
from modelhelpers import load_model
from extract_features import load_features


parser = argparse.ArgumentParser(description='Asymmetric Image Retrieval Testing')

# network
student_model = parser.add_mutually_exclusive_group(required=True)
student_model.add_argument('--student', '-s', metavar='STUDENT', default='resnet50')
student_model.add_argument('--student-path', '-spath', metavar='SNETWORK',
                    help="pretrained network or network path (destination where network is saved)")

teacher_model = parser.add_mutually_exclusive_group(required=False)
teacher_model.add_argument('--teacher', '-t', metavar='TEACHER')
teacher_model.add_argument('--teacher-path', '-tpath', metavar='TNETWORK',
                    help="pretrained network or network path (destination where network is saved)")
parser.add_argument('--pool', '-p', metavar='POOL', default='gem', choices=pool_names,
                        help='pooling options: ' +
                            ' | '.join(pool_names) +
                            ' (default: gem)')
parser.add_argument('--whitening', '-w', default=0, type=int,
                        help='set the final embedding dimension given by learnable whitening (linear layer) after the pooling')

# test options
parser.add_argument('--data-root', metavar='DATA_DIR', default='data',
                    help='destination where the datasets are located')
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='cub-test',
                    help="comma separated list of test datasets")
parser.add_argument('--image-size', '-imsize', default=224, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--teacher-image-size', '-timsize', default=448, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]', 
                    help="use multiscale vectors for testing, " + 
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")
parser.add_argument('--sym', dest='sym', action='store_true',
                    help='Runs symmetric testing by default')

def main():
    args = parser.parse_args()
    args.pretrained = True

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    net_student = load_model(args.data_root, args, args.student, args.student_path)

    # setting up the multiscale parameters
    ms = list(eval(args.multiscale))
    if len(ms)>1 and net_student.meta['pooling'] == 'gem' and not net_student.meta['regional'] and not net_student.meta['whitening']:
        msp = net_student.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))            
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1

    test_dataset_names = args.datasets.split(',')
    cfgs = keydefaultdict(lambda x: get_dataset_config(args.data_root, x))
    feats = {}
    if args.teacher or args.teacher_path:
        feats = load_features(test_dataset_names, args, cfgs)
    datasets = get_testsets(test_dataset_names, args, cfgs, feats)
    net_student.cuda()

    logger = Logger('no_logger')
    avg_score = run_tests(datasets, None, net_student, args.image_size, args.teacher_image_size,
                          logger=logger, asym=args.teacher is not None or args.teacher_path is not None, sym=args.sym, ms=ms, msp=msp)
    print(f"Average test score for datasets: {avg_score * 100}")


def run_tests(datasets, t_model, s_model, image_size, t_image_size, logger=None, asym=True, sym=True, ms=(1,), msp=1):
    s_model.eval()
    if t_model:
        t_model.eval()

    results = []
    for dataset in datasets:
        with torch.no_grad():
            mAP = test(t_model, s_model, dataset, image_size, t_image_size, ms, msp, logger, asym=asym, sym=sym)
            results.append(mAP)

    avg_score = np.mean(results)
    return avg_score


def find_ranks(vecs, qvecs):
    scores = torch.mm(vecs.T, qvecs)
    return torch.argsort(scores, descending=True, dim=0)


def test(net_teacher, net_student, dataset, image_size, t_image_size, ms, msp, logger=None, asym=True, sym=True):
    # evaluate on test datasets
    start = time.time()
    print('>> {}: Extracting...'.format(dataset.name))
    qvecs = dataset.extract_query(net_student, ms, msp, image_size)

    print('>> {}: Evaluating...'.format(dataset.name))
    if asym:
        vecs = dataset.load_or_extract_db(net_teacher, ms, msp, t_image_size)
        ranks = find_ranks(vecs, qvecs)
        asym_mAP = compute_map_and_print(dataset.name, ranks.numpy(), dataset.cfg, logger)

    if sym:
        ranks = find_ranks(qvecs, qvecs)
        sym_mAP = compute_map_and_print(dataset.name + ' + sym', ranks.numpy(), dataset.cfg, logger)

    print('>> {}: elapsed time: {}'.format(dataset.name, htime(time.time()-start)))
    return asym_mAP if asym else sym_mAP if sym else 0


if __name__ == '__main__':
    main()
