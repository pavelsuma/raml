import argparse

pool_names = ['no_pool', 'mac', 'spoc', 'gem', 'gemmp', 'rmac']
mode_names = ['ts_reg',  'ts_aug', 'sym']
optimizer_names = ['sgd', 'adam', 'adamW']
scheduler_names = ['onecycle', 'cyclic', 'exponential']


def create_parser():
    parser = argparse.ArgumentParser(description='Resolution Asymmetric Metric Learning')
    
    # export directory, training and val datasets, test datasets
    parser.add_argument('--data-root', metavar='DATA_DIR', default="data",
                        help='destination where the datasets are located')
    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')
    parser.add_argument('--training-dataset', '-d', metavar='DATASET', default='cub')
    parser.add_argument('--no-val', dest='val', action='store_false',
                        help='do not run validation')
    parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='roxford5k,rparis6k',
                        help='comma separated list of test datasets')
    parser.add_argument('--val-freq', default=20, type=int, metavar='N',
                        help='run val evaluation every N epochs (default: 20)')
    parser.add_argument('--save-freq', default=20, type=int, metavar='N',
                        help='save model checkpoint every N epochs (default: 20)')
                        
    # network architecture and initialization options
    # network
    student_model = parser.add_mutually_exclusive_group(required=True)
    student_model.add_argument('--student', '-s', metavar='STUDENT', default='resnet50')
    student_model.add_argument('--student-path', '-spath', metavar='SNETWORK',
                               help="pretrained network or network path (destination where network is saved)")

    teacher_model = parser.add_mutually_exclusive_group(required=True)
    teacher_model.add_argument('--teacher', '-t', metavar='TEACHER')
    teacher_model.add_argument('--teacher-path', '-tpath', metavar='TNETWORK',
                               help="pretrained network or network path (destination where network is saved)")

    parser.add_argument('--pool', '-p', metavar='POOL', default='gem', choices=pool_names,
                        help='pooling options: ' +
                            ' | '.join(pool_names) +
                            ' (default: gem)')
    parser.add_argument('--whitening', '-w', default=512, type=int, metavar='DIM',
                        help='set the final embedding dimension given by learnable whitening (linear layer) after the pooling')
    parser.add_argument('--not-pretrained', dest='pretrained', action='store_false',
                        help='initialize model with random weights (default: pretrained on imagenet)')
    parser.add_argument('--mode', '-m', metavar='MODE', default='ts_aug',
                        choices=mode_names,
                        help='training mode options: ' +
                            ' | '.join(mode_names) +
                            ' (default: ts_aug)')
    parser.add_argument('--loss-margin', '-lm', metavar='LM', default=0.7, type=float,
                        help='loss margin: (default: 0.7)')
    parser.add_argument('--lam-1', default=0.0, type=float, metavar='N',
                        help='relative loss lambda 1: (default: 0.0)')
    parser.add_argument('--lam-2', default=0.0, type=float, metavar='N',
                        help='relative loss lambda 2: (default: 0.0)')

    # train/val options specific for image retrieval learning
    parser.add_argument('--image-size', default=224, type=int, metavar='N',
                        help='maximum size of longer image side used for training (default: 1024)')
    parser.add_argument('--teacher-image-size', default=448, type=int, metavar='N',
                        help='maximum size of longer image side used for training (default: 1024)')
    parser.add_argument('--neg-num', '-nn', default=4, type=int, metavar='N',
                        help='number of negative image per train/val tuple (default: 5)')
    parser.add_argument('--pos-num', '-pn', default=8, type=int, metavar='N',
                        help='number of positive images per train/val tuple (default: 1)')
    parser.add_argument('--query-size', '-qs', default=8000, type=int, metavar='N',
                        help='number of queries randomly drawn per one train epoch (default: 2000)')
    parser.add_argument('--pool-size', '-ps', default=20000, type=int, metavar='N',
                        help='size of the pool for hard negative mining (default: 20000)')

    # standard train/val options
    parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                        help='gpu id used for training (default: 0)')
    parser.add_argument('--workers', '-j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--batch-size', '-b', default=8, type=int, metavar='N',
                        help='number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)')
    parser.add_argument('--update-every', '-u', default=25, type=int, metavar='N',
                        help='update model weights every N batches, used to handle really large batches, ' + 
                            'batch_size effectively becomes update_every x batch_size (default: 1)')
    parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adamW',
                        choices=optimizer_names,
                        help='optimizer options: ' +
                            ' | '.join(optimizer_names) +
                            ' (default: adamW)')
    parser.add_argument('--scheduler', metavar='SCHEDULER', default='onecycle', choices=scheduler_names,
                        help='scheduler options: ' +
                            ' | '.join(scheduler_names))
    parser.add_argument('--lr', '--learning-rate', type=float,
                        metavar='LR', help='initial learning rate (default: 1e-6)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                        metavar='W', help='weight decay (default: 1e-6)')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume-student', default='', type=str, metavar='FILENAME',
                        help='name of the latest checkpoint (default: None)')
    parser.add_argument('--comment', '-c', default='', type=str, metavar='COMMENT',
                        help='additional experiment comment')
    parser.add_argument('--logger', type=str, metavar='LOGGER',
                        help='Name of the logger to use, options: <>, tensorboard')
    parser.add_argument('--seed', default=1234, type=int, metavar='SEED',
                        help='random seed for the experiment (default: 0)')
    parser.add_argument('--optimize', dest='optimize', action='store_true',
                        help='Optimize hyperparameters on validation set first.')

    return parser
    
def parse_commandline_args():
    parser = create_parser().parse_args()
    return parser