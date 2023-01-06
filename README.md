## Resolution Asymmetric Metric Learning

**Large-to-small Image Resolution Asymmetry in Deep Metric Learning**,
Suma P., Tolias G. 
[[arXiv](https://arxiv.org/abs/2210.05463)]

### Content

This repository provides the means to train and test all the models presented in the paper. This includes:

1. Code to train the models with and without the teacher (asymmetric and symmetric).
1. Code to do symmetric and asymmetric testing.
1. Best pre-trainend teacher models.

Other standard backbone architectures from `torchvision` and `timm` should load off the bat.

### Dependencies
Necessary python libraries to run this repository are specified in the file `requirements.txt` which can be installed with `pip install -r requirements.txt`.
If you wish to optimize the hyperparameters, you have to additionally `pip install optuna`.


### Reproducing the paper
First, download the datasets used in the paper with the provided commands below (assuming the data to be located in the current folder).

**CUB**
```bash
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
mkdir -p data/cub && tar -xf CUB_200_2011.tgz -C data/cub --strip-components=1
rm CUB_200_2011.tgz
```

**Cars196**
```bash
mkdir -p data/cars && cd $_
wget http://ai.stanford.edu/~jkrause/car196/car_ims.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_annos.mat
tar -xf car_ims.tgz && rm car_ims.tgz
```

**SOP**
```bash
wget ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip -P data/sop
cd data/sop && ln -s . Stanford_Online_Products && unzip Stanford_Online_Products.zip
rm Stanford_Online_Products && rm Stanford_Online_Products.zip
```

Minimal command to reproduce the resolution distillation approach of our paper is:

```bash
python main.py [-h] (--student-path NETWORK | --student NETWORK)
                    (--teacher-path NETWORK | --teacher NETWORK)
                    [--training-dataset DATASET] [--test-datasets DATASETS] 
                    [--directory EXPORT_DIR] [--data-root DATA_DIR]
                    [--teacher-image-size DR] [--image-size QR]
                    [--lr LR] [--lam-1 L1] [--lam-2 L2]
```
Optimized hyperparameters for each dataset and the default seed (1234) along with the resulting score:

| Dataset | DR  | QR  | LR        | λ<sub>1 | λ<sub>2 |   |  Asym mAP   |   Asym R@1   |  Sym mAP   |   Sym R@1   |
|---------|-----|-----|-----------|---------|---------|---|:-----------:|:------------:|:----------:|:-----------:|
| cub     | 448 | 224 | 1.4025e-4 | 0.7718  | 0.6684  |   |    40.80    |    70.43     |   41.08    |    71.79    |
| cars    | 448 | 224 | 1.1692e-4 | 0.6371  | 0.7731  |   |    39.43    |    85.06     |   40.19    |    86.87    |
| sop     | 448 | 224 | 1.7579e-4 | 0.7482  | 0.6778  |   |    60.23    |    81.14     |   60.49    |    81.44    |

The available TEACHER model names have the following format `{dataset}-{resolution}r-resnet50-512-gem-w`, 
such as `cub-448r-resnet50-512-gem-w`

#### Example:

```bash
python main.py --training-dataset 'cub' --test-datasets 'cub-test' \
               --directory 'exp' --data-root 'data' \
               --student 'resnet50' --teacher-path 'cub-448r-resnet50-512-gem-w' \
               --teacher-image-size 448 --image-size 224 \
               --lr 1.4025e-4 --lam-1 0.7718 --lam-2 0.6684
```

### Test the models
To only obtain a test performance of a model:
```bash
python test.py [-h] (--student-path NETWORK | --student NETWORK)
                    (--teacher-path NETWORK | --teacher NETWORK)
               [--datasets DATASETS] [--data-root DATA_DIR]
               [--image-size N] [--teacher-image-size N] 
               [--sym] [--workers]
```
Asymmetric test only runs if either `--teacher-path` or `--teacher` is specified. 
If `--sym` argument is present, symmetric test is performed. To test only a single model sym. performance, load the network as a student.


#### Examples:

Perform symmetric test with a pre-trained teacher model:

```bash
python test.py --datasets 'cub-test' --sym \
               --student-path 'cub-448r-resnet50-512-gem-w' \ 
               --image-size 224
```

Perform both symmetric and asymmetric test of distilled model:
```bash
python test.py --datasets 'cub-test' --sym \
               --student-path 'cub-S224r-T448r-resnet50-512-gem-w' \
               --teacher-path 'cub-448r-resnet50-512-gem-w' \
               --image-size 224 --teacher-image-size 448
```

### Using your own dataset

This repository defines a train/val/test split in a dictionary file with the following structure:

```python
split = {
    'train': [[class_1_img_1, class_1_img_2], [class_2_img_1, ...], ...],
    'test': [[...]],
    'val': [[...]]
}
```
where each list contains lists of strings of image paths coming from the same class. The given paths should be relative to the dataset folder.
In order to register your dataset, add an if-branch in `get_dataset_config` function inside `lib/datasets/datahelpers.py` which loads the *split* dictionary.
By default, the code looks up the dataset folder with the given name in the `data_root` folder specified in arguments.


### Additional run arguments
All possible arguments for training are listed below. 
```bash
python main.py [-h] (--student-path SNETWORK | --student STUDENT)
                    (--teacher-path TNETWORK | --teacher TEACHER)
                  [--pool POOL] [--whitening DIM] 
                  [--training-dataset DATASET] [--directory EXPORT_DIR]
                  [--test-datasets DATASETS] [--data-root DATA_DIR]
                  [--no-val] [--val-freq N] [--save-freq N]
                  [--not-pretrained] [--loss LOSS] [--loss-margin LM] 
                  [--mode MODE] [--image-size N] [--teacher-image-size N]
                  [--neg-num N] [--query-size N] [--pool-size N]
                  [--pos-num N] [--lam-1 N] [--lam-2 N] 
                  [--batch-size N] [--update-every N] [--epochs N] [--lr LR]
                  [--optimizer OPTIMIZER] [--scheduler SCHEDULER]
                  [--gpu-id N] [--workers N] [--momentum M] [--weight-decay W]
                  [--print-freq N] [--comment COMMENT]
                  [--optimize] [--seed SEED] [--resume-student FILENAME]
                  
```

- `--data-root` is the system path to the folder with datasets
- `--directory` is the system path to the output of the single experiment. Resulting model and optionally tensorboard log will be stored there.
- `--pool` selects the global pooling method after the last embedding layer of the backbone network, options are `gem`, `gemmp`, `mac` (max), `spoc` (avg)
- `--whitening` appends a linear layer after the global pooling with the specified output dimension
- `--mode` can be one of `ts_aug`, `ts_reg` or `sym`. The first two correspond to distillation with the full loss or just the absolute term. The last option is for standard symmetric metric learning with triplet loss.
- `--pos-num` sets the number of augmentations per image in distillation modes
- `--query-size` defines how many images are processed in one epoch (bounded by the total dataset size)
- `--loss-margin` only applies to triplet loss, i.e. when mode is set to `--sym`
- `--neg-num` represents the number of negatives per anchor in triplet loss
- `--pool-size` is the size of the randomly selected pool of images used to mine negatives for triplet loss (bounded by the total dataset size)
- `--scheduler` can be one of `onecycle`, `cyclic`, `exponential` or none
- `--optimizer` can be one of `adam`, `adamW` and `sgd`
- `--update-every` may be facilitated to create virtually larger batch size, such that the total batch size is `batch-size * update-every` but only `batch-size * pos-num` images have to fit in the VRAM at the same time

### Acknowledgements

This code is based on the two amazing repositories:
1. [CNN Image Retrieval in PyTorch: Training and evaluating CNNs for Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) by F. Radenović.
2. [Asymmetric metric learning](https://github.com/budnikm/aml) by M. Budnik.

