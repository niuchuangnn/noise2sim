# Suppression of Correlated Noises with Similarity-based Unsupervised Deep Learning


## Introduction
Noise2Sim is a general unsupervised deep denoising method.
On the common benchmark and practical low-dose CT datasets,
Noise2Sim performs as effectively as or even better than the supervised learning methods.
It has potential in various applications.
Details can be found on the [project page](http://chuangniu.info/projects/noise2im/) and in the [paper](https://arxiv.org/abs/2011.03384).

## News
Jan 6, 2022, noise2sim 0.1.2 released

Dec 24, 2021, noise2sim 0.1.0 released


## Installation
[Pytorch](https://pytorch.org/) is required, for example,
```shell script
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch  # install pytorch
```
Then, Noise2Sim can be easily installed through pip,
```shell script
pip install noise2sim
```
Or Noise2Sim can be installed from source,
```shell
git clone https://github.com/niuchuangnn/noise2sim.git
cd noise2sim
pip install -e .
```

## Train denoising model with Noise2Sim

### Low-dose CT Images
The low-dose CT dataset can be obtained at [Low Dose CT Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/).
The preprocessed FDA data can be obtained [here](https://drive.google.com/drive/folders/1ggt6FxBTwmMa8paDgOGQOutu4bNV26Si?usp=sharing).

Arrange the Mayo data like:

    ├── datasets   
        ├── Mayo                   
            ├── L067                    
            ├── L096
            ...

List training and testing files:
```shell
from noise2sim.tools import prepare_ldct
python prepare_ldct.py --patient-folder L067 L096 L109 L143 L192 L286 L291 L310 --output-file ./datasets/Mayo/mayo_train.txt
python prepare_ldct.py --patient-folder L506 L333 --output-file ./datasets/Mayo/mayo_test.txt
```
Run on 4 GPUs:
```shell
python ./scripts/train.py --config-file ./configs/ldct_mayo_unet2.py # for Mayo dataset
```
```shell
python ./scripts/train.py --config-file ./configs/ldct_fda_unet2.py # for FDA dataset
```

### Photon-counting CT Images
The photon-counting datasets can be obtained [here](https://drive.google.com/drive/folders/1ggt6FxBTwmMa8paDgOGQOutu4bNV26Si?usp=sharing),
and put it under ```./datasets/```.

Run on 4 GPUs:
```shell
python ./tools/train.py --config-file ./configs/pcct_livemouse_unet2.py # for live mouse dataset
```
```shell
python ./tools/train.py --config-file ./configs/pcct_leg_unet2.py # for live leg dataset
```
```shell
python ./tools/train.py --config-file ./configs/pcct_diedmouse_unet2.py # for died mouse dataset
```

### Natural Images

Download BSD68 test dataset at [here](https://drive.google.com/drive/folders/1b_RvBwIr9yLg8yPWb0BHYmWiOEVUvG4K?usp=sharing),
and put them under the folder  ```./datasets/```

Prepare dataset:
```shell script
python noise2sim.tools.prepare_bsd400_lmdb.py
```

Run on 1 GPU:
```shell script
python ./scripts/train.py --config-file ./configs/bsd400_unet2_ps3_ns8_gpu1.py # simultaneous training and testing
```
Run on 8 GPUs:
```shell script
python ./sctipts/train.py --config-file ./configs/bsd400_unet2_ps3_ns8_gpu8.py # simultaneous training and testing
```
The results in the paper were obtained using 8 GPUs, you can obtain similar results with 1 GPU.

## Using our pretrained models
Download our pretrained model [here](https://drive.google.com/drive/folders/1l9yLRBlCAo1snjiJrFhkTOPeC_lPoXc7?usp=sharing), and put these models under ```results``` folder.
Then, run the corresponding test script as

```shell
python scripts/test_ldct.py --config-file ./configs/ldct_mayo_unet2.py # for Mayo dataset
```
```shell
python scripts/test_ldct.py --config-file ./configs/ldct_fda_unet2.py # for FDA dataset
```
```shell
python scripts/test_pcct.py --config-file ./configs/pcct_livemouse_unet2.py # for live mouse dataset
```
```shell
python scripts/test_pcct.py --config-file ./configs/pcct_leg_unet2.py # for chicken leg dataset
```
```shell
python scripts/test_pcct.py --config-file ./configs/pcct_diedmouse_unet2.py # for died mouse dataset
```

## Citation

```shell
@inproceedings{noise2sim2021,
  title={Noise2Sim – Similarity-based Self-Learning for Image Denoising},
  author={Niu, Chuang and Wang, Ge},
  booktitle={arXiv:2011.03384},
  year={2020}
}
```
