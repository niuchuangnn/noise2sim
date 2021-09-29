# Suppression of Independent and Correlated Noise with Similarity-based Unsupervised Deep Learning
Under review. (**In Updating**)

[comment]: <> (<tr>)

[comment]: <> (<td><img  height="360" src="./figs/training_samples.png"></td>)

[comment]: <> (</tr>)

## Introduction
This is a general similarity-based unsupervised deep denoising approach to suppress not only independent but also correlated image noise.
Theoretical analysis proves the equivalent effectiveness of this unsupervised approach to the supervised counterpart.
This project shows several application cases, including denoising natural, microscopic, low-dose CT and photon-counting micro-CT images.
This general approach can be applied to many other fields by constructing the similar training samples based on the domain-specific prior.


## Installation
Assuming [Anaconda](https://www.anaconda.com/) with python 3.6, the required packages for this project can be installed as:
```shell script
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install faiss-gpu cudatoolkit=10.0 -c pytorch
conda install matplotlib
conda install -c conda-forge python-lmdb tqdm imageio addict tensorboard opencv
```
Then, clone this repo
```shell script
git clone https://github.com/niuchuangnn/noise2sim.git
cd noise2sim
```

## Run
To train the model, simply run the following commands.

Download BSD68 test dataset

```shell script
python ./tools/download_bsd68_noise2void.py
```
Prepare dataset:
```shell script
python ./tools/prepare_bsd400_lmdb.py
```

Run on 1 GPU:
```shell script
python ./tools/train_dist.py --config-file ./configs/bsd400_unet2_ps3_ns8_gpu1.py
```
Run on 8 GPUs:
```shell script
python ./tools/train_dist.py --config-file ./configs/bsd400_unet2_ps3_ns8_gpu8.py
```
The results in paper were obtained using 8 GPUs, you can obtain similar results with 1 GPU.

## TODO
More experiments.

## Citation

```shell
@inproceedings{noise2sim2021,
  title={Noise2Sim – Similarity-based Self-Learning for Image Denoising},
  author={Niu, Chuang and Wang, Ge},
  booktitle={arXiv:2011.03384},
  year={2020}
}
```
