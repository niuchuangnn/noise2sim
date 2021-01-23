import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
# import faiss
# from tools.nearest_search import search_raw_array_pytorch
import lmdb
import pickle


# Number of channels enforcer while retaining dtype.
def set_color_channels(x, num_channels):
    assert x.shape[0] in [1, 3, 4]
    x = x[:min(x.shape[0], 3)]  # drop possible alpha channel
    if x.shape[0] == num_channels:
        return x
    elif x.shape[0] == 1:
        return np.tile(x, [3, 1, 1])
    y = np.mean(x, axis=0, keepdims=True)
    if np.issubdtype(x.dtype, np.integer):
        y = np.round(y).astype(x.dtype)
    return y


def random_crop_numpy(img, crop_size):
    y = np.random.randint(img.shape[1] - crop_size + 1)
    x = np.random.randint(img.shape[2] - crop_size + 1)
    return img[:, y: y + crop_size, x: x + crop_size]


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def _get_keys_shapes_pickle(meta_info_file):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(meta_info_file, 'rb'))
    keys = meta_info['keys']
    shapes = meta_info['shapes']
    return keys, shapes


def _read_img_noise_lmdb(env, key, shape):
    with env.begin(write=False) as txn:
        buf = txn.get("{}_noise".format(key).encode('ascii'))
    img_noise_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = shape
    img_noise = img_noise_flat.reshape(H, W, C)
    return img_noise


def _read_img_noise_sim_lmdb(env, key, shape, num_select):
    with env.begin(write=False) as txn:
        buf = txn.get("{}_noise_sim".format(key).encode('ascii'))
    data_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = shape
    img_noise_sim = data_flat.reshape(num_select, H, W, C)
    return img_noise_sim


class BSDNPY(data.Dataset):
    """

    """

    def __init__(self, data_file='/media/niuchuang/Storage/DataSets/BSD68_reproducibility_data/test/bsd68_gaussian25.npy',
                 target_file='/media/niuchuang/Storage/DataSets/BSD68_reproducibility_data/test/bsd68_groundtruth.npy',
                 norm=[0, 255.], std=25.0, **kwargs):

        self.data = np.load(data_file, allow_pickle=True)
        self.target = np.load(target_file, allow_pickle=True)
        self.norm = norm
        self.std = std
        self.total_images = self.data.shape[0]

    def __getitem__(self, index):
        img_noise = self.data[index]
        img_target = self.target[index]

        if self.norm is not None:
            img_noise = (img_noise - self.norm[0]) / (self.norm[1] - self.norm[0])
            img_target = (img_target - self.norm[0]) / (self.norm[1] - self.norm[0])

        if len(img_noise.shape) > 2:
            [H, W, C] = img_noise.shape
        else:
            [H, W] = img_noise.shape
            C = 1

        img_noise = torch.from_numpy(img_noise.reshape([C, H, W])).to(torch.float32)
        img_target = torch.from_numpy(img_target.reshape([C, H, W])).to(torch.float32)
        return img_noise, img_target, self.std, index

    def __len__(self):
        return self.total_images


if __name__ == "__main__":
    # data_file = '/media/niuchuang/Storage/DataSets/BSD68_reproducibility_data/test/bsd68_poisson30.npy'
    # data_file = '/media/niuchuang/Storage/DataSets/BSD68_reproducibility_data/train/DCNN400_train_poisson30.npy'
    data_file = "/media/niuchuang/Storage/DataSets/koak/gaussian25.npy"
    target_file = "/media/niuchuang/Storage/DataSets/koak/groundtruth.npy"
    dataset = BSDNPY(data_file=data_file, target_file=target_file)

    for iteration, (img_noise, img_target, std, idx) in enumerate(dataset):

        img_noise = img_noise.numpy()
        img_target = img_target.numpy()

        if len(img_noise.shape) > 2:
            img_noise = img_noise.transpose([1, 2, 0])

        if len(img_target.shape) > 2:
            img_target = img_target.transpose([1, 2, 0])

        plt.figure()
        plt.imshow(img_target.squeeze(), cmap="gray")
        plt.figure()
        plt.imshow(img_noise.squeeze(), cmap="gray")

        plt.show()
