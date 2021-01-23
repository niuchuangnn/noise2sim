import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import lmdb
import pickle
import cv2
import random


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


def random_rotate_mirror(img_0, random_mode):
    if random_mode == 0:
        img = img_0
    elif random_mode == 1:
        img = img_0[::-1, ...]
    elif random_mode == 2:
        img = cv2.rotate(img_0, cv2.ROTATE_90_CLOCKWISE)
    elif random_mode == 3:
        img_90 = cv2.rotate(img_0, cv2.ROTATE_90_CLOCKWISE)
        img = img_90[:, ::-1, ...]
    elif random_mode == 4:
        img = cv2.rotate(img_0, cv2.ROTATE_180)
    elif random_mode == 5:
        img_180 = cv2.rotate(img_0, cv2.ROTATE_180)
        img = img_180[::-1, ...]
    elif random_mode == 6:
        img = cv2.rotate(img_0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif random_mode == 7:
        img_270 = cv2.rotate(img_0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = img_270[:, ::-1, ...]
    else:
        raise TypeError
    return img


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


def _read_img_lmdb(env, key, shape, dtype=np.uint8):
    with env.begin(write=False) as txn:
        buf = txn.get("{}".format(key).encode('ascii'))
    img_noise_flat = np.frombuffer(buf, dtype=dtype)
    H, W, C = shape
    img_noise = img_noise_flat.reshape(H, W, C)
    return img_noise


def _read_img_noise_lmdb(env, key, shape, dtype=np.uint8):
    with env.begin(write=False) as txn:
        buf = txn.get("{}_noise".format(key).encode('ascii'))
    img_noise_flat = np.frombuffer(buf, dtype=dtype)
    H, W, C = shape
    img_noise = img_noise_flat.reshape(H, W, C)
    return img_noise


def _read_img_noise2_lmdb(env, key, shape, dtype=np.uint8):
    with env.begin(write=False) as txn:
        buf = txn.get("{}_noise2".format(key).encode('ascii'))
    img_noise_flat = np.frombuffer(buf, dtype=dtype)
    H, W, C = shape
    img_noise = img_noise_flat.reshape(H, W, C)
    return img_noise


def _read_img_noise_sim_lmdb(env, key, shape, num_sim, num_select, dtype=np.uint8):
    with env.begin(write=False) as txn:
        buf = txn.get("{}_noise_sim".format(key).encode('ascii'))
    data_flat = np.frombuffer(buf, dtype=dtype)
    H, W, C = shape
    img_noise_sim = data_flat.reshape(num_sim, H, W, C)
    img_noise_sim = img_noise_sim[0:num_select, ...]
    return img_noise_sim


def _read_patches_sim_lmdb(env, key, shape, num_patches_per_img, num_select, dtype=np.uint8):
    with env.begin(write=False) as txn:
        buf = txn.get("{}_noise_sim".format(key).encode('ascii'))
    data_flat = np.frombuffer(buf, dtype=dtype)
    num_patches, num_sim, H, W, C = shape
    patches_noise_sim = data_flat.reshape(num_patches, num_sim, H, W, C)

    idx_patches = np.arange(num_patches)
    np.random.shuffle(idx_patches)
    idx_patches = idx_patches[0:num_patches_per_img]
    patches_noise_sim = patches_noise_sim[idx_patches, :, :, :, :]

    patches_noise_sim = patches_noise_sim[:, 0:num_select, ...]
    return patches_noise_sim


class LMDB(data.Dataset):
    """

    """

    def __init__(self, lmdb_file, meta_info_file, crop_size, target_type, random_flip=False, num_max_patch=128,
                 prune_dataset=None, patch_size=5, num_sim=32, num_select=None, std=25.0, norm_value=255.0,
                 load_data_all=False, incorporate_noise=False, dtype="uint8", num_patches_per_img=1024,
                 ps_th=None, **kwargs):

        self.keys, self.shapes = _get_keys_shapes_pickle(meta_info_file)
        if prune_dataset is not None:
            self.keys = self.keys[0:prune_dataset]
            self.shapes = self.shapes[0:prune_dataset]

        self.lmdb_file = lmdb_file
        self.data_env = None

        self.crop_size = crop_size
        self.random_flip = random_flip
        self.target_type = target_type
        self.patch_size = patch_size
        self.num_sim = num_sim
        self.num_patches_per_img = num_patches_per_img
        if num_sim is not None:
            self.num_select = num_select
        else:
            self.num_select = num_sim
        self.load_data_all = load_data_all
        self.incorporate_noise = incorporate_noise
        self.total_images = len(self.keys)
        self.std = std
        self.norm_value = norm_value
        self.num_max_patch = num_max_patch
        if dtype == "uint8":
            self.dtype = np.uint8
        elif dtype == "float32":
            self.dtype = np.float32
        else:
            raise TypeError

        self.ps_th = ps_th

    def random_crop(self, img_noise_sim, img_noise, img_clean):

        y = np.random.randint(img_noise.shape[0] - self.crop_size + 1)
        x = np.random.randint(img_noise.shape[1] - self.crop_size + 1)
        img_noise = img_noise[y: y + self.crop_size, x: x + self.crop_size, :]
        if img_clean is not None:
            img_clean = img_clean[y: y + self.crop_size, x: x + self.crop_size, :]
        img_noise_sim = img_noise_sim[:, y: y + self.crop_size, x: x + self.crop_size, :]

        return img_noise_sim, img_noise, img_clean

    def random_crop_img2(self, img1, img2):
        y = np.random.randint(img1.shape[0] - self.crop_size + 1)
        x = np.random.randint(img1.shape[1] - self.crop_size + 1)
        img1 = img1[y: y + self.crop_size, x: x + self.crop_size, :]
        img2 = img2[y: y + self.crop_size, x: x + self.crop_size, :]
        return img1, img2

    def filtered_random_crop_img2(self, img1_ori, img2_ori):
        cropping = True
        while cropping:
            img1, img2 = self.random_crop_img2(img1_ori.copy(), img2_ori.copy())
            # plt.figure()
            # plt.imshow(img1)
            # plt.title(np.std(img1*self.norm_value))
            # plt.show()
            if np.std(img1*self.norm_value) > self.ps_th and np.std(img2*self.norm_value) > self.ps_th:
                cropping = False

        return img1, img2

    def noisify(self, img):
        if self.noise_type == "gaussian":
            std = self.std / 255.0
            img_noise = img + np.random.normal(size=img.shape) * std
        else:
            raise TypeError
        return img_noise

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image_noise, image_target) where image_target can be clean or noisy image.
        """
        if self.data_env is None:
            self.data_env = lmdb.open(self.lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)

        if self.target_type in ["noise-clean", "noise-noise"]:
            img_shape = [int(s) for s in self.shapes[index].split('_')]
            img_noise = (_read_img_noise_lmdb(self.data_env, self.keys[index], img_shape, self.dtype) / self.norm_value).astype(np.float32)

            if self.target_type in ["noise-clean"]:
                img_target = (_read_img_lmdb(self.data_env, self.keys[index], img_shape, self.dtype) / self.norm_value).astype(np.float32)
            elif self.target_type in ["noise-noise"]:
                img_target = (_read_img_noise2_lmdb(self.data_env, self.keys[index], img_shape, self.dtype) / self.norm_value).astype(np.float32)
            else:
                raise TypeError

            if self.crop_size is not None:
                img_noise, img_target = self.random_crop_img2(img_noise, img_target)

            if self.random_flip:
                random_mode = np.random.randint(0, 8)
                img_noise = random_rotate_mirror(img_noise.squeeze(), random_mode)
                img_target = random_rotate_mirror(img_target.squeeze(), random_mode)

            if len(img_noise.shape) < 3:
                img_noise = img_noise.reshape([img_noise.shape[0], img_noise.shape[1], 1])

            if len(img_target.shape) < 3:
                img_target = img_target.reshape([img_target.shape[0], img_target.shape[1], 1])

            img_noise = torch.from_numpy(img_noise.transpose([2, 0, 1]).copy()).to(torch.float32)
            img_target = torch.from_numpy(img_target.transpose([2, 0, 1]).copy()).to(torch.float32)

            return img_noise, img_target, self.std, index

        elif self.target_type in ["random_noise-mapping"]:
            img_shape = [int(s) for s in self.shapes[index].split('_')]
            img_noise = (_read_img_noise_lmdb(self.data_env, self.keys[index], img_shape, self.dtype) / self.norm_value).astype(np.float32)
            img_noise_sim = (_read_img_noise_sim_lmdb(self.data_env, self.keys[index], img_shape, self.num_sim, self.num_select, self.dtype) / self.norm_value).astype(np.float32)

            # Update
            H, W, C = img_noise.shape
            if self.incorporate_noise:
                img_noise_sim = np.concatenate([img_noise.reshape(1, H, W, C), img_noise_sim], axis=0)

            img_noise_sim_2d = img_noise_sim.reshape([img_noise_sim.shape[0], H*W*C])

            idx_rand1 = np.random.randint(0, img_noise_sim_2d.shape[0], (H*W*C,))
            idx_rand2 = np.random.randint(0, img_noise_sim_2d.shape[0], (H*W*C,))

            idx_fix = np.arange(H*W*C)

            img_noise_1 = img_noise_sim_2d[idx_rand1, idx_fix].reshape(H, W, C)
            img_noise_2 = img_noise_sim_2d[idx_rand2, idx_fix].reshape(H, W, C)

            # plt.figure()
            # plt.imshow(img_noise_1.squeeze())
            # plt.figure()
            # plt.imshow(img_noise_2.squeeze())
            # plt.figure()
            # plt.imshow(img_noise.squeeze())
            # plt.show()

            if self.crop_size is not None:
                if self.ps_th is not None:
                    img_noise_1, img_noise_2 = self.filtered_random_crop_img2(img_noise_1, img_noise_2)
                else:
                    img_noise_1, img_noise_2 = self.random_crop_img2(img_noise_1, img_noise_2)

            if self.random_flip:
                random_mode = np.random.randint(0, 8)
                img_noise_1 = random_rotate_mirror(img_noise_1.squeeze(), random_mode)
                img_noise_2 = random_rotate_mirror(img_noise_2.squeeze(), random_mode)

            if len(img_noise_1.shape) < 3:
                img_noise_1 = img_noise_1.reshape([img_noise_1.shape[0], img_noise_1.shape[1], 1])

            if len(img_noise_2.shape) < 3:
                img_noise_2 = img_noise_2.reshape([img_noise_2.shape[0], img_noise_2.shape[1], 1])

            img_noise_1 = torch.from_numpy(img_noise_1.transpose([2, 0, 1]).copy()).to(torch.float32)
            img_noise_2 = torch.from_numpy(img_noise_2.transpose([2, 0, 1]).copy()).to(torch.float32)

            return img_noise_1, img_noise_2, self.std, index

        elif self.target_type in ["fix_noise-mapping"]:
            img_shape = [int(s) for s in self.shapes[index].split('_')]
            img_noise = (_read_img_noise_lmdb(self.data_env, self.keys[index], img_shape, self.dtype) / self.norm_value).astype(np.float32)
            img_noise_sim = (_read_img_noise_sim_lmdb(self.data_env, self.keys[index], img_shape, self.num_sim, self.num_select, self.dtype) / self.norm_value).astype(np.float32)

            # Update
            H, W, C = img_noise.shape
            if self.incorporate_noise:
                img_noise_sim = np.concatenate([img_noise.reshape(1, H, W, C), img_noise_sim], axis=0)

            num_sim = img_noise_sim.shape[0]

            idxs = list(range(num_sim))
            random.shuffle(idxs)
            idx_rand1 = idxs[0]
            idx_rand2 = idxs[1]

            img_noise_1 = img_noise_sim[idx_rand1, ...]
            img_noise_2 = img_noise_sim[idx_rand2, ...]

            # plt.figure()
            # plt.imshow(img_noise_1.squeeze())
            # plt.figure()
            # plt.imshow(img_noise_2.squeeze())
            # plt.figure()
            # plt.imshow(img_noise.squeeze())
            # plt.show()

            if self.crop_size is not None:
                img_noise_1, img_noise_2 = self.random_crop_img2(img_noise_1, img_noise_2)

            if self.random_flip:
                random_mode = np.random.randint(0, 8)
                img_noise_1 = random_rotate_mirror(img_noise_1.squeeze(), random_mode)
                img_noise_2 = random_rotate_mirror(img_noise_2.squeeze(), random_mode)

            if len(img_noise_1.shape) < 3:
                img_noise_1 = img_noise_1.reshape([img_noise_1.shape[0], img_noise_1.shape[1], 1])

            if len(img_noise_2.shape) < 3:
                img_noise_2 = img_noise_2.reshape([img_noise_2.shape[0], img_noise_2.shape[1], 1])

            img_noise_1 = torch.from_numpy(img_noise_1.transpose([2, 0, 1]).copy()).to(torch.float32)
            img_noise_2 = torch.from_numpy(img_noise_2.transpose([2, 0, 1]).copy()).to(torch.float32)

            return img_noise_1, img_noise_2, self.std, index

        elif self.target_type in ["noise-similarity", "similarity-noise"]:
            img_shape = [int(s) for s in self.shapes[index].split('_')]
            img_noise = (_read_img_noise_lmdb(self.data_env, self.keys[index], img_shape, self.dtype) / self.norm_value).astype(np.float32)
            img_noise_sim = (_read_img_noise_sim_lmdb(self.data_env, self.keys[index], img_shape, self.num_sim, self.num_select, self.dtype) / self.norm_value).astype(np.float32)

            # Update
            H, W, C = img_noise.shape
            if self.incorporate_noise:
                img_noise_sim = np.concatenate([img_noise.reshape(1, H, W, C), img_noise_sim], axis=0)

            img_noise_sim_2d = img_noise_sim.reshape([img_noise_sim.shape[0], H*W*C])
            idx_rand1 = np.random.randint(0, img_noise_sim_2d.shape[0], (H*W*C,))
            idx_fix = np.arange(H*W*C)
            img_noise_1 = img_noise_sim_2d[idx_rand1, idx_fix].reshape(H, W, C)

            # plt.figure()
            # plt.imshow(img_noise_1.squeeze())
            # plt.figure()
            # plt.imshow(img_noise.squeeze())
            # plt.show()

            if self.crop_size is not None:
                img_noise_1, img_noise = self.random_crop_img2(img_noise_1, img_noise)

            if self.random_flip:
                random_mode = np.random.randint(0, 8)
                img_noise_1 = random_rotate_mirror(img_noise_1.squeeze(), random_mode)
                img_noise = random_rotate_mirror(img_noise.squeeze(), random_mode)

            if len(img_noise_1.shape) < 3:
                img_noise_1 = img_noise_1.reshape([img_noise_1.shape[0], img_noise_1.shape[1], 1])

            if len(img_noise.shape) < 3:
                img_noise = img_noise.reshape([img_noise.shape[0], img_noise.shape[1], 1])

            img_noise_1 = torch.from_numpy(img_noise_1.transpose([2, 0, 1]).copy()).to(torch.float32)
            img_noise = torch.from_numpy(img_noise.transpose([2, 0, 1]).copy()).to(torch.float32)

            if self.target_type == "noise-similarity":
                return img_noise, img_noise_1, self.std, index
            elif self.target_type == "similarity-noise":
                return img_noise_1, img_noise, self.std, index
            else:
                raise TypeError

        elif self.target_type in ["noise-adjacent"]:
            img_shape = [int(s) for s in self.shapes[index].split('_')]
            img_noise = (_read_img_noise_lmdb(self.data_env, self.keys[index], img_shape, self.dtype) / self.norm_value).astype(np.float32)

            if index == 0:
                index_adj = index+1
            elif index == self.total_images - 1:
                index_adj = index-1
            else:
                if np.random.rand() > 0.5:
                    index_adj = index+1
                else:
                    index_adj = index-1

            img_noise_adj = (_read_img_noise_lmdb(self.data_env, self.keys[index_adj], img_shape, self.dtype) / self.norm_value).astype(np.float32)

            # plt.figure()
            # plt.imshow(img_noise_adj.squeeze())
            # plt.figure()
            # plt.imshow(img_noise.squeeze())
            # plt.show()

            if self.crop_size is not None:
                img_noise, img_noise_adj = self.random_crop_img2(img_noise, img_noise_adj)

            if self.random_flip:
                random_mode = np.random.randint(0, 8)
                img_noise_adj = random_rotate_mirror(img_noise_adj.squeeze(), random_mode)
                img_noise = random_rotate_mirror(img_noise.squeeze(), random_mode)

            if len(img_noise_adj.shape) < 3:
                img_noise_adj = img_noise_adj.reshape([img_noise_adj.shape[0], img_noise_adj.shape[1], 1])

            if len(img_noise.shape) < 3:
                img_noise = img_noise.reshape([img_noise.shape[0], img_noise.shape[1], 1])

            img_noise_adj = torch.from_numpy(img_noise_adj.transpose([2, 0, 1]).copy()).to(torch.float32)
            img_noise = torch.from_numpy(img_noise.transpose([2, 0, 1]).copy()).to(torch.float32)

            return img_noise, img_noise_adj, self.std, index

        elif self.target_type in ["patch-mapping"]:
            img_shape = [int(s) for s in self.shapes[index].split('_')]
            patches = (_read_patches_sim_lmdb(self.data_env, self.keys[index], img_shape,
                                                self.num_patches_per_img, self.num_select, self.dtype) / self.norm_value).astype(np.float32)
            num_patches_per_img, num_select, H, W, C = patches.shape

            patches1 = np.zeros([num_patches_per_img, H, W, C]).astype(np.float32)
            patches2 = np.zeros([num_patches_per_img, H, W, C]).astype(np.float32)

            for i in range(num_patches_per_img):
                idx_select = np.arange(num_select)
                np.random.shuffle(idx_select)
                patches1[i, :, :, :] = patches[i, idx_select[0], :, :, :]
                patches2[i, :, :, :] = patches[i, idx_select[1], :, :, :]

                if self.random_flip:
                    random_mode = np.random.randint(0, 8)
                    patches1[i, ...] = random_rotate_mirror(patches1[i, :, :, :].squeeze(), random_mode).reshape(H, W, C)
                    patches2[i, ...] = random_rotate_mirror(patches2[i, :, :, :].squeeze(), random_mode).reshape(H, W, C)

            # plt.figure()
            # plt.imshow(patches1[0].squeeze(), cmap="gray")
            # plt.figure()
            # plt.imshow(patches2[0].squeeze(), cmap="gray")
            # plt.show()

            patches1 = torch.from_numpy(patches1.transpose([0, 3, 1, 2]).copy()).to(torch.float32)
            patches2 = torch.from_numpy(patches2.transpose([0, 3, 1, 2]).copy()).to(torch.float32)

            if self.num_max_patch < num_patches_per_img:
                idx_select = np.arange(num_patches_per_img)
                np.random.shuffle(idx_select)
                idx_select = idx_select[0:self.num_max_patch]
                patches1 = patches1[idx_select, ...]
                patches2 = patches2[idx_select, ...]

            return patches1, patches2, self.std, index

        else:
            raise TypeError

    def __len__(self):
        return self.total_images


if __name__ == "__main__":
    from denoiser.data import build_dataset
    from denoiser.config import Config

    # cfg = Config.fromfile("./configs/imagenet_val/unet_random_noise-mapping.py")
    # cfg = Config.fromfile("./configs/imagenet_val/imagenet_dist_unet2_random_noise-mapping.py")
    # cfg = Config.fromfile("./configs/imagenet_val/bsd400_dist_unet2_fix_noise-mapping.py")
    # cfg = Config.fromfile("./configs/imagenet_val/mars_dist_unet2_random_noise-mapping.py")
    # cfg = Config.fromfile("./configs/ctc_msc/ctc_msc_dist_unet2_random_noise-mapping.py")
    # cfg = Config.fromfile("./configs/ctc_msc/ctc_gowt1_dist_unet2_random_noise-mapping.py")
    # cfg = Config.fromfile("./configs/bsd400_gray/dist_unet2_bsd400_gaussian25_ps7_ns8_sim2sim.py")

    # cfg = Config.fromfile("/media/niuchuang/Storage/ModelResults/Denoising/ecllipse/dist_unet2_noise-clean_lmdb-ecllipse-gaussian25-ps5-ns8-residual-gpu8/config.py")

    cfg = Config.fromfile("./configs/rectangles/rectangles.py")

    dataset = build_dataset(cfg.data_train)
    print(len(dataset))

    for i in range(len(dataset)):
        (img_noise, img_target, std, idx) = dataset[i]
        print(idx)
        if len(img_noise.shape) > 3:
            img_noise = img_noise[0]

        if len(img_target.shape) > 3:
            img_target = img_target[0]

        plt.figure()
        plt.imshow(np.transpose(img_noise.numpy(), [1, 2, 0]).squeeze())
        plt.title(img_noise.max())

        plt.figure()
        plt.imshow(np.transpose(img_target.numpy(), [1, 2, 0]).squeeze())
        plt.title(img_target.max())

        plt.show()
