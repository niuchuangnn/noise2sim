import numpy as np
import torch.utils.data as data
import torch
import lmdb
import pickle
import cv2


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

    def __init__(self, lmdb_file, meta_info_file, crop_size, target_type, random_flip=False,
                 prune_dataset=None, num_sim=32, num_select=None, norm_value=255.0, dtype="uint8", **kwargs):

        self.keys, self.shapes = _get_keys_shapes_pickle(meta_info_file)
        if prune_dataset is not None:
            self.keys = self.keys[0:prune_dataset]
            self.shapes = self.shapes[0:prune_dataset]

        self.lmdb_file = lmdb_file
        self.data_env = None

        self.crop_size = crop_size
        self.random_flip = random_flip
        self.target_type = target_type
        self.num_sim = num_sim
        if num_sim is not None:
            self.num_select = num_select
        else:
            self.num_select = num_sim
        self.total_images = len(self.keys)
        self.norm_value = norm_value
        if dtype == "uint8":
            self.dtype = np.uint8
        elif dtype == "float32":
            self.dtype = np.float32
        else:
            raise TypeError

    def random_crop_img2(self, img1, img2):
        y = np.random.randint(img1.shape[0] - self.crop_size + 1)
        x = np.random.randint(img1.shape[1] - self.crop_size + 1)
        img1 = img1[y: y + self.crop_size, x: x + self.crop_size, :]
        img2 = img2[y: y + self.crop_size, x: x + self.crop_size, :]
        return img1, img2

    def __getitem__(self, index):
        """

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

            return img_noise, img_target, -1, index

        elif self.target_type in ["noise-sim"]:
            img_shape = [int(s) for s in self.shapes[index].split('_')]
            img_noise = (_read_img_noise_lmdb(self.data_env, self.keys[index], img_shape, self.dtype) / self.norm_value).astype(np.float32)
            img_noise_sim = (_read_img_noise_sim_lmdb(self.data_env, self.keys[index], img_shape, self.num_sim, self.num_select, self.dtype) / self.norm_value).astype(np.float32)

            H, W, C = img_noise.shape

            img_noise_sim = np.concatenate([img_noise.reshape(1, H, W, C), img_noise_sim], axis=0)

            img_noise_sim_2d = img_noise_sim.reshape([img_noise_sim.shape[0], H*W*C])

            idx_rand1 = np.random.randint(0, img_noise_sim_2d.shape[0], (H*W*C,))
            idx_rand2 = np.random.randint(0, img_noise_sim_2d.shape[0], (H*W*C,))

            idx_fix = np.arange(H*W*C)

            img_noise_1 = img_noise_sim_2d[idx_rand1, idx_fix].reshape(H, W, C)
            img_noise_2 = img_noise_sim_2d[idx_rand2, idx_fix].reshape(H, W, C)

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

            return img_noise_1, img_noise_2, -1, index

        else:
            raise TypeError

    def __len__(self):
        return self.total_images
