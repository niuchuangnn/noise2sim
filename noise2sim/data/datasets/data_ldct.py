import numpy as np
import torch.utils.data as data
import torch
import cv2
import random
from pydicom import dcmread
import torch.nn.functional as F


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


def read_files(data_file):
    data_files = []
    fid = open(data_file, 'r')
    lines = fid.readlines()
    for l in lines:
        file_l = l.split()
        data_files.append(file_l)
    return data_files


def read_dicom(fpath):
    with open(fpath, 'rb') as infile:
        ds = dcmread(infile)
    data = ds.pixel_array
    return data


class LDCT(data.Dataset):
    """

    """

    def __init__(self, data_file, crop_size, neighbor=1, random_flip=False, target_type='noise-sim',
                 hu_range=None, ks=7, th=None, data_type='dcm', **kwargs):
        if hu_range is None:
            hu_range = [-160, 240]
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.data_files = read_files(data_file)
        self.target_type = target_type
        self.range = hu_range
        self.ks = ks
        self.th = th
        self.neighbor = neighbor
        self.kernel = torch.ones(1, 1, ks, ks).to(torch.float32)
        self.data_type = data_type

    def random_crop(self, img_noise_sim, img_noise, img_clean):

        y = np.random.randint(img_noise.shape[0] - self.crop_size + 1)
        x = np.random.randint(img_noise.shape[1] - self.crop_size + 1)
        img_noise = img_noise[y: y + self.crop_size, x: x + self.crop_size, :]
        if img_clean is not None:
            img_clean = img_clean[y: y + self.crop_size, x: x + self.crop_size, :]
        img_noise_sim = img_noise_sim[:, y: y + self.crop_size, x: x + self.crop_size, :]

        return img_noise_sim, img_noise, img_clean

    def random_crop_img2(self, img_list):
        y = np.random.randint(img_list[0].shape[0] - self.crop_size + 1)
        x = np.random.randint(img_list[0].shape[1] - self.crop_size + 1)
        img_out = []
        for img in img_list:
            img_out.append(img[y: y + self.crop_size, x: x + self.crop_size, ...])
        return img_out

    def random_crop_img(self, img1):
        y = np.random.randint(img1.shape[0] - self.crop_size + 1)
        x = np.random.randint(img1.shape[1] - self.crop_size + 1)
        img1 = img1[y: y + self.crop_size, x: x + self.crop_size, :]
        return img1

    def __getitem__(self, index):
        """

        """

        if self.target_type in ["noise-clean"]:
            path_ldct_1, path_ldct_2, label, _, _ = self.data_files[index]

            img_noise_1 = read_dicom(path_ldct_1)
            img_noise_2 = read_dicom(path_ldct_2)

            img_noise_1 = img_noise_1.astype(np.float32)
            img_noise_2 = img_noise_2.astype(np.float32)

            img_noise_1 = np.clip(img_noise_1 - 1024, self.range[0], self.range[1])
            img_noise_2 = np.clip(img_noise_2 - 1024, self.range[0], self.range[1])

            img_noise_1 = (img_noise_1 - self.range[0]) / (self.range[1] - self.range[0])
            img_noise_2 = (img_noise_2 - self.range[0]) / (self.range[1] - self.range[0])

            if self.crop_size is not None:
                img_noise_1, img_noise_2 = self.random_crop_img2([img_noise_1, img_noise_2])

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

        elif self.target_type in ["noise-sim"]:
            path_ldct_1, _, label, idx_min, idx_max = self.data_files[index]
            img_noise_1 = read_dicom(path_ldct_1)
            idx_min = int(idx_min)
            idx_max = int(idx_max)

            if label == '0':
                lr = 1
            elif label == '-1':
                lr = -1
            else:
                if np.random.rand() > 0.5:
                    lr = 1
                else:
                    lr = -1

            if lr == 1:
                neightbor_max = min(max(1, idx_max - index), self.neighbor)
                ln = random.randint(1, neightbor_max)
            else:
                neightbor_max = min(max(1, index - idx_min), self.neighbor)
                ln = -random.randint(1, neightbor_max)

            path_ldct_2, _, _, _, _ = self.data_files[index + ln]

            img_noise_2 = read_dicom(path_ldct_2)
            img_noise_1 = img_noise_1.astype(np.float32)
            img_noise_2 = img_noise_2.astype(np.float32)

            img_noise_1 = np.clip(img_noise_1 - 1024, self.range[0], self.range[1])
            img_noise_2 = np.clip(img_noise_2 - 1024, self.range[0], self.range[1])

            if self.crop_size is not None:
                img_noise_1, img_noise_2 = self.random_crop_img2([img_noise_1, img_noise_2])

            if self.th is not None:
                data1 = torch.from_numpy(img_noise_1).to(torch.float32)
                data1 = F.conv2d(data1.unsqueeze(0).unsqueeze(0), self.kernel, None, 1, (self.ks-1)//2, 1, 1) / (self.ks * self.ks)
                data1 = data1.squeeze()

                data2 = torch.from_numpy(img_noise_2).to(torch.float32)
                data2 = F.conv2d(data2.unsqueeze(0).unsqueeze(0), self.kernel, None, 1, (self.ks-1)//2, 1, 1) / (self.ks * self.ks)
                data2 = data2.squeeze()

                img_ignore = torch.abs(data1 - data2) > self.th
                img_ignore = img_ignore.to(torch.float32).numpy()
            else:
                img_ignore = None

            img_noise_1 = (img_noise_1 - self.range[0]) / (self.range[1] - self.range[0])
            img_noise_2 = (img_noise_2 - self.range[0]) / (self.range[1] - self.range[0])

            if self.crop_size is not None:
                if self.th is not None:
                    img_noise_1, img_noise_2, img_ignore = self.random_crop_img2([img_noise_1, img_noise_2, img_ignore])
                else:
                    img_noise_1, img_noise_2 = self.random_crop_img2([img_noise_1, img_noise_2])

            if self.random_flip:
                random_mode = np.random.randint(0, 8)
                img_noise_1 = random_rotate_mirror(img_noise_1.squeeze(), random_mode)
                img_noise_2 = random_rotate_mirror(img_noise_2.squeeze(), random_mode)
                if self.th is not None:
                    img_ignore = random_rotate_mirror(img_ignore.squeeze(), random_mode)

            if len(img_noise_1.shape) < 3:
                img_noise_1 = img_noise_1.reshape([img_noise_1.shape[0], img_noise_1.shape[1], 1])

            if len(img_noise_2.shape) < 3:
                img_noise_2 = img_noise_2.reshape([img_noise_2.shape[0], img_noise_2.shape[1], 1])

            if self.th is not None:
                if len(img_ignore.shape) < 3:
                    img_ignore = img_ignore.reshape([img_noise_2.shape[0], img_noise_2.shape[1], 1])

            img_noise_1 = torch.from_numpy(img_noise_1.transpose([2, 0, 1]).copy()).to(torch.float32)
            img_noise_2 = torch.from_numpy(img_noise_2.transpose([2, 0, 1]).copy()).to(torch.float32)
            if self.th is not None:
                img_ignore = torch.from_numpy(img_ignore.transpose([2, 0, 1]).copy()).to(torch.float32)
            else:
                img_ignore = -1

            return img_noise_1, img_noise_2, img_ignore, index

        else:
            raise TypeError

    def __len__(self):
        return len(self.data_files)
