import numpy as np
import torch.utils.data as data
import torch
import random
import torch.nn.functional as F
from scipy.io import loadmat


def random_rotate_mirror(img_0, random_mode):
    if random_mode == 0:
        img = img_0
    elif random_mode == 1:
        img = torch.flip(img_0, [1])
    elif random_mode == 2:
        img = torch.rot90(img_0, 1, [2, 1])
    elif random_mode == 3:
        img_90 = torch.rot90(img_0, 1, [2, 1])
        img = torch.flip(img_90, [2])
    elif random_mode == 4:
        img = torch.rot90(img_0, 2, [2, 1])
    elif random_mode == 5:
        img_180 = torch.rot90(img_0, 2, [1, 2])
        img = torch.flip(img_180, [1])
    elif random_mode == 6:
        img = torch.rot90(img_0, 1, [1, 2])
    elif random_mode == 7:
        img_270 = torch.rot90(img_0, 1, [1, 2])
        img = torch.flip(img_270, [2])
    else:
        raise TypeError
    return img


def read_mat(fpath):
    data = loadmat(fpath)
    data = data['data']
    return data


class PCCT(data.Dataset):
    """

    """

    def __init__(self, data_file, crop_size, neighbor=2, center_crop=None, random_flip=False, target_type='noise-sim',
                 hu_range=None, ks=7, th=None, slice_crop=None, data_file_clean=None, **kwargs):
        if hu_range is None:
            hu_range = [0, 1.5]
        if slice_crop is None:
            slice_crop = [20, 100]
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.data = read_mat(data_file)
        if data_file_clean is not None:
            self.data_clean = read_mat(data_file_clean)
        else:
            self.data_clean = None

        if slice_crop is not None:
            self.data = self.data[:, :, slice_crop[0]:slice_crop[1], :]
            if self.data_clean is not None:
                self.data_clean = self.data_clean[:, :, slice_crop[0]:slice_crop[1], :]

        if center_crop is not None:
            self.data = self.data[center_crop[0]:center_crop[1], center_crop[2]:center_crop[3], ...]
            if self.data_clean is not None:
                self.data_clean = self.data_clean[center_crop[0]:center_crop[1], center_crop[2]:center_crop[3], ...]

        self.data = np.clip(self.data, hu_range[0], hu_range[1])
        self.num_slices = self.data.shape[2]

        self.target_type = target_type
        self.range = hu_range
        self.ks = ks
        self.th = th
        self.neighbor = neighbor
        self.kernel = torch.ones(1, 1, ks, ks).to(torch.float32)

    def random_crop_img2(self, img_list):
        y = torch.randint(0, img_list[0].shape[1] - self.crop_size + 1, (1,))
        x = torch.randint(0, img_list[0].shape[2] - self.crop_size + 1, (1,))
        img_out = []
        for img in img_list:
            img_out.append(img[:, y: y + self.crop_size, x: x + self.crop_size])
        return img_out

    def __getitem__(self, index):
        """

        """

        if index == 0:
            lr = 1
        elif index == self.data.shape[2]-1:
            lr = -1
        else:
            if np.random.rand() > 0.5:
                lr = 1
            else:
                lr = -1

        if lr == 1:
            neightbor_max = min(max(1, self.data.shape[2]-1-index), self.neighbor)
            ln = random.randint(1, neightbor_max)
        else:
            neightbor_max = min(max(1, index-self.neighbor), self.neighbor)
            ln = -random.randint(1, neightbor_max)

        if self.target_type in ["noise-clean"]:
            img_noise_1 = self.data[:, :, index, :].transpose([2, 0, 1])
            img_noise_2 = self.data_clean[:, :, index + ln, :].transpose([2, 0, 1])

            img_noise_1 = torch.from_numpy(img_noise_1).to(torch.float32)
            img_noise_2 = torch.from_numpy(img_noise_2).to(torch.float32)

            img_noise_1 = (img_noise_1 - self.range[0]) / (self.range[1] - self.range[0])
            img_noise_2 = (img_noise_2 - self.range[0]) / (self.range[1] - self.range[0])

            if self.crop_size is not None:
                img_noise_1, img_noise_2 = self.random_crop_img2([img_noise_1, img_noise_2])

            if self.random_flip:
                random_mode = np.random.randint(0, 8)
                img_noise_1 = random_rotate_mirror(img_noise_1, random_mode)
                img_noise_2 = random_rotate_mirror(img_noise_2, random_mode)

            return img_noise_1, img_noise_2, -1, index

        elif self.target_type in ["noise-sim"]:
            img_noise_1 = self.data[:, :, index, :].transpose([2, 0, 1])
            img_noise_2 = self.data[:, :, index+ln, :].transpose([2, 0, 1])

            img_noise_1 = torch.from_numpy(img_noise_1).to(torch.float32)
            img_noise_2 = torch.from_numpy(img_noise_2).to(torch.float32)

            if self.th is not None:
                data1 = torch.zeros_like(img_noise_1)
                data2 = torch.zeros_like(img_noise_2)
                for c in range(data1.shape[0]):
                    data1_c = F.conv2d(img_noise_1[c].unsqueeze(0).unsqueeze(0), self.kernel, None, 1, (self.ks-1)//2, 1, 1) / (self.ks * self.ks)
                    data1[c, ...] = data1_c.squeeze()

                    data2_c = F.conv2d(img_noise_2[c].unsqueeze(0).unsqueeze(0), self.kernel, None, 1, (self.ks-1)//2, 1, 1) / (self.ks * self.ks)
                    data2[c, ...] = data2_c.squeeze()

                img_ignore = torch.sqrt((torch.abs(data1 - data2)**2).sum(dim=0)) > self.th
                img_ignore = img_ignore.to(torch.float32)

                img_ignore = img_ignore.reshape([1, img_ignore.shape[0], img_ignore.shape[1]])

            else:
                img_ignore = -1

            img_noise_1 = (img_noise_1 - self.range[0]) / (self.range[1] - self.range[0])
            img_noise_2 = (img_noise_2 - self.range[0]) / (self.range[1] - self.range[0])

            if self.crop_size is not None:
                if self.th is not None:
                    img_noise_1, img_noise_2, img_ignore = self.random_crop_img2([img_noise_1, img_noise_2, img_ignore])
                else:
                    img_noise_1, img_noise_2 = self.random_crop_img2([img_noise_1, img_noise_2])

            if self.random_flip:
                random_mode = np.random.randint(0, 8)
                img_noise_1 = random_rotate_mirror(img_noise_1, random_mode)
                img_noise_2 = random_rotate_mirror(img_noise_2, random_mode)
                if self.th is not None:
                    img_ignore = random_rotate_mirror(img_ignore, random_mode)

            return img_noise_1, img_noise_2, img_ignore, index

        else:
            raise TypeError

    def __len__(self):
        return self.num_slices