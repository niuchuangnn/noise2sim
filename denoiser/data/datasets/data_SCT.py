import numpy as np
import torch.utils.data as data
import torch
import random
from pydicom import dcmread
import torch.nn.functional as F
import mat73


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


def read_files(data_file):
    data_files = []
    fid = open(data_file, 'r')
    lines = fid.readlines()
    for l in lines:
        # file_l = l.split()
        # file_l[1] = file_l[1][0:-1]
        data_files.append(l[0:-1])
    return data_files


def read_dicom(fpath):
    with open(fpath, 'rb') as infile:
        ds = dcmread(infile)
    data = ds.pixel_array
    return data


def read_mat(fpath):
    data = mat73.loadmat(fpath)
    data = data['rec_spirSIRT']
    return data


class SCT(data.Dataset):
    """

    """

    def __init__(self, data_file, crop_size, neighbor=2, center_crop=None, random_flip=False, target_type='noise-sim',
                 range=[-300, 250], ks=7, th=None, **kwargs):
        self.crop_size = crop_size
        self.random_flip = random_flip
        # self.data_files = read_files(data_file)
        self.data = read_mat(data_file)[:, :, 50:600, :]
        if center_crop is not None:
            self.data = self.data[center_crop[0]:center_crop[1], center_crop[2]:center_crop[3], ...]

        self.data = np.clip(self.data, 0, 1)
        self.data[:, :, :, 0] = (self.data[:, :, :, 0] - 0.020572) / 0.020572 * 1000
        self.data[:, :, :, 1] = (self.data[:, :, :, 1] - 0.020954) / 0.020954 * 1000
        self.data[:, :, :, 2] = (self.data[:, :, :, 2] - 0.019928) / 0.019928 * 1000
        self.data[:, :, :, 3] = (self.data[:, :, :, 3] - 0.022309) / 0.022309 * 1000
        self.data[:, :, :, 4] = (self.data[:, :, :, 4] - 0.019151) / 0.019151 * 1000
        self.data = np.clip(self.data, range[0], range[1])
        self.num_slices = self.data.shape[2]

        self.target_type = target_type
        self.range = range
        self.ks = ks
        self.th = th
        self.neighbor = neighbor

        self.kernel = torch.ones(1, 1, ks, ks).to(torch.float32)

    def random_crop(self, img_noise_sim, img_noise, img_clean):

        y = np.random.randint(img_noise.shape[0] - self.crop_size + 1)
        x = np.random.randint(img_noise.shape[1] - self.crop_size + 1)
        img_noise = img_noise[y: y + self.crop_size, x: x + self.crop_size, :]
        if img_clean is not None:
            img_clean = img_clean[y: y + self.crop_size, x: x + self.crop_size, :]
        img_noise_sim = img_noise_sim[:, y: y + self.crop_size, x: x + self.crop_size, :]

        return img_noise_sim, img_noise, img_clean

    def random_crop_img2(self, img_list):
        y = torch.randint(0, img_list[0].shape[1] - self.crop_size + 1, (1,))
        x = torch.randint(0, img_list[0].shape[2] - self.crop_size + 1, (1,))
        img_out = []
        for img in img_list:
            img_out.append(img[:, y: y + self.crop_size, x: x + self.crop_size])
        return img_out

    def random_crop_img(self, img1):
        y = np.random.randint(img1.shape[0] - self.crop_size + 1)
        x = np.random.randint(img1.shape[1] - self.crop_size + 1)
        img1 = img1[y: y + self.crop_size, x: x + self.crop_size, :]
        return img1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image_noise, image_target) where image_target can be clean or noisy image.
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

        # print(ln)

        if self.target_type in ["noise-clean"]:
            img_noise_1 = self.data[:, :, index, :].transpose([2, 0, 1])
            img_noise_2 = self.data[:, :, index + ln, :].transpose([2, 0, 1])

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
            # channel = torch.randint(0, self.data.shape[3], (1,))
            img_noise_1 = self.data[:, :, index, :].transpose([2, 0, 1])
            img_noise_2 = self.data[:, :, index+ln, :].transpose([2, 0, 1])

            img_noise_1 = torch.from_numpy(img_noise_1).to(torch.float32)
            img_noise_2 = torch.from_numpy(img_noise_2).to(torch.float32)

            if self.th is not None:
                data1 = torch.zeros_like(img_noise_1)
                data2 = torch.zeros_like(img_noise_2)
                # data1 = torch.from_numpy(img_noise_1).to(torch.float32)
                for c in range(data1.shape[0]):
                    data1_c = F.conv2d(img_noise_1[c].unsqueeze(0).unsqueeze(0), self.kernel, None, 1, (self.ks-1)//2, 1, 1) / (self.ks * self.ks)
                    data1[c, ...] = data1_c.squeeze()

                    data2_c = F.conv2d(img_noise_2[c].unsqueeze(0).unsqueeze(0), self.kernel, None, 1, (self.ks-1)//2, 1, 1) / (self.ks * self.ks)
                    data2[c, ...] = data2_c.squeeze()

                img_ignore = torch.sqrt((torch.abs(data1 - data2)**2).sum(dim=0)) > self.th
                img_ignore = img_ignore.to(torch.float32)

                img_ignore = img_ignore.reshape([1, img_ignore.shape[0], img_ignore.shape[1]])

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
                img_noise_1 = random_rotate_mirror(img_noise_1, random_mode)
                img_noise_2 = random_rotate_mirror(img_noise_2, random_mode)
                if self.th is not None:
                    img_ignore = random_rotate_mirror(img_ignore, random_mode)

            return img_noise_1, img_noise_2, img_ignore, index

        else:
            raise TypeError

    def __len__(self):
        return self.num_slices