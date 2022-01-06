import numpy as np
import torch.utils.data as data
import torch


class LDCTNPY(data.Dataset):
    """

    """

    def __init__(self, data_file='', target_file='', **kwargs):

        self.data = np.load(data_file, allow_pickle=True)
        self.target = np.load(target_file, allow_pickle=True)
        self.total_images = self.data.shape[0]

    def __getitem__(self, index):
        img_noise = self.data[index]
        img_target = self.target[index]

        img_noise = torch.from_numpy(img_noise).to(torch.float32)
        img_target = torch.from_numpy(img_target).to(torch.float32)

        return img_noise, img_target, -1, index

    def __len__(self):
        return self.total_images
