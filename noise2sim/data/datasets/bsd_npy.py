import numpy as np
import torch.utils.data as data
import torch


class BSDNPY(data.Dataset):
    """

    """

    def __init__(self, data_file, target_file, norm=[0, 255.], **kwargs):

        self.data = np.load(data_file, allow_pickle=True)
        self.target = np.load(target_file, allow_pickle=True)
        self.norm = norm
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
        return img_noise, img_target, -1, index

    def __len__(self):
        return self.total_images
