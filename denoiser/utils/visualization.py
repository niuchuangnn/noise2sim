from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import PIL
from PIL import Image
import numpy as np


def show_tensor_imgs(tensor, nrow=8, padding=2, show=True,
               normalize=False, range=None, scale_each=False, pad_value=0):
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    if show:
        plt.figure()
        plt.imshow(ndarr)
        plt.axis("off")
    # plt.show()
    return ndarr


def show_examples(imgs, scores, attention_maps, scores_att):
    num_img = imgs.shape[0]

    for i in range(num_img):
        fig = plt.figure(figsize=(24, 6))
        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)

        img = imgs[i].transpose([1, 2, 0])
        ax1.imshow(img)
        ax1.axis("off")

        x_data = [str(i) for i in range(0, 10)]
        ax2.bar(x=x_data, height=scores[i])
        ax2.axis([-1, 10, 0, 1])
        ax2.tick_params(labelsize=23)

        att = attention_maps[i, 0]
        att = Image.fromarray(np.uint8(att*255))
        att = att.resize((imgs.shape[2], imgs.shape[3]))
        att = np.asarray(att)

        att_mask = np.zeros_like(img)
        att_mask[:, :, 0] = att

        alpha = 0.1
        img_att = np.uint8((1-alpha)*img + alpha*att_mask)

        cmap = plt.get_cmap('jet')
        attMap = att
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2) * 255

        attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img +\
                 (attMap ** 0.7).reshape(attMap.shape + (1,)) * attMapV

        # attMap = (1-alpha)*img + alpha*attMapV

        ax3.imshow(attMap)
        ax3.axis("off")

        ax4.bar(x=x_data, height=scores_att[i])
        ax4.axis([-1, 10, 0, 1])
        ax4.tick_params(labelsize=23)

        plt.tight_layout()
        plt.savefig("{}/{}.png".format("./results/stl10_new/examples/", i))

        # plt.show()


def show_examples3(imgs, scores, attention_maps, save_folder, num_cluster=10):
    num_img = imgs.shape[0]

    for i in range(num_img):
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        img = imgs[i].transpose([1, 2, 0])
        ax1.imshow(img)
        ax1.axis("off")

        x_data = [str(i) for i in range(0, num_cluster)]
        ax2.bar(x=x_data, height=scores[i])
        ax2.axis([-1, 10, 0, 1])
        ax2.tick_params(labelsize=23)

        att = attention_maps[i, 0]
        att = Image.fromarray(np.uint8(att*255))
        att = att.resize((imgs.shape[2], imgs.shape[3]), resample=PIL.Image.BILINEAR)
        att = np.asarray(att)

        att_mask = np.zeros_like(img)
        att_mask[:, :, 0] = att

        alpha = 0.1
        img_att = np.uint8((1-alpha)*img + alpha*att_mask)

        cmap = plt.get_cmap('jet')
        attMap = att
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2) * 255

        # attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img +\
        #          (attMap ** 0.7).reshape(attMap.shape + (1,)) * attMapV

        attMap = 0.4*img*255 + 0.6*attMapV
        attMap = attMap.astype(np.uint8)

        ax3.imshow(attMap)
        ax3.axis("off")

        plt.tight_layout()
        plt.savefig("{}/{}.png".format(save_folder, i))

        # plt.show()