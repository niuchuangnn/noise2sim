import builtins
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import torch.utils.data
import torch.utils.data.distributed

import torchvision.models as models
from ..config import Config
from ..modeling.architectures import build_architecture
from ..data.bulid_data import build_dataset
from ..utils.miscellaneous import mkdir
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as SSIM
from matplotlib.pyplot import imsave
from scipy.io import savemat


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def PSNR(gt, img):
    mse = np.mean(np.square((gt - img)*255))
    return 20 * np.log10(255) - 10 * np.log10(mse)


def to_image(arr):
    img = np.clip(arr, 0, 1)
    return img


def normalize(arr):
    img = (arr - arr.min()) / (arr.max() - arr.min())
    return img


def to_hu(arr, range=[-160, 240], hu_target=[-160, 240]):
    img = arr * (range[1] - range[0]) + range[0]
    img = np.clip(img, hu_target[0], hu_target[1])
    img = (img - hu_target[0]) / (hu_target[1] - hu_target[0])
    return img


def main(config_file):

    cfg = Config.fromfile(config_file)
    cfg.gpu = 0
    output_dir = cfg.results.output_dir
    output_dir_test = "{}/test".format(output_dir)
    if output_dir_test:
        mkdir(output_dir_test)
    cfg.output_dir_test = output_dir_test

    main_worker(cfg.gpu, cfg)


def main_worker(gpu, cfg):
    cfg.gpu = gpu

    # suppress printing if not master
    if cfg.multiprocessing_distributed and cfg.gpu != 0:
        def print_pass(*cfg):
            pass
        builtins.print = print_pass

    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    # create model
    model = build_architecture(cfg.model)
    print(model)

    torch.cuda.set_device(cfg.gpu)
    model = model.cuda(cfg.gpu)

    # load model from a checkpoint
    checkpoint = torch.load(cfg.model_weight)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # Initialize the feature module with encoder_q of moco.
        if k.startswith('module'):
            # remove prefix
            state_dict[k[len('module.'):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    model.load_state_dict(state_dict)

    cudnn.benchmark = True

    # Data loading code
    dataset_val = build_dataset(cfg.data_test)

    model.eval()
    psnrs = []
    ssims = []
    show = False

    inputs_all = []
    targets_all = []

    for n in range(len(dataset_val)):
        images, images_clean, _, idx = dataset_val[n]
        images = images.unsqueeze(0)
        images_clean = images_clean.unsqueeze(0)

        inputs_all.append(images.numpy())
        targets_all.append(images_clean.numpy())

        inputs = images.to(cfg.gpu)
        with torch.no_grad():
            outputs = model(inputs)

        img_noise = to_image(images.cpu().squeeze().numpy())
        img_clean = to_image(images_clean.squeeze().numpy())
        img_out = to_image(outputs.squeeze().cpu().numpy())

        multichannel = len(img_out.shape) > 2

        if multichannel:
            img_noise = img_noise.transpose([1, 2, 0])
            img_clean = img_clean.transpose([1, 2, 0])
            img_out = img_out.transpose([1, 2, 0])

        psnr = PSNR(img_clean, img_out)
        ssim = SSIM(img_clean, img_out, multichannel=multichannel)
        psnrs.append(psnr)
        ssims.append(ssim)

        print(psnr)
        print(ssim)

        img_save = [img_noise, img_out, img_clean]
        names = ['ldct', 'pred', 'ndct']

        for i in range(len(img_save)):
            name = names[i]
            if 'pred' in name:
                name = '{}_psnr_{}_ssim_{}'.format(name, psnr, ssim)
            img_path = "{}/{}_{}.png".format(cfg.output_dir_test, idx, name)
            print(idx, img_path)
            img = img_save[i] * 255
            img = img.astype(np.uint8)
            if multichannel:
                imsave(img_path, img)
            else:
                imsave(img_path, img, cmap='gray')
            if show:
                plt.figure()
                plt.imshow(img, cmap='gray')
                plt.title(name)
        if show:
            plt.show()
        pass

    psnrs = np.array(psnrs)
    ssims = np.array(ssims)

    savemat('{}/psnrs.mat'.format(cfg.output_dir_test), {'data': psnrs})
    savemat('{}/ssims.mat'.format(cfg.output_dir_test), {'data': ssims})

    psnr_mean = psnrs.mean()
    psnr_std = psnrs.std()
    print("PSNR mean: {}, std: {}".format(psnr_mean, psnr_std))
    ssim_mean = ssims.mean()
    ssim_std = ssims.std()
    print("SSIM mean: {}, std: {}".format(ssim_mean, ssim_std))

