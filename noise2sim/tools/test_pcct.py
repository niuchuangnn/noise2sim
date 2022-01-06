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
from matplotlib.pyplot import imsave
from scipy.io import savemat


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def clip_to_uint8(arr):
    if isinstance(arr, np.ndarray):
        return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    x = torch.clamp(arr * 255.0 + 0.5, 0, 255).to(torch.uint8)
    return x


def PSNR(gt, img):
    mse = np.mean(np.square((gt - img)*255))
    return 20 * np.log10(255) - 10 * np.log10(mse)


def to_image(arr):
    img = np.clip(arr, 0, 1) * 255
    img = img.astype(np.uint8)
    return img


def normalize(arr):
    img = (arr - arr.min()) / (arr.max() - arr.min())
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

    if cfg.data_test is not None:
        dataset_val = build_dataset(cfg.data_test)
    else:
        cfg.data_train.random_flip = False
        cfg.data_train.crop_size = None
        dataset_val = build_dataset(cfg.data_train)

    crop = None
    model.eval()
    hu_range = cfg.data_train.hu_range
    show = False
    noisy = []
    preds = []
    for n in range(0, len(dataset_val)):
        images, images_clean, _, idx = dataset_val[n]
        data_list = [images, images_clean]
        for l in range(len(data_list)):
            data_list[l] = data_list[l].unsqueeze(0)
        images, images_clean = data_list
        if crop is not None:
            images = images[:, :, crop[0]:crop[1], crop[2]:crop[3]]
        inputs = images.to(cfg.gpu)

        with torch.no_grad():
            outputs = model(inputs)

        pred_n = outputs.cpu().numpy()
        pred_n = pred_n * (hu_range[1] - hu_range[0]) + hu_range[0]
        noisy_n = inputs.cpu().numpy()
        noisy_n = noisy_n * (hu_range[1] - hu_range[0]) + hu_range[0]

        preds.append(pred_n)
        noisy.append(noisy_n)

        img_noise = to_image(images.cpu()[0].numpy())
        img_out = to_image(outputs.cpu()[0].numpy())
        noise_out = img_noise - img_out
        noise_out = normalize(noise_out.astype(np.float))

        img_save = [img_noise, img_out, noise_out]
        names = ['ldct', 'pred', 'pred_noise']

        for i in range(len(img_save)):
            name = names[i]
            img = img_save[i]
            for ii in range(img.shape[0]):
                img_path = "{}/{}_{}_{}.png".format(cfg.output_dir_test, idx, name, ii)
                print(idx, img_path)

                imsave(img_path, img[ii, :, :], cmap='gray')
                if show:
                    plt.figure()
                    plt.imshow(img[ii, :, :], cmap='gray')
                    plt.title('{}_{}'.format(name, ii))
        if show:
            plt.show()

    preds = np.concatenate(preds, axis=0)
    noisy = np.concatenate(noisy, axis=0)

    savemat("{}/preds.mat".format(cfg.output_dir_test), {'preds': preds})
    savemat("{}/noisy.mat".format(cfg.output_dir_test), {'noisy': noisy})

