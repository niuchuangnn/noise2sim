import os
import argparse
import PIL.Image
import numpy as np
import lmdb
from tqdm import tqdm
from imageio import imread
import shutil
import pickle

from ..config import Config
from ..modeling.architectures import build_architecture

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from collections import defaultdict
import torch
import faiss
from ..tools.nearest_search import search_raw_array_pytorch

# resource object, can be re-used over calls
res = faiss.StandardGpuResources()
# put on same stream as pytorch to avoid synchronizing streams
res.setDefaultNullStreamAllDevices()

size_stats = defaultdict(int)
format_stats = defaultdict(int)


def load_image(fname):
    global format_stats, size_stats
    im = PIL.Image.open(fname)
    format_stats[im.mode] += 1
    if (im.width < 256 or im.height < 256):
        size_stats['< 256x256'] += 1
    else:
        size_stats['>= 256x256'] += 1
    arr = np.array(im.convert('RGB'), dtype=np.uint8)
    assert len(arr.shape) == 3
    # return arr.transpose([2, 0, 1])
    return arr


def filter_image_sizes(images):
    filtered = []
    for idx, fname in enumerate(images):
        if (idx % 100) == 0:
            print ('loading images', idx, '/', len(images))
        try:
            with PIL.Image.open(fname) as img:
                w = img.size[0]
                h = img.size[1]
                if (w > 512 or h > 512) or (w < 256 or h < 256):
                    continue
                filtered.append((fname, w, h))
        except:
            print ('Could not load image', fname, 'skipping file..')
    return filtered


def nofilter_image_sizes(images):
    filtered = []
    for idx, fname in enumerate(images):
        if (idx % 100) == 0:
            print ('loading images', idx, '/', len(images))
        try:
            with PIL.Image.open(fname) as img:
                w = img.size[0]
                h = img.size[1]
                if (w > 512 or h > 512) or (w < 256 or h < 256):
                    continue
                filtered.append((fname, w, h))
        except:
            print ('Could not load image', fname, 'skipping file..')
    return filtered


def noisify(img, noise_type="gaussian", std=25, lam=30):
    if noise_type == "gaussian":
        std = std / 255.0
        img_noise = img + np.random.normal(size=img.shape) * std
    elif noise_type == "poisson":
        img_noise = np.random.poisson(img*lam) / float(lam)
    else:
        raise TypeError
    return img_noise


def shift_concat_image(img_noise, patch_size):
    if len(img_noise.shape) < 3:
        [H, W] = img_noise.shape
        C = 1
        img_noise = img_noise.reshape([H, W, C])
    else:
        [H, W, C] = img_noise.shape
    img_noise_pad = np.pad(img_noise, pad_width=((patch_size, patch_size),
                                                (patch_size, patch_size),
                                                (0, 0)),
                           mode="reflect")
    patches = np.zeros([H, W, patch_size * patch_size, C])
    for i in range(-(patch_size - 1) // 2, (patch_size - 1) // 2 + 1):
        for j in range(-(patch_size - 1) // 2, (patch_size - 1) // 2 + 1):
            if i == 0 and j == 0:
                continue  # exclude the center pixel
            h_start = max(0, i + patch_size)
            h_end = min(H + 2 * patch_size, i + patch_size + H)
            w_start = max(0, j + patch_size)
            w_end = min(W + 2 * patch_size, j + patch_size + W)

            pi = i + (patch_size - 1) // 2
            pj = j + (patch_size - 1) // 2

            patches[:, :, (pi * patch_size + pj), :] = \
                img_noise_pad[h_start:h_end, w_start:w_end, :]

    return patches


def search_nlm_images_gpu(img, patches, num_select):
    [H, W, D, C] = patches.shape
    patches = patches.reshape([H * W, D * C]).astype(np.float32)
    patches = torch.from_numpy(patches).to("cuda:0")
    dist, ind_y = search_raw_array_pytorch(res, patches, patches, num_select, metric=faiss.METRIC_L2)

    images_sim = np.zeros([num_select, H * W, C]).astype(np.float32)

    for s in range(num_select):
        images_sim[s, :, :] = img.reshape([H * W, C])[ind_y[:, s].cpu().numpy(), :]

    images_sim = images_sim.reshape([num_select, H, W, C])
    return images_sim


def compute_sim_images(img, patch_size, num_select, img_ori=None):
    if img_ori is not None:
        patches = shift_concat_image(img_ori, patch_size)
    else:
        patches = shift_concat_image(img, patch_size)
    images_sim = search_nlm_images_gpu(img, patches, num_select+1)
    return images_sim[1::, ...]


def clip_to_unit8(arr):
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def read_data_file(data_file):
    f = open(data_file, 'r')
    lines = f.readlines()
    data_pathes = []
    for line in lines:
        data_pathes.append(line[0:-1])
    return data_pathes


def _get_keys_shapes_pickle(meta_info_file):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(meta_info_file, 'rb'))
    keys = meta_info['keys']
    shapes = meta_info['shapes']
    return keys, shapes


def _read_img_noise_lmdb(env, key, shape, dtype=np.uint8):
    with env.begin(write=False) as txn:
        buf = txn.get("{}_noise".format(key).encode('ascii'))
    img_noise_flat = np.frombuffer(buf, dtype=dtype)
    H, W, C = shape
    img_noise = img_noise_flat.reshape(H, W, C)
    return img_noise


def main(args):
    img_files = os.listdir(args.data_folder)
    num_images = len(img_files)

    if args.noise_type == "gaussian":
        noise_param = args.std
    elif args.noie_type == "poisson":
        noise_param = args.lam
    else:
        raise TypeError

    args.output_file = "{}/bsd400_{}{}_ps{}_ns{}_lmdb".format(args.output_folder, args.noise_type,
                                                                  noise_param, args.patch_size, args.num_sim)

    if args.config_file is None:
        data_noise = []
        for img_name in img_files:
            img_path = "{}/{}".format(args.data_folder, img_name)
            img = imread(img_path).astype(np.float32)
            img_norm = img / 255.
            img_noise = noisify(img_norm, noise_type=args.noise_type, std=args.std, lam=args.lam) * 255.
            data_noise.append(img_noise)

    else:
        data_noise = []
        keys, shapes = _get_keys_shapes_pickle(args.key_file)
        num_images = len(keys)
        data_env = lmdb.open(args.lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)
        for i in range(num_images):
            img_shape = [int(s) for s in shapes[i].split('_')]
            img_noise = _read_img_noise_lmdb(data_env, keys[i], img_shape, np.float32).astype(np.float32)
            data_noise.append(img_noise.squeeze())

        # Create network
        device = torch.device("cuda:0")
        cfg = Config.fromfile(args.config_file)
        model = build_architecture(cfg.model)
        print(model)
        model = model.to(device)

        # load model from a checkpoint
        checkpoint = torch.load(args.model_weight)
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # Initialize the feature module with encoder_q of moco.
            if k.startswith('module'):
                # remove prefix
                state_dict[k[len('module.'):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict)

        model.eval()

    data_noise = np.array(data_noise).astype(np.float32)
    data_noise_norm = (data_noise - data_noise.min()) / (data_noise.max() - data_noise.min())
    # ----------------------------------------------------------
    if os.path.exists(args.output_file):
        print("{} exists, deleted...".format(args.output_file))
        shutil.rmtree(args.output_file)

    commit_interval = 10

    # Estimate the lmdb size.
    data_nbytes = data_noise.astype(np.float32).nbytes
    data_size = data_nbytes * (args.num_sim + 1)

    env = lmdb.open(args.output_file, map_size=data_size*1.5)

    txn = env.begin(write=True)
    shapes = []
    tqdm_iter = tqdm(enumerate(range(num_images)), total=num_images, leave=False)

    keys = []
    for idx, key in tqdm_iter:

        tqdm_iter.set_description('Write {}'.format(key))
        keys.append(str(key))

        img_noise = data_noise[idx]
        img_noise_norm = data_noise_norm[idx]

        if args.config_file is not None:
            img_noise_norm_gpu = torch.from_numpy(img_noise_norm).to(torch.float32).to(device).reshape(1, 1, img_noise_norm.shape[0], img_noise_norm.shape[1])
            with torch.no_grad():
                img_noise_norm = model(img_noise_norm_gpu)
                img_noise_norm = img_noise_norm.cpu().numpy().squeeze()

        img_noise_sim = compute_sim_images(img_noise, patch_size=args.patch_size, num_select=args.num_sim,
                                           img_ori=img_noise_norm)

        key_noise_byte = "{}_noise".format(key).encode('ascii')
        key_noise_sim_byte = "{}_noise_sim".format(key).encode('ascii')

        txn.put(key_noise_byte, img_noise)
        txn.put(key_noise_sim_byte, img_noise_sim)

        H, W = img_noise.shape
        C = 1
        shapes.append('{:d}_{:d}_{:d}'.format(H, W, C))

        # plt.figure()
        # plt.imshow(img_noise, cmap="gray")

        # for i in [1, 3, 5, 7]:
        #     plt.figure()
        #     plt.imshow(img_noise_sim[i].squeeze(), cmap="gray")
        #
        # plt.show()

        if (idx + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish writing lmdb')

    meta_info = {"shapes": shapes,
                 "keys": keys,
                 "num_sim": args.num_sim,
                 "patch_size": args.patch_size}

    pickle.dump(meta_info, open("{}_meta_info.pkl".format(args.output_file), "wb"))
    print('Finish creating lmdb meta info.')





