import os
import argparse
import PIL.Image
import numpy as np
import lmdb
from tqdm import tqdm
from tifffile import imread
import shutil
import pickle
import os.path as path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from collections import defaultdict
import torch
import faiss
from tools.nearest_search import search_raw_array_pytorch

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


def read_dir(dir_path, predicate=None, name_only=False, recursive=False):
    if type(predicate) is str:
        if predicate in {'dir', 'file'}:
            predicate = {
                'dir': lambda x: path.isdir(path.join(dir_path, x)),
                'file':lambda x: path.isfile(path.join(dir_path, x))
            }[predicate]
        else:
            ext = predicate
            predicate = lambda x: ext in path.splitext(x)[-1]
    elif type(predicate) is list:
        exts = predicate
        predicate = lambda x: path.splitext(x)[-1][1:] in exts

    def read_dir_(output, dir_path, predicate, name_only, recursive):
        if not path.isdir(dir_path): return
        for f in os.listdir(dir_path):
            d = path.join(dir_path, f)
            if predicate is None or predicate(f):
                output.append(f if name_only else d)
            if recursive and path.isdir(d):
                read_dir_(output, d, predicate, name_only, recursive)

    output = []
    read_dir_(output, dir_path, predicate, name_only, recursive)
    return sorted(output)


examples='''examples:

  python %(prog)s --input-dir=./ILSVRC2012_img_val --out=imagenet_val_raw.h5
'''

def main():
    parser = argparse.ArgumentParser(
        description='Convert a set of image files into a HDF5 dataset file.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-folder", default='/media/niuchuang/Storage/DataSets/Fluo-N2DH-GOWT1-test',
                        help="Directory containing ImageNet images (can be glob pattern for subdirs)")
    parser.add_argument("--output-file", default='/media/niuchuang/Storage/DataSets/Fluo-N2DH-GOWT1-test/ctc_gowt1_ps7_ns8_lmdb',
                        help="Filename of the output file")
    args = parser.parse_args()

    img_files = read_dir(args.data_folder, predicate=lambda x: x.endswith("tif"), recursive=True)[0:10]
    num_images = len(img_files)

    data_noise = []
    for img_path in img_files:
        # img_path = "{}/{}".format(args.data_folder, img_name)
        img_noise = imread(img_path).astype(np.float32)
        data_noise.append(img_noise)

    # ----------------------------------------------------------
    if os.path.exists(args.output_file):
        print("{} exists, deleted...".format(args.output_file))
        # sys.exit(1)
        shutil.rmtree(args.output_file)

    num_sim = 8
    patch_size = 7

    commit_interval = 10

    # Estimate the lmdb size.
    data_size_per_img = data_noise[-1].nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * num_images * (num_sim + 1)

    env = lmdb.open(args.output_file, map_size=data_size*1.1)

    txn = env.begin(write=True)
    shapes = []
    tqdm_iter = tqdm(enumerate(range(num_images)), total=num_images, leave=False)

    keys = []
    for idx, key in tqdm_iter:

        tqdm_iter.set_description('Write {}'.format(key))
        keys.append(str(key))

        img_noise = data_noise[idx]
        img_noise_norm = img_noise / 255.0
        print(img_noise_norm.max())

        img_noise_sim = compute_sim_images(img_noise, patch_size=patch_size, num_select=num_sim, img_ori=img_noise_norm)

        key_noise_byte = "{}_noise".format(key).encode('ascii')
        key_noise_sim_byte = "{}_noise_sim".format(key).encode('ascii')

        txn.put(key_noise_byte, img_noise)
        txn.put(key_noise_sim_byte, img_noise_sim)

        H, W = img_noise.shape
        C = 1
        shapes.append('{:d}_{:d}_{:d}'.format(H, W, C))

        # plt.figure()
        # plt.imshow(img_noise, cmap="gray")
        #
        # for i in [1, 9, 31, 63]:
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
                 "num_sim": num_sim,
                 "patch_size": patch_size}

    pickle.dump(meta_info, open("{}_metal_info.pkl".format(args.output_file), "wb"))
    print('Finish creating lmdb meta info.')


if __name__ == "__main__":
    main()





