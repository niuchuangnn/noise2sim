import argparse
from noise2sim.tools.train import main
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/bsd400_unet2_ps3_ns8_gpu1.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)


def train():
    args = parser.parse_args()
    main(args.config_file)


if __name__ == '__main__':
    train()
