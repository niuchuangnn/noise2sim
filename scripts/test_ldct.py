import argparse
from noise2sim.tools.test_ldct import main
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/ldct_mayo_unet2.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)


def test():
    args = parser.parse_args()
    main(args.config_file)


if __name__ == '__main__':
    test()
