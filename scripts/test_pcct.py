import argparse
from noise2sim.tools.test_pcct import main

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/pcct_livemouse_unet2.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)


def test():
    args = parser.parse_args()
    main(args.config_file)


if __name__ == '__main__':
    test()
