import argparse
from noise2sim.tools.prepare_bsd400_lmdb import main


def prepare_bsd400():
    parser = argparse.ArgumentParser(
        description='Compute the similarity images and save them into a LMDB dataset file.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-folder", default='./datasets/Train400')
    parser.add_argument("--output-folder", default='./datasets')
    parser.add_argument("--noise-type", default="gaussian")
    parser.add_argument("--std", default=25)
    parser.add_argument("--lam", default=30)
    parser.add_argument("--num-sim", default=8)
    parser.add_argument("--patch-size", default=3)
    parser.add_argument("--max-files", default=1000)
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--model_weight", default=None)
    parser.add_argument("--lmdb_file", default=None)
    parser.add_argument("--key_file", default=None)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    prepare_bsd400()





