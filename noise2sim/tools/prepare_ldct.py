import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Compute the similarity images and save them into a LMDB dataset file.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-folder", default='./datasets/Mayo')
    parser.add_argument("--patient-folder", nargs="+", default=['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310'])  # ['L506', 'L333']
    parser.add_argument("--output-file", default='./datasets/Mayo/mayo_train.txt')
    args = parser.parse_args()

    fid_save = open(args.output_file, 'w')
    idx_min = 0

    for pf in args.patient_folder:
        folder_LDCT = '{}/{}/quarter_1mm'.format(args.data_folder, pf)
        folder_NDCT = '{}/{}/full_1mm'.format(args.data_folder, pf)

        names_LDCT = os.listdir(folder_LDCT)
        names_LDCT.sort()
        num_LDCT = len(names_LDCT)
        names_NDCT = os.listdir(folder_NDCT)
        names_NDCT.sort()
        num_NDCT = len(names_NDCT)

        assert num_NDCT == num_LDCT
        idx_max = idx_min + num_NDCT - 1
        for i in range(num_LDCT):
            name_LDCT = names_LDCT[i]
            file_LDCT = '{}/{}'.format(folder_LDCT, name_LDCT)
            name_NDCT = names_NDCT[i]
            file_NDCT = '{}/{}'.format(folder_NDCT, name_NDCT)
            print(file_LDCT, file_NDCT)
            assert os.path.exists(file_LDCT)
            assert os.path.exists(file_NDCT)
            if i == 0:
                label = 0
            elif i == num_LDCT - 1:
                label = -1
            else:
                label = 1
            fid_save.write('{} {} {} {} {}\n'.format(file_LDCT, file_NDCT, label, idx_min, idx_max))

        idx_min = idx_min + num_NDCT

    fid_save.close()


if __name__ == "__main__":
    main()





