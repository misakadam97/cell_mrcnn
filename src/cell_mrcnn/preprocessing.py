from cell_mrcnn.utils import preproc_pipeline, get_data_path_from_config_file
from cell_mrcnn.data import get_channel_paths
from skimage.io import imread
from PIL import Image
from os.path import join, isdir, split, dirname, realpath
from os import mkdir
import os
import glob
import argparse
from pathlib import Path


def preprocess(filters, output_folder, mask_pattern='w3', sample_pattern='w2'):
    """filters: list of strings in the experiment folder which toghether are unique to the experiment
    output_folder: name of the folder where the processed images will be stored"""

    data_dir = get_data_path_from_config_file()

    # folders = glob.glob(data_dir + '/*/')

    subfolders = glob.glob(data_dir + '**/**/', recursive=True)

    for filt in filters:
        subfolders = [x for x in subfolders if filt in x]

    active_folder = list(set(subfolders))[0]

    print(f'Selected folder for analysis: {active_folder}')

    w2_paths = get_channel_paths(active_folder, sample_pattern,
                                 ('A01', 'A02', 'A03', 'B01', 'B02', 'B03'))

    w2_paths.sort()

    w3_paths = get_channel_paths(active_folder, mask_pattern,
                                 ('A01', 'A02', 'A03', 'B01', 'B02', 'B03'))

    w3_paths.sort()

    output_folder = join(data_dir, output_folder)

    if not isdir(output_folder):
        mkdir(output_folder)

    # save the active folder path
    with open(join(output_folder, 'experiment_path.path'), 'w') as f:
        f.write(active_folder)

    composite_folder = join(output_folder, 'composite')

    if not isdir(composite_folder):
        mkdir(composite_folder)

    for i, (red_path, blue_path) in enumerate(zip(w2_paths, w3_paths)):
        print(
            '\rCreating composite images: {}/{}'.format(i + 1, len(w2_paths)),
            end='...')
        try:
            red, blue = imread(red_path), imread(blue_path)
            comp = preproc_pipeline(red, blue)

            # todo: refractor w/ utils.get_well_and_pos()
            # todo: also use '_'.join(); and maybe strip the "s" b4 position number
            # red output filename
            red_tail_of_path = split(red_path)[1]
            red_well_and_pos = red_tail_of_path.split('_')[1:3]
            red_well_and_pos = ''.join(red_well_and_pos)
            # the output filename
            blue_tail_of_path = split(blue_path)[1]
            blue_well_and_pos = blue_tail_of_path.split('_')[1:3]
            blue_well_and_pos = ''.join(blue_well_and_pos)

            assert blue_well_and_pos == red_well_and_pos, "well-pos mismatch"

            fname = blue_well_and_pos
            Image.fromarray(comp).save(join(composite_folder, fname + '.png'))

        except:
            print(f'image {i} processing failed')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filters", nargs='+',
                        help="give filters to select the right folder and subfolder (e.g Timepoint 1.)",
                        required=True)
    parser.add_argument("-o", "--output_folder",
                        help="give filters to select the right folder and subfolder (e.g Timepoint 1.)",
                        required=True)
    parser.add_argument("-m", "--mask_pattern",
                        help="pattern which defines the mask channel e.g. w3",
                        required=False)
    parser.add_argument("-s", "--sample_pattern",
                        help="pattern which defines the sample channel e.g. w2",
                        required=False)
    args = parser.parse_args()

    if not args.mask_pattern:
        mask_pattern = 'w3'
    else:
        mask_pattern = args.mask_pattern

    if not args.sample_pattern:
        sample_pattern = 'w2'
    else:
        sample_pattern = args.sample_pattern

    filters = args.filters
    output_folder = args.output_folder

    preprocess(filters, output_folder, mask_pattern, sample_pattern)
