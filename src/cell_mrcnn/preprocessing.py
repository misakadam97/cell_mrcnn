from cell_mrcnn.utils import preproc_pipeline
from cell_mrcnn.data import get_channel_paths
from skimage.io import imread
from PIL import Image
from os.path import join, isdir, split, dirname, realpath
from os import mkdir
import os
import glob
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("-f","--filters", nargs='+', help="give filters to select the right folder and subfolder (e.g Timepoint 1.)", required=True)
parser.add_argument("-o","--output_folder",  help="give filters to select the right folder and subfolder (e.g Timepoint 1.)", required=True)
args = parser.parse_args()

src_path = dirname(realpath(__file__))
ROOT_DIR = join(src_path.split('cell_mrcnn')[0], 'cell_mrcnn')

data_dir = join(ROOT_DIR, 'data')

#folders = glob.glob(data_dir + '/*/')

subfolders = glob.glob(data_dir + '**/**/', recursive = True)

for filt in args.filters:
    subfolders = [x for x in subfolders if filt in x]

active_folder = list(set(subfolders))[0]

print(f'Selected folder for analysis: {active_folder}')

w2_paths = get_channel_paths(active_folder, 'w2', ('A01', 'A02', 'A03', 'B01', 'B02', 'B03'))

w2_paths.sort()

w3_paths = get_channel_paths(active_folder, 'w3', ('A01', 'A02', 'A03', 'B01', 'B02', 'B03'))

w3_paths.sort()

output_folder = join(data_dir, args.output_folder)

if not isdir(output_folder):
     mkdir(output_folder)

composite_folder = join(output_folder, 'composite')

if not isdir(composite_folder):
     mkdir(composite_folder)


for i, (red_path, blue_path) in enumerate(zip(w2_paths, w3_paths)):
    print('\rCreating composite images: {}/{}'.format(i+1,len(w2_paths)),
          end='...')
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


