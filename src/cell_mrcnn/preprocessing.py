from cell_mrcnn.utils import preproc_pipeline
from cell_mrcnn.data import get_channel_paths
from cell_mrcnn import __file__ as src_path
from skimage.io import imread
from PIL import Image
from os.path import join, isdir, split
from os import mkdir
ROOT_DIR = src_path.split('src')[0]
data_dir = join(ROOT_DIR, 'data')
# todo: B03s5 is all blue for some reason
if __name__ == '__main__':
    w2_paths = get_channel_paths(
        join(data_dir,
             '20201112-CB2-1wt-2Q63R-3L133i-4Cer '
             'L10/2020-11-12/1340_cellm/TimePoint_1'),
        'w2', ('A01', 'A02', 'A03', 'B01', 'B02', 'B03'))
    w2_paths.sort()
    w3_paths = get_channel_paths(
        join(data_dir,
             '20201112-CB2-1wt-2Q63R-3L133i-4Cer L10/2020-11-12/1340_cellm/TimePoint_1'),
        'w3', ('A01', 'A02', 'A03', 'B01', 'B02', 'B03'))
    w3_paths.sort()

    output_folder = join(data_dir,
                         '20201112-CB2-1wt-2Q63R-3L133i-4Cer L10/2020-11-12/1340_cellm')
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

        # red output filename
        red_tail_of_path = split(red_path)[1]
        red_col_and_pos = red_tail_of_path.split('_')[1:3]
        red_col_and_pos = ''.join(red_col_and_pos)
        # the output filename
        blue_tail_of_path = split(blue_path)[1]
        blue_col_and_pos = blue_tail_of_path.split('_')[1:3]
        blue_col_and_pos = ''.join(blue_col_and_pos)

        assert blue_col_and_pos == red_col_and_pos, "col-pos mismatch"

        fname = blue_col_and_pos
        Image.fromarray(comp).save(join(composite_folder, fname + '.png'))




