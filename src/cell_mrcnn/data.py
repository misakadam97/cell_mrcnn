from os import listdir, walk, mkdir, path
from os.path import isfile, join, isdir, split, basename
from glob import glob
import sys
import numpy as np
import skimage.draw
from skimage.io import imread
from cell_mrcnn.utils import get_concentric_intensities, \
    correct_central_brightness, subtract_bg, convert_to_bit8, \
    get_data_path_from_config_file
import read_roi
import gc
import pandas as pd
from datetime import datetime
from shutil import copyfile

# data directory
data_dir = get_data_path_from_config_file()
dataset_dir = join(data_dir, 'annotated_datasets/')

# Import Mask RCNN
from cell_mrcnn.config import Config
from cell_mrcnn import utils
from PIL import Image


# todo: some of the functions below might be better in utils?
def get_channel_paths(folder, channel,
                      wells=('A01', 'A02', 'A03', 'A04', 'B01', 'B02',
                               'B03', 'B04')):
    """

    :param folder:
    :param channel: 'w1' or 'w2' or 'w3'
    :param wells: any comb of ['A01', 'A02', 'A03', 'A04', 'B01', 'B02',
    'B03', 'B04']
    :return:
    """
    paths = glob(join(folder, '**/*.tif'), recursive=True)
    channel_paths = []
    for path in paths:
        path_ = split(path)[1]
        if 'thumb' in path_:
            continue
        if path_.split('_')[-1][:2] != channel:
            continue
        if path_.split('_')[1] not in wells:
            continue
        channel_paths.append(path)
    return channel_paths


def calculate_percentiles(im_paths):
    # takes ~2mins
    # on 2048x2048 images the below percentiles leave about _ pixels out:
    # 99: 42k
    # 99.9: 4.2k
    # 99.99: 420
    # 99.999: 42
    # 99.9999: 4
    # 100: 0 , this is equal to max
    percentiles = [99, 99.9, 99.99, 99.999, 99.9999, 100]
    perc_df = pd.DataFrame(columns=percentiles,
                           index=pd.RangeIndex(0, len(im_paths)))
    for i, im_path in enumerate(im_paths):
        im = imread(im_path)
        im = correct_central_brightness(im.astype(np.float))
        im = subtract_bg(im)
        percentile_list = []
        for perc in percentiles:
            percentile_list.append(np.percentile(im, perc))
        perc_df.loc[i, :] = percentile_list

    perc_df.loc["min", :] = perc_df.min(axis=0)
    perc_df.loc["max", :] = perc_df.loc[perc_df.index[0:-1], :].max(axis=0)
    perc_df.loc["mean", :] = perc_df.loc[perc_df.index[0:-2], :].mean(axis=0)
    perc_df.loc["median", :] = perc_df.loc[perc_df.index[0:-3], :].median(
        axis=0)
    perc_df.loc["var", :] = perc_df.loc[perc_df.index[0:-4], :].var(axis=0)
    perc_df.loc["std", :] = perc_df.loc[perc_df.index[0:-5], :].std(axis=0)

    return perc_df


def transfer_w3_channel_images(data_dir):
    cit_dir = join(data_dir, 'w3/cit')
    cellmembrane_dir = join(data_dir, 'w3/cellmembrane')
    if not isdir(cit_dir):
        mkdir(cit_dit)
    if not isdir(cellmembrane_dir):
        mkdir(cellmembrane_dir)
    # get the path to all tif images
    im_paths = glob(join(data_dir, '**/*.tif'), recursive=True)
    # select the non-thumbnail w3 images
    for i, im_path in enumerate(im_paths):
        print('\rSeparating cit. and cellmembrane w3 channel images: ', i + 1,
              '/',
              len(im_paths), end='')
        if 'thumb' in im_path:
            continue
        if im_path.split('_')[-1][:2] != 'w3':
            continue
        im = imread(im_path)
        im = convert_to_bit8(im)
        im = Image.fromarray(im)
        if 'cit' in im_path:
            im.save(join(cit_dir, im_path.split('/')[-1].split('.')[0]
                         + '.png'))
        else:
            im.save(join(cellmembrane_dir, im_path.split('/')[-1].split('.')[0]
                         + '.png'))


def cell_groups_to_bg(image, roi_set):
    cell_group_rois = [roi_set[key] for key in roi_set.keys() if 'cell_group'
                       in key]
    im = np.copy(image)
    for cg_roi in cell_group_rois:
        rr, cc = skimage.draw.polygon(cg_roi['y'], cg_roi['x'])
        mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
        mask[rr, cc] = 1
        rand_bg = np.random.randint(im.mean() - im.std(), im.mean() +
                                    im.std(), size=mask.sum())
        mask[rr, cc] = rand_bg
        im[rr, cc] = mask[rr, cc]
    return im


def read_roi_or_roiset(impath):
    impath = impath.split('.')[0]
    # if rois are in a zip (image contains multiple rois)
    if path.isfile(join(impath + '_roi.zip')):
        rois = read_roi.read_roi_zip(
            join(impath + '_roi.zip'))
    # if roi is in roi format(image only contains 1 roi)
    elif path.isfile(join(impath + '.roi')):
        rois = read_roi.read_roi_file(join(impath + '.roi'))
    else:
        print("rois couldn't be found for:", impath)

    return rois


def calc_avg_pixel_value(image_paths, output_file = None):
    channel1, channel2, channel3 = [], [], []
    for image_path in image_paths:
        image = imread(image_path)
        channel1.append(image[:, :, 0].mean())
        channel2.append(image[:, :, 1].mean())
        channel3.append(image[:, :, 2].mean())

    channel1 = np.array(channel1).mean().reshape(1)
    channel2 = np.array(channel2).mean().reshape(1)
    channel3 = np.array(channel3).mean().reshape(1)

    if output_file:
        with open(output_file, 'w') as f:
            f.write('Channel 1 mean: {}\nChannel 2  mean: {}\nChannel 3 '
                    'mean: {}'.format(channel1, channel2, channel3))

    return np.concatenate((channel1, channel2, channel3))


def copy_annotated(input_folder, output_folder):
    """
    creates a folder named the current datetime; and copies the roi.zip files
    and corresponding images from the input folder into it
    :param input_folder:
    :param output_folder:
    :return:
    """

    # get a list of roi paths
    roi_paths = glob(join(input_folder, '*_roi.zip'))

    # get the corresponding image paths
    image_paths = []
    for roi_path in roi_paths:
        image_paths.append(roi_path.split('_roi.zip')[0] + '.png')

    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_folder = join(output_folder, date)
    if not isdir(output_folder):
        mkdir(output_folder)

    for roi, im in zip(roi_paths, image_paths):
        copyfile(roi, join(output_folder, basename(roi)))
        copyfile(im, join(output_folder, basename(im)))





class CellTransformData(utils.Dataset):

    def load_cell(self, dataset_dir):
        """Load the cell dataset resize the images and rois to a
        uniform dimension.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("hulab", 1, "cell")

        image_names = [file_name for file_name in listdir(dataset_dir) if
                       'png' in file_name]
        # todo: revisit this naming convention, come up w/ a unifrom system
        # just so that the orderly assigned id is in the utils. Dataset class
        # is in the same order as the numbers in the image names
        # image_names = [int(name.split('.png')[0]) for name in image_names]
        # image_names.sort()
        # image_names = [str(n) + '.png' for n in image_names]

        # Add images
        for image_name in image_names:
            id = image_name.split('.')[0]
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance.
            # RoI format:
            # roi = {
            #       region1 : {
            #                 'type' = 'freehand'
            #                 'x' = [...]
            #                 'y' = [...]
            #                 'n' = 412
            #                 'width' = 0 # always 0, not informative
            #                 'name' = the region name (same as the key)
            #                 'position' = 0 # always 0, not informative
            #                 }
            #        ...more regions
            #       }
            rois = read_roi_or_roiset(join(dataset_dir, id))
            im_path = join(dataset_dir, image_name)
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, RoI doesn't include it, so we must read
            # the image. This is only managable since the dataset is tiny.
            h, w = imread(im_path).shape[:2]

            self.add_image(
                "hulab",
                image_id=id,  # use file name as a unique image id
                path=im_path,
                width=w, height=h,
                polygons=rois)

    def polygon_to_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a confonc dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "hulab":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # /proc/sys/vm/overcommit_memory has to be "1" for larger arrays
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.bool)
        for i, (key, vals) in enumerate(info["polygons"].items()):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(vals['y'], vals['x'])
            mask[rr, cc, i] = True
            del (rr, cc)
            gc.collect()

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def split_to_trainval(self, dataset_dir):
        # set up the output directory's subfolders
        [mkdir(join(dataset_dir, folder)) for folder in ['train', 'val'] if
         folder not in listdir(dataset_dir)]

        np.random.seed(seed=54646)
        train_set = np.random.choice(self.image_ids,
                                     int(len(self.image_ids) * 0.7),
                                     replace=False)

        means = []
        for image_id_ in self.image_ids:
            info = self.image_info[image_id_]
            im = self.load_image(image_id_)
            means.append(im.mean(axis=(0, 1)))

            mask = self.polygon_to_mask(image_id_)[0]
            id_ = info['id']

            if image_id_ in train_set:
                output_dir = join(dataset_dir, 'train', str(id_))
            else:
                output_dir = join(dataset_dir, 'val', str(id_))
            if not isdir(output_dir):
                mkdir(output_dir)

            for m in range(mask.shape[2]):
                mask_ = Image.fromarray((mask[:, :, m] * 255).astype(np.uint8),
                                        mode='L')
                mask_ = mask_.convert(mode='1')
                mask_.save(join(output_dir, str(id_) + '_mask_' + str(m) \
                                + '.png'))

            copyfile(info['path'], join(output_dir, str(id_) + '.png'))
        print('Average pixel value is: {}'.format(np.array(means).mean(
            axis=0)))
        return np.array(means)


if __name__ == '__main__':
    # load the dataset
    dataset_path = join(dataset_dir, '2020_11_22_02_55_03')
    ds = CellTransformData()
    ds.load_cell(dataset_path)
    ds.prepare()
    ds.split_to_trainval(dataset_path)

    show_example = False

    if show_example:
        # test
        image_id = 12
        image = ds.load_image(image_id)
        mask, class_ids = ds.polygon_to_mask(image_id)
        original_shape = image.shape
        # Resize
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=2048,
            max_dim=2048,
            mode="none")
        mask = utils.resize_mask(mask, scale, padding, crop)
        # Compute Bounding box
        bbox = utils.extract_bboxes(mask)
        # Display image and additional stats
        print("image_id: ", image_id, ds.image_reference(image_id))
        print("Original shape: ", original_shape)
        from mrcnn.model import log
        from mrcnn import visualize

        log("image", image)
        log("mask", mask)
        log("class_ids", class_ids)
        log("bbox", bbox)
        # Display image and instances
        visualize.display_instances(image, bbox, mask, class_ids,
                                    ds.class_names)
