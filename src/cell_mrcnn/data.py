from os import listdir, walk, mkdir, path
from os.path import isfile, join, isdir
from glob import glob
import sys
import json
import datetime
import numpy as np
import skimage.draw
from cell_mrcnn import __file__ as src_path
from cell_mrcnn.utils import convert_to_bit8
import read_roi
import matplotlib.pyplot as plt  # skimage.io.imread gives runtimeWarinig
import yaml
import copy
import gc
from nd2reader import ND2Reader

# Root directory of the project
ROOT_DIR = src_path.split('src')[0]
dataset_dir = ROOT_DIR + 'data/pngs/'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from cell_mrcnn.config import Config
from cell_mrcnn import utils
from PIL import Image


def transfer_w3_channel_images(data_dir):
    cit_dir = join(data_dir, 'w3/cit')
    cellmembrane_dir = join(data_dir, 'w3/cellmembrane')
    if not isdir(cit_dir):
        mkdir(cit_dit)
    if not isdir(cellmembrane_dir):
        mkdir(cellmembrane_dir)
    # get the path to all tif images
    im_paths = glob(join(data_dir,'**/*.tif'), recursive=True)
    # select the non-thumbnail w3 images
    for i, im_path in enumerate(im_paths):
        print('\rSeparating cit. and cellmembrane w3 channel images: ',i+1,'/',
              len(im_paths), end='')
        if 'thumb' in im_path:
            continue
        if im_path.split('_')[-1][:2] != 'w3':
            continue
        # todo skimage.io.imread might be the correct read method
        im = plt.imread(im_path)
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
                                    im.std(), size = mask.sum())
        mask[rr, cc] = rand_bg
        im[rr,cc] = mask[rr,cc]
    return im


def reao_roi_or_roiset(impath):

    impath = impath.split('.')[0]
    # if rois are in a zip (image contains multiple rois)
    if path.isfile(join(impath + '_RoiSet.zip')):
        rois = read_roi.read_roi_zip(
            join(impath + '_RoiSet.zip'))
    # if roi is in roi format(image only contains 1 roi)
    elif path.isfile(join(impath + '.roi')):
        rois = read_roi.read_roi_file(join(impath + '.roi'))
    else:
        print("rois couldn't be found for:", impath)

    return rois


class CellTransformData(utils.Dataset):

    def load_cell(self, dataset_dir=dataset_dir, output_dir=
    dataset_dir + 'rescaled/', dim=7680):
        """Load the cell dataset resize the images and rois to a
        uniform dimension.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("cell", 1, "cell")

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
            rois = reao_roi_or_roiset(join(dataset_dir,id))
            cg_keys = [key for key in rois.keys() if 'cell_group' in key]
            for k in cg_keys:
                rois.pop(k)
            im_path = join(dataset_dir, image_name)
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, RoI doesn't include it, so we must read
            # the image. This is only managable since the dataset is tiny.
            h, w = plt.imread(im_path).shape[:2]

            self.add_image(
                "cell",
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
        if image_info["source"] != "cell":
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

    def crop_im_and_mask(self, image_id, crop_dir,
                         dataset_dir, crop_dim=2048):
        # set up the output directory's subfolders
        if not isdir(crop_dir):
            mkdir(crop_dir)
        [mkdir(crop_dir + folder) for folder in ['train', 'val'] if
         folder not in listdir(crop_dir)]

        # If not a confonc dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cell":
            return super(self.__class__, self).load_mask(image_id)

        # get image info and image
        info = self.image_info[image_id]
        im = ds.load_image(image_id)

        # get the indexes of the square crops
        crop_ydim, crop_xdim = crop_dim, crop_dim
        y_num = int(im.shape[0] / crop_ydim)
        x_num = int(im.shape[1] / crop_xdim)
        if im.shape[0] % crop_ydim > im.shape[0]*0.05:
            y_num += 1
        if im.shape[1] % crop_xdim > im.shape[1]*0.05:
            x_num += 1
        idxs = [(i, j) for i in range(y_num) for j in range(x_num)]



        # Randomly choose70% of images to be training data.
        # image_names = [file_name for file_name in listdir(dataset_dir) if
        #                'png' in file_name]
        # np.random.seed(seed=546)
        # train_set = np.random.choice(image_names, int(len(image_names) * 0.7),
        #                              replace=False)

        # randomly choose 70% of the crops to be training data. This way a
        # part of an image will be training data, the other part test data
        # np.random.seed(seed=546)
        train_set = np.random.choice(range(len(idxs)), int(len(idxs) * 0.7),
                                     replace=False)

        # the whole mask can't be loaded into ram (would be ~ 7k x 7k x 200
        # array). So every instance is cropped separately.
        train_image_number, val_image_number = 0, 0
        means = []
        for n, (i, j) in enumerate(idxs):
            # crop image
            y_range = (i * crop_ydim, (i + 1) * crop_ydim)
            x_range = (j * crop_xdim, (j + 1) * crop_xdim)
            crop_id = str(image_id) + '_' + str(i) + '_' + str(j)
            cropped_image = im[y_range[0]:y_range[1], x_range[0]:x_range[1]]

            # crop the 2D masks individually
            mask = None
            for m, (key, vals) in enumerate(info["polygons"].items()):
                print('\r Cropping image {}: {},{} / {},{}. Cropping mask: {}'
                      '/ {}'.format(id, i, j, y_num, x_num, m+1, len(info[
                                                                       "polygons"])),
                      end='. ')
                # Create a mask w/ same shape as image
                mask_ = np.zeros(
                    [info["height"], info["width"], 1],
                    dtype=np.bool)
                # Get indexes of pixels inside the polygon
                rr, cc = skimage.draw.polygon(vals['y'], vals['x'])
                try:
                    mask_[rr, cc, 0] = True
                # the mask coordinates that can be assigned when annotating
                # ranges: 0-width; but if it's exactly width there'll be an
                # index error
                except IndexError:
                    for i, r in enumerate(rr):
                        if r >= info["width"]:
                            rr[i] = info["width"]-1
                    for i, c in enumerate(cc):
                        if c >= info["height"]:
                            cc[i] = info["height"]-1
                    mask_[rr, cc, 0] = True
                # crop the mask
                mask_ = mask_[y_range[0]:y_range[1], x_range[0]:x_range[1], :]

                # if there are no instances in this mask move to next mask
                if mask_.sum(axis=0).sum(axis=0) == 0:
                    continue

                if mask is None:
                    mask = mask_
                else:
                    mask = np.concatenate((mask, mask_), axis=2)

            if mask is None:
                # For now skip image w/ 0 ground truth.
                # https: // github.com / matterport / Mask_RCNN / pull / 1088
                continue

            # When I add th vesicules as 2nd class, here will go the class ids

            # save resized, cropped image and scaled cropped masks
            if np.random.random() < 0.7:
                train_image_number += 1
                output_dir = crop_dir + 'train/'
            else:
                val_image_number += 1
                output_dir = crop_dir + 'val/'

            # set up the output directory for the image crop
            if crop_id not in listdir(output_dir):
                mkdir(output_dir + crop_id)
            output_dir = output_dir + crop_id + '/'

            for m in range(mask.shape[2]):
                mask_ = Image.fromarray((mask[:,:,m]*255).astype(np.uint8),
                                        mode = 'L')
                mask_ = mask_.convert(mode='1')
                mask_.save(output_dir + crop_id + '_mask_' + str(m) + '.png')
            # np.save(output_dir + crop_id + '_mask', cropped_mask)
            means.append(cropped_image.mean())
            cropped_image = Image.fromarray(cropped_image[:, :, 0])
            cropped_image.save(output_dir + crop_id + '.png')
            del (cropped_image, mask)
            gc.collect()
        print('\n Cropping is done. {} images were created({} train, {} val'
              ').'.format(train_image_number + val_image_number,
                          train_image_number, val_image_number))
        print('Average pixel value is: {}'.format(np.array(means).mean()))


def show_im_and_mask(im, mask):
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(im)
    fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()


if __name__ == '__main__':
    # load the dataset
    data_path = src_path.split('src')[0] + 'data/'
    # transfer_w3_channel_images(data_path)
    # nd2 = ND2Reader(data_path + 'arrestin_dots_20200124.nd2')
    # for i, im in enumerate(nd2):
    #     bit8 = convert_to_bit8(im)
    #     png = Image.fromarray(bit8)
    #     png.save(data_path + str(i) + '.png')
    # convert cell groups that can't be annotated to background noise
    for im_path in glob(dataset_dir + '*.png'):
        roi_set = reao_roi_or_roiset(im_path.split('.')[0])
        image = skimage.io.imread(im_path)
        image = cell_groups_to_bg(image, roi_set)
        image = Image.fromarray(image)
        image.save(im_path)

    ds = CellTransformData()
    ds.load_cell()
    ds.prepare()
    for id in ds.image_ids:
        ds.crop_im_and_mask(id, dataset_dir + 'crops/', dataset_dir)

    # ds.crop_im_and_mask(10, dataset_dir + 'low_res/')
    # for id in ds.image_ids:
    #     #print(id)
    #     ds.crop_im_and_mask(id, dataset_dir+'low_res/')
    #     gc.collect()

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
