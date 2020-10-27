"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=imagenet

    # Apply color splash to an image
    python3 cell.py splash --weights=/path/to/weights/file.h5 --image=<URL or
    path to file>

    # Apply color splash to video using the last weights you trained
    python3 cell.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from cell_mrcnn import __file__ as src_path
import matplotlib.pyplot as plt  # skimage.io.imread gives runtimeWarinig
import yaml
from matplotlib import pyplot as plt
from scipy.sparse import bsr_matrix
from imgaug import augmenters as iaa
import tensorflow as tf
import keras

# Root directory of the project
ROOT_DIR = src_path.split('src')[0]

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from cell_mrcnn.config import Config
from cell_mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CellConfig(Config):
    """Configuration for training on the LIDC dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cell"

    # we are working w/ grayscacle images
    IMAGE_CHANNEL_COUNT = 1
    MEAN_PIXEL = np.array([35])

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cell

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200  # NUM OF TRAIN IMAGES

    DETECTION_MIN_CONFIDENCE = 0.8

    # Length of square anchor side in pixels
    # cell are bigger than 32 pixels, but if vesicules are added add 8,
    # 16 scales to the tuple
    # RPN_ANCHOR_SCALES = (64, 128, 256)

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    # RPN proposals w/ an IoU  higher, than threshold get deleted, so lower
    # threshold => more filtered.
    # overlapping cells are common(so you'd increase this), but nucleuses
    # could ne predicted as separate cells (so you'd lower it)
    RPN_NMS_THRESHOLD = 0.8

    # my 1050ti is oom
    BACKBONE = "resnet18"
    DIV_FILTERNB = 2
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 562
    POST_NMS_ROIS_INFERENCE = 1024
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # # Random crops of size
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    # MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 100

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50


class CellInferenceConfig(CellConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"


############################################################
#  Dataset
############################################################

class CellDataset(utils.Dataset):

    def load_cell(self, dataset_dir, subset):
        """Load the cell images.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("cell", 1, "cell")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        subfolders = [os.path.join(dataset_dir, subfolder) for subfolder in
                      os.listdir(dataset_dir)]

        for subfolder in subfolders:
            image_names = [file_name for file_name in os.listdir(subfolder) if
                           'mask' not in file_name]
            # Add images
            for image_name in image_names:
                id = image_name.split('.')[0]
                im_path = subfolder + '/' + image_name
                self.add_image(
                    "cell",
                    image_id=id, # use file name as a unique image id
                    path=im_path)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cell dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cell":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        im_path = self.image_info[image_id]['path']
        subfolder = '/'.join(im_path.split('/')[:-1]) + '/'
        mask_names = [file_name for file_name in os.listdir(subfolder)
                      if 'mask' in file_name]
        mask = None
        for mask_name in mask_names:
            mask_path = subfolder + mask_name
            mask_ = plt.imread(mask_path)
            # add new dim so shape is (512,512,1) the 3rd dim is for different
            # objects
            mask_ = mask_[..., np.newaxis]
            if mask is None:
                mask = mask_
            else:
                mask = np.concatenate((mask, mask_), axis=2)


        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CellDataset()
    # dataset_train.load_cell(args.dataset, "train")
    dataset_train.load_cell(join(ROOT_DIR, 'data/pngs/crops/'), "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellDataset()
    dataset_val.load_cell(join(ROOT_DIR, 'data/pngs/crops/'), "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.Add((-10, 10)),
        iaa.Multiply((0.8, 1.5)),
        # iaa.GaussianBlur(sigma=(0,3.0))
    ])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=5,
    #             augmentation=augmentation,
    #             layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                augmentation=augmentation,
                layers='all')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    config = CellConfig()
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    # weights_path = COCO_WEIGHTS_PATH
    # model.load_weights('/home/miska/repos/cell_mrcnn/logs/cell20201016T0008
    # /mask_rcnn_cell_0010.h5')#, by_name=True, exclude=[
    #     "mrcnn_class_logits", "mrcnn_bbox_fc",
    #     "mrcnn_bbox", "mrcnn_mask", "conv1"])
    train(model)


    # for model debugging
    # import keras
    # model.keras_model.summary()
    # keras.utils.plot_model(model.keras_model, 'mrcnn_resnet18.png',
    #                        show_shapes=True)



    #
    # import argparse
    #
    # # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description='Train Mask R-CNN to detect ct nodules.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'splash'")
    # parser.add_argument('--dataset', required=False,
    #                     metavar="/path/to/cell/dataset/",
    #                     help='Directory of the cell dataset')
    # parser.add_argument('--weights', required=True,
    #                     metavar="/path/to/weights.h5",
    #                     help="Path to weights .h5 file or 'coco'")
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--image', required=False,
    #                     metavar="path or URL to image",
    #                     help='Image to apply the color splash effect on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    # args = parser.parse_args()
    #
    # # Validate arguments
    # if args.command == "train":
    #     assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "splash":
    #     assert args.image or args.video,\
    #            "Provide --image or --video to apply color splash"
    #
    # print("Weights: ", args.weights)
    # print("Dataset: ", args.dataset)
    # print("Logs: ", args.logs)
    #
    # # Configurations
    # if args.command == "train":
    #     config = CtConfig()
    # else:
    #     config = CtInferenceConfig()
    # config.display()
    #
    # # Create model
    # if args.command == "train":
    #     model = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir=args.logs)
    # else:
    #     model = modellib.MaskRCNN(mode="inference", config=config,
    #                               model_dir=args.logs)
    #
    # # Select weights file to load
    # if args.weights.lower() == "coco":
    #     weights_path = COCO_WEIGHTS_PATH
    #     # Download weights file
    #     if not os.path.exists(weights_path):
    #         utils.download_trained_weights(weights_path)
    # elif args.weights.lower() == "last":
    #     # Find last trained weights
    #     weights_path = model.find_last()
    # elif args.weights.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     weights_path = model.get_imagenet_weights()
    # else:
    #     weights_path = args.weights
    #
    # # Load weights
    # print("Loading weights ", weights_path)
    # if args.weights.lower() == "coco":
    #     # Exclude the last layers because they require a matching
    #     # number of classes
    #     model.load_weights(weights_path, by_name=True, exclude=[
    #         "mrcnn_class_logits", "mrcnn_bbox_fc",
    #         "mrcnn_bbox", "mrcnn_mask", "conv1"])
    # else:
    #     model.load_weights(weights_path, by_name=True)
    #
    # # Train or evaluate
    # if args.command == "train":
    #     train(model)
    # elif args.command == "splash":
    #     detect_and_color_splash(model, image_path=args.image,
    #                             video_path=args.video)
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'splash'".format(args.command))