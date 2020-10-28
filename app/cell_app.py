import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import skimage.io
import cell_mrcnn.model as modellib
import time
from cell_mrcnn import __file__ as path
import os
from cell_mrcnn import cell
from cell_mrcnn import visualize
from cell_mrcnn.utils import calc_layers
import tensorflow as tf
ROOT_DIR = path.split('src')[0]
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


config = cell.CellInferenceConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

st.title('Cell prediction, concentric intensity measurement')
images = st.file_uploader("Choose the input images",
                                  accept_multiple_files=True)
# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
# todo: despite  this it still runs on gpu
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
weights_path = "/home/mrcnn/logs/mask_rcnn_cell_0024.h5"
st.write("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

for image in images:
    image = skimage.io.imread(image)
    # If grayscale
    # add new dim so shape goes from (512,512) -> (512,512,1); and then
    # at model building i think it'll become (1,512,512,1). Other option is
    # to change the whole keras model which seems like more complicated

    if image.ndim < 3:
        image = image[..., np.newaxis]
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]

    results = model.detect([image], verbose=0)
    r = results[0]
    fig = visualize.display_instances(skimage.color.gray2rgb(image[:,:,0]),
                                 r['rois'], r['masks'],
                                r['class_ids'], ['cell']*len(r['class_ids']),r['scores'])
    st.pyplot(fig)
    layers = calc_layers(image, r['masks'])
    try:
        st.write(layers)
    except:
        st.write(str(layers))
