import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import skimage.io
import cell_mrcnn.model as modellib
import time
from cell_mrcnn import __file__ as path
from os.path import join
import os
from cell_mrcnn import cell
from cell_mrcnn import visualize
from cell_mrcnn.utils import calc_layers
import tensorflow as tf
import matplotlib.pyplot as plt
import base64
ROOT_DIR = path.split('src')[0]
MODEL_DIR = join(ROOT_DIR, "logs")


config = cell.CellInferenceConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# todo: streamlit docs say html could be a security risk. but idk how else
#  to implement download
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

config = InferenceConfig()
config.display()

st.title('Cell prediction, concentric intensity measurement')
images = st.file_uploader("Choose the input images",
                                  accept_multiple_files=True)
# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
# todo: despite  this it still runs on gpu; 2nd time it ran on cpu \_(".")_/
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

analyze = st.button('Analyze')

if analyze:
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    weights_path = join(MODEL_DIR, "cell20201018T1949/mask_rcnn_cell_0024.h5")
    st.write("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)


    avg_df = pd.DataFrame()
    for i, image in enumerate(images):
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
        df = pd.DataFrame(layers).T
        df['mean'] = df.mean(axis=1)
        st.write(st.markdown(get_table_download_link(df),
                             unsafe_allow_html=True))

        st.write(df)

        avg_df[i] = df['mean']
    avg_df['mean'] = avg_df.mean(axis=1)
    st.write(avg_df)
    st.write(st.markdown(get_table_download_link(avg_df),
                         unsafe_allow_html=True))

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(avg_df['mean'])), avg_df['mean'])
    st.pyplot(fig)






