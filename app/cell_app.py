# 1st part Upload
# todo: convert images to 8bit, (and png?)
# todo: save converted images to data/smthg folder
# todo: allow filtering of images through tags
# 2nd Analyze
# todo: list the selected images, select a group name 4 these,
#  save these to a folder (called the group name)
# todo: with a checklist be able to select certain images w/ tags
# todo: predict more images at once, save the masks,
# todo: create a csvs for each group, calc mean for each layer in
#  the same group, download w/ a link
# todo: try to cache model or weights to speed up the process
# 3rd Explore
# todo: be able to explore the result: ...
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import skimage.io
import cell_mrcnn.model as modellib
import time
from cell_mrcnn import __file__ as path
from os import mkdir
from os.path import join, isdir
from cell_mrcnn import cell
from cell_mrcnn import visualize
from cell_mrcnn.utils import calc_layers, convert_to_bit8
import tensorflow as tf
import matplotlib.pyplot as plt
import base64
import time
from PIL import Image
import glob


ROOT_DIR = path.split('src')[0]
MODEL_DIR = join(ROOT_DIR, "logs")
data_dir = join(ROOT_DIR, 'data')


if tf.test.gpu_device_name():

    st.write('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:

    st.write("Please install GPU version of TF")

config = cell.CellInferenceConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.95

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

def generate_image_paths(data_directory, group_tag, channel_tags):
    """
    Generates paths to the images that are in the given group and are the
    given channels
    :param data_dir:
    :param groups:
    :param channels:
    :return:
    """

    paths = []
    for channel in channel_tags:
        paths.extend(glob.glob(join(data_directory,group_tag)+'/*'+channel+'*'))

    return paths
config = InferenceConfig()
config.display()
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"

st.title('Cell prediction, concentric intensity measurement')

groups_tags = ['wt', 'Q63R', 'L133i', 'Cer']
selected_groups = st.multiselect('Select groups to be analyzed', groups_tags)
channels = ['w1','w2','w3']
mask_image_channels = st.selectbox('Select channel(s) for mask prediction',
                              channels)
layers_image_channels = st.selectbox("Select channel(s) for concentric "
                                      "intensity measurement", channels)
# todo: w/ all these nested dict, it might be better to use a Class to hold
#  these values
if mask_image_channels and layers_image_channels:
    st.write('Loading images...')
    image_dict = {}
    # keys: groups / keys: postions / values: [mask_image, layer_image]
    for group in selected_groups:
        image_dict[group] = {}
# todo: it might ba better to only store the image paths, and load the
#  images as needed to save memory
        mask_image_paths = generate_image_paths(data_dir, group, mask_image_channels)
        layers_image_paths = generate_image_paths(data_dir, group, layers_image_channels)
        for mask_image_path in mask_image_paths:
            position = mask_image_path.split('_')[-2]
            mask_image = skimage.io.imread(mask_image_path)
            for i, layers_image_path in enumerate(layers_image_paths):
                layer_position = layers_image_path.split('_')[-2]
                if layer_position == position:
                    layers_image = skimage.io.imread(layers_image_path)
                    layers_image_paths.pop(i)


            if mask_image.ndim < 3:
                mask_image = mask_image[..., np.newaxis]
            if mask_image.dtype != np.uint8:
                mask_image = convert_to_bit8(mask_image)

            if layers_image.ndim < 3:
                layers_image = layers_image[..., np.newaxis]

            image_dict[group][position] = [mask_image]
            image_dict[group][position].append(layers_image)

    st.write('Done!')

    analyze = st.button('Analyze and save results')

    if analyze:
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)
        weights_path = join(
            "/home/mrcnn/logs/cell20201123T1456/mask_rcnn_cell_0017.h5")
        st.write("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True)

        n_images = 0
        for group in image_dict.keys():
            n_images += len(image_dict[group])

        result_dict = {}
        st.write('Predicting masks...')
        progressbar = st.progress(0)
        i = 0
        for group in image_dict.keys():
            result_dict[group] = {}
            for pos, image_list in image_dict[group].items():
                result = model.detect([image_list[0]], verbose=0)[0]
                result_dict[group][pos] = result
                i += 1
                progressbar.progress(i / n_images)


        df_dic = {}
        st.write('Calculating layers...')
        progressbar = st.progress(0)
        i = 0
        for group in image_dict.keys():
            df_dic[group] = pd.DataFrame()
            for pos, image_list in image_dict[group].items():
                r = result_dict[group][pos]
                layers = calc_layers(image_list[1], r['masks'])
                df = pd.DataFrame(layers).T
                df_dic[group] = pd.concat([df_dic[group], df], axis=1, ignore_index = True)
                i += 1
                progressbar.progress(i / n_images)

            df_dic[group]['mean'] = df_dic[group].mean(axis=1)
            st.write(group)
            st.write(df_dic[group])
            st.write(st.markdown(get_table_download_link(df_dic[group]),
                                 unsafe_allow_html=True))

            fig, ax = plt.subplots()
            # todo: use lineplot (ax.plot)
            ax.bar(np.arange(len(df_dic[group]['mean'])), df_dic[group]['mean'])
            st.pyplot(fig)

        # save the results
        results_dir = join(ROOT_DIR, 'results')
        if not isdir(results_dir):
            mkdir(results_dir)
        for group in result_dict.keys():
            group_dir = join(results_dir, group)
            if not isdir(group_dir):
                mkdir(group_dir)
            with open(join(group_dir, 'results.csv'), 'w') as f:
                f.write(df_dic[group].to_csv())
            for pos, r in result_dict[group].items():
                for m in range(r['masks'].shape[2]):
                    mask_ = Image.fromarray((r['masks'][:,:,m]*255).astype(np.uint8),
                                            mode = 'L')
                    mask_ = mask_.convert(mode='1')
                    mask_.save(join(group_dir, pos + '_mask_' + str(m) +
                               '.png'))
