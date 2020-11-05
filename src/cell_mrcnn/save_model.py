import cell_mrcnn.model as modellib
from cell_mrcnn import __file__ as path
from os import mkdir
from os.path import join, isdir
from cell_mrcnn import cell
from cell_mrcnn import visualize
from cell_mrcnn.utils import calc_layers, convert_to_bit8
import tensorflow as tf

#####################
### load the model ##
#####################

ROOT_DIR = path.split('src')[0]
MODEL_DIR = join(ROOT_DIR, "logs")
data_dir = join(ROOT_DIR, 'data')


config = cell.CellInferenceConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.95

config = InferenceConfig()

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
weights_path = join("/home/mrcnn/logs/mask_rcnn_cell_0024.h5")
model.load_weights(weights_path, by_name=True)

######################
#### save the model ##
######################

tf.saved_model.save(model, "/home/mrcnn/models/1")
