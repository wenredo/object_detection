import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops
from utils.non_max_suppression import non_max_suppression
from utils.drawing_boxes_tools import draw_boxes
from utils.letter_box import letter_box_image
from utils import label_map_util
from utils import visualization_utils as vis_util

# Tensorflow warning
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


# What model to download.
MODEL_NAME = 'yolov3_608'
INPUT_SIZE = 608

MODEL_FILE = MODEL_NAME + '.tar.gz'


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Download model if necessary
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())
# gpu options
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# python dictionary formed by the integer index and the string of the category
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def load_cv_image_into_numpy_array(image):
    (im_width, im_height, _) = image.shape
    return np.array(image).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def load_coco_names_no_multiline(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            new_name = name.split('\n')
            names[id] = new_name[0]
    return names

# Load Classes file
classes = load_coco_names_no_multiline(os.path.join('data', 'coco.names'))

# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 6)]

# video detection
PATH_TO_VIDEOS_DIR = 'test_videos'
PATH_TO_TEST_VIDEOS_DIR = 'test'
PATH_TO_RES_VIDEOS_DIR = 'res'
TEST_VIDEO_PATHS = [video_filename for video_filename in os.listdir(os.path.join(PATH_TO_VIDEOS_DIR, PATH_TO_TEST_VIDEOS_DIR))]

# # Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_video_yolo_v3(video, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            input_tensor = tf.get_default_graph().get_tensor_by_name('inputs:0')
            output_tensor = tf.get_default_graph().get_tensor_by_name('output_boxes:0')

            # Load Video with opencv
            INPUT_VIDEO = os.path.join(PATH_TO_VIDEOS_DIR,PATH_TO_TEST_VIDEOS_DIR, video)
            #####################################################################################
            # open video handle
            #####################################################################################
            cap = cv2.VideoCapture(INPUT_VIDEO)

            #####################################################################################
            # Prepare for saving the detected video
            #####################################################################################
            sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fourcc = cv2.VideoWriter_fourcc(*'mpeg')
            vout = cv2.VideoWriter()
            vout.open(os.path.join(PATH_TO_VIDEOS_DIR, PATH_TO_RES_VIDEOS_DIR, video), fourcc, 20, sz, True)

            #####################################################################################
            # open video handle
            #####################################################################################
            cap = cv2.VideoCapture(INPUT_VIDEO)

            while cap.isOpened():
                ret, image = cap.read()

                if ret:
                    # shape = image.shape
                    # image_np = load_cv_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
                    img_resized = letter_box_image(pil_image, INPUT_SIZE, INPUT_SIZE, 128)
                    img_resized = img_resized.astype(np.float32)
                    image_np_expanded = np.expand_dims(img_resized, axis=0)
                    output_dict = sess.run(output_tensor,
                                           feed_dict={input_tensor: image_np_expanded})

                    filtered_boxes = non_max_suppression(output_dict,
                                                         confidence_threshold=0.75,
                                                         iou_threshold=0.1)

                    draw_boxes(filtered_boxes, pil_image, classes, (INPUT_SIZE, INPUT_SIZE), True, True)

                    # vis_util.visualize_boxes_and_labels_on_image_array(
                    #     image,
                    #     output_dict['detection_boxes'],
                    #     output_dict['detection_classes'],
                    #     output_dict['detection_scores'],
                    #     category_index,
                    #     instance_masks=output_dict.get('detection_masks'),
                    #     use_normalized_coordinates=True,
                    #     line_thickness=8)

                    # Save the video frame by frame
                    array_image = np.array(pil_image)
                    cv2.imshow('pred',array_image)
                    vout.write(array_image)

                    # show image
                    # cv2.imshow("detection", array_image)
                    # if cv2.waitKey(110) & 0xff == 27:
                    #     break

                else:
                    cv2.destroyAllWindows()
                    break

            vout.release()
            cap.release()


for video in TEST_VIDEO_PATHS:
    run_inference_for_video_yolo_v3(video,detection_graph)

cv2.destroyAllWindows()




