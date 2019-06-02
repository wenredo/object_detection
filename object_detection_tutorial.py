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
from utils.non_max_suppression import non_max_suppression
from utils.drawing_boxes_tools import draw_boxes
from utils.letter_box import letter_box_image
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'yolov3_416'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Download model
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())

########################################################################################################################
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
########################################################################################################################


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


classes = load_coco_names_no_multiline(os.path.join('data', 'coco.names'))

# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_IMAGES_DIR = 'test_images'
PATH_TO_TEST_IMAGES_DIR = 'test'
PATH_TO_RES_IMAGES_DIR = 'res'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 6)]

# video detection
PATH_TO_VIDEOS_DIR = 'test_videos'
PATH_TO_TEST_VIDEOS_DIR = 'test'
PATH_TO_RES_VIDEOS_DIR = 'res'
TEST_VIDEO_PATHS = [video_filename for video_filename in
                    os.listdir(os.path.join(PATH_TO_VIDEOS_DIR, PATH_TO_TEST_VIDEOS_DIR))]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image_ssd(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict



def run_inference_for_single_video_ssd(graph, video):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            # INPUT TENSOR OF DETECTION GRAPH
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Load Video with opencv
            INPUT_VIDEO = os.path.join(PATH_TO_VIDEOS_DIR, PATH_TO_TEST_VIDEOS_DIR,video)
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

            while cap.isOpened():
                ret, image = cap.read()

                if ret:
                    # shape = image.shape
                    # image_np = load_cv_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    #pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
                    #img_resized = letter_box_image(pil_image, int(416), int(416), 128)
                    #img_resized = img_resized.astype(np.float32)
                    image_np_expanded = np.expand_dims(image, axis=0)
                    # OBTAIN OUTPUT TENSOR
                    # Run inference
                    output_dict = sess.run(tensor_dict,
                                           feed_dict={image_tensor: image_np_expanded})

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.int64)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        output_dict['detection_masks'] = output_dict['detection_masks'][0]


                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=3)

                    # Save the video frame by frame
                    vout.write(image)

                    # show_image = plt.figure()
                    cv2.imshow("detection", image)
                    # show_image.show()
                    # image_np.show()
                    if cv2.waitKey(110) & 0xff == 27:
                        break

                else:
                    break

            vout.release()
            cap.release()




def run_inference_for_video_yolo_v3(video, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            image_tensor = tf.get_default_graph().get_tensor_by_name('inputs:0')
            output_tensor = tf.get_default_graph().get_tensor_by_name('output_boxes:0')

            # Load Video with opencv
            INPUT_VIDEO = os.path.join(PATH_TO_VIDEOS_DIR, PATH_TO_TEST_VIDEOS_DIR, video)
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
            vout.open(os.path.join(PATH_TO_VIDEOS_DIR, PATH_TO_TEST_VIDEOS_DIR, video), fourcc, 20, sz, True)

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
                    img_resized = letter_box_image(pil_image, int(416), int(416), 128)
                    img_resized = img_resized.astype(np.float32)
                    image_np_expanded = np.expand_dims(img_resized, axis=0)
                    output_dict = sess.run(output_tensor,
                                           feed_dict={image_tensor: image_np_expanded})

                    filtered_boxes = non_max_suppression(output_dict,
                                                         confidence_threshold=0.5,
                                                         iou_threshold=0.4)

                    draw_boxes(filtered_boxes, pil_image, classes, (416, 416), True, True)

                    # Save the video frame by frame
                    array_image = np.array(pil_image)
                    vout.write(array_image)

                    # show_image = plt.figure()
                    cv2.imshow("detection", array_image)
                    # show_image.show()
                    # image_np.show()
                    if cv2.waitKey(110) & 0xff == 27:
                        break

                else:
                    break

            vout.release()
            cap.release()


def run_inference_for_image_yolo_v3(graph, image):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            image_tensor = tf.get_default_graph().get_tensor_by_name('inputs:0')
            output_tensor = tf.get_default_graph().get_tensor_by_name('output_boxes:0')

            # Load Video with opencv
            INPUT_IMAGE = os.path.join(PATH_TO_IMAGES_DIR, PATH_TO_TEST_IMAGES_DIR, image)
            OUTPUT_IMAGE = os.path.join(PATH_TO_IMAGES_DIR, PATH_TO_RES_IMAGES_DIR, image)
            #####################################################################################
            # open video handle
            #####################################################################################
            img = cv2.imread(INPUT_IMAGE)

            # shape = image.shape
            # image_np = load_cv_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            pil_image = Image.fromarray(img.astype('uint8'), 'RGB')
            img_resized = letter_box_image(pil_image, int(416), int(416), 128)
            img_resized = img_resized.astype(np.float32)
            image_np_expanded = np.expand_dims(img_resized, axis=0)
            output_dict = sess.run(output_tensor,
                                   feed_dict={image_tensor: image_np_expanded})

            filtered_boxes = non_max_suppression(output_dict,
                                                 confidence_threshold=0.5,
                                                 iou_threshold=0.4)

            draw_boxes(filtered_boxes, pil_image, classes, (416, 416), True, True)

            # Save the video frame by frame
            array_image = np.array(pil_image)

            # show_image = plt.figure()
            #cv2.imshow("detection", array_image)
            cv2.imwrite(OUTPUT_IMAGE,array_image)
            # show_image.show()
            # image_np.show()
            # if cv2.waitKey(110) & 0xff == 27:
            #     break



# for image_path in TEST_IMAGE_PATHS:
#     image = Image.open(image_path)
#     # the array based representation of the image will be used later in order to prepare the
#     # result image with boxes and labels on it.
#     image_np = load_image_into_numpy_array(image)
#     # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#     image_np_expanded = np.expand_dims(image_np, axis=0)
#     # Actual detection.
#     output_dict = run_inference_for_single_image_ssd(image_np_expanded, detection_graph)
#     # Visualization of the results of a detection.
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
#         category_index,
#         instance_masks=output_dict.get('detection_masks'),
#         use_normalized_coordinates=True,
#         line_thickness=8)
#     # plt.figure(figsize=IMAGE_SIZE)
#     # plt.imshow(image_np,aspect="auto")
#     show_image = plt.figure()
#     plt.imshow(image_np)
#     # show_image.show()
#     # image_np.show()
#     continue
#     # cv2.imshow('image_np',image_np)
#     # if cv2.waitKey(25) & 0xFF == ord('q'):
#     #     cv2.destroyAllWindows()

if __name__ == '__main__':
    for image in os.listdir('test_images/test'):
        run_inference_for_image_yolo_v3(detection_graph, image)
