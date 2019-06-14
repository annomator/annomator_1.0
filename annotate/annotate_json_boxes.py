# Python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print("Loading modules...")

import time
start_time = time.time()

# Import first to prevent warnings
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements

import numpy as np # for arrays and data manipulation
import os # for os safe file paths

import tensorflow as tf

from collections import defaultdict # text - storing
from io import StringIO #  translating
from matplotlib import pyplot as plt # image display
from PIL import Image # for image import
import PIL.ImageDraw as ImageDraw # for quick box visual

import sys # for folder reference
sys.path.append(os.path.join('..', 'anno_repo'))

import json

import tf_detections
import category_names
category_index = category_names.category_index
import gen_functions
tsf = gen_functions.time_seconds_format
import image_utils

# For more models
# 'http://download.tensorflow.org/models/object_detection/'

FROZEN_GRAPH = os.path.join(os.path.abspath('.'), 'frozen_graph', 'frozen_inference_graph.pb')

# Format for linux, windows and mac safe paths
TEST_IMAGES = os.path.join(os.path.abspath('.'), 'test_images')
OUTPUT_DIR = os.path.join(os.path.abspath('.'), 'ouput_boxes')

JSON_SUMMARY = os.path.join(".", "test_images_summary.json") 
JSON_IMAGE_EXT = "_img"

# Note for large datasets or limited space - visuals will be about same size as images
CREATE_VISUAL_IMAGE = True 
VISUAL_FORMAT = "image" # options: "image", ".png", ".jpg", ".pdf"
# Display results of detection to screen
DISPLAY_TO_SCREEN = False


CONFIDENCE = 0.75
MAX_OBJECTS = 100


########################################################################
# Code
########################################################################


# Error checking and status
if not os.path.exists(TEST_IMAGES):
    print("Error: TEST_IMAGES folder not found", TEST_IMAGES)
    exit()
if not os.path.exists(FROZEN_GRAPH):
    print("Error: FROZEN_GRAPH folder not found", FROZEN_GRAPH)
    exit()

# File management
test_images = os.listdir(TEST_IMAGES)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Load detection graph
detection_graph = tf_detections.load_frozen_graph(FROZEN_GRAPH)

# Start Tensorflow Session
with detection_graph.as_default():
    with tf.Session() as session:
        # For each image
        image_summary = []
        images = []
        instances = []
        image_count = 0
        for test_image in test_images:
            image_start = time.time()
            image_name, ext = os.path.splitext(test_image)
            if ext != ".jpg" and ext != ".png":
                continue # skip if not jpg or png (could be hidden files, folders or not image)
            
            image_count+=1
            image_path = os.path.join(TEST_IMAGES, test_image)
            json_path = os.path.join(OUTPUT_DIR, image_name + JSON_IMAGE_EXT + ".json")
            # If json exists, collect for summary and skip detection
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_image = json.load(f)
                images.append(json_image)
                continue # skip to next image

            # Detect
            image, output_dict = tf_detections.detect_filepath_for_boxes_session(image_path, session)
            imgX, imgY = image.size
            instances = [] 
            instance_count = 0
            cat_count_list = [0] * 256 
            instance_total = len(output_dict['detection_classes'])
            
            # Visual
            draw = ""
            for i in range(instance_total):
                class_id = output_dict['detection_classes'][i]
                if class_id in category_index.keys():
                    class_name = category_index[class_id]['name']
                else:
                    class_name = "NO ID"
                score = output_dict['detection_scores'][i]
                box = output_dict['detection_boxes'][i]
                if instance_count > MAX_OBJECTS:
                    break
                if score < CONFIDENCE:
                    continue
                instance_count +=1
                cat_count_list[class_id] +=1
                
                # Convert to json safe formats
                json_instance = {
                    'instance_id': int(instance_count),
                    'category_id': int(class_id),
                    'category_count': int(cat_count_list[class_id]),
                    'score': float(score),
                    'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])]}
                instances.append(json_instance)
                
                if CREATE_VISUAL_IMAGE:
                    label = str(instance_count) +" "+ str(class_name) +" "+ str(cat_count_list[class_id])
                    draw = image_utils.draw_box_and_label_on_image(image, draw, box, label)
                  
            # If no instances met criteria, ie nothing found, return a valid count of 0
            if instance_count == 0:
                instances.append({'instance_id': 0}) # nothing found in image
            json_image = {'image': test_image, 'width': imgX, 'height': imgY, 'instances': instances}
            images.append(json_image)
            with open(json_path, 'w') as f:
                json.dump(json_image, f)
            if CREATE_VISUAL_IMAGE:
                if VISUAL_FORMAT == "image":
                    visual_file = image_name + "_visual" + ext
                else:
                    visual_file = image_name + "_visual" + VISUAL_FORMAT 
                image.save(os.path.join(OUTPUT_DIR, visual_file))

            # If display to screen, display detections of each image
            plt.close('all')
            if DISPLAY_TO_SCREEN:
                print(json_image)
            print(image_count, test_image, '-', tsf(time.time() - image_start))

# Write json summary
json_summary = {'category_index': category_index, 'images': images}
with open(JSON_SUMMARY, 'w') as f:
    json.dump(json_summary, f)

print('-'*50)
print("Processed", image_count, "images.  Total time", tsf(time.time() - start_time))
print('-'*50)
