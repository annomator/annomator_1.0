# Python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print("Loading modules...")

import time


# Import first to prevent warnings
# Can solve Mac issues with matplotlib backend (macos) TkAgg (ref)
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements

import numpy as np # for arrays and data manipulation
import os # for os safe file paths
#import six.moves.urllib as urllib # for downloading

#import tarfile # for zip files
import tensorflow as tf
#import zipfile # for zip files

from collections import defaultdict # text - storing
from io import StringIO # text - translating
from matplotlib import pyplot as plt # for image display
from PIL import Image # for image import
#import PIL.ImageDraw as ImageDraw # for quick box visual


import sys
ANNO_REPO_DIR = os.path.join('..', 'anno_repo')
# prev sys.path.append('../', 'anno_repo')
sys.path.append(ANNO_REPO_DIR)
import category_names
category_index = category_names.category_index
import gen_functions
tsf = gen_functions.time_seconds_format
import image_utils
import tf_detections


FROZEN_GRAPH = os.path.join(os.path.abspath('.'), 'frozen_graph', 'frozen_inference_graph.pb')

# # Detection
# Format for linux, windows and mac safe paths
TEST_IMAGES = os.path.join(os.path.abspath('.'), 'test_images')
OUTPUT_DIR = os.path.join(os.path.abspath('.'), 'ouput_text')


TEXT_FORMAT = ".txt" # ".txt" or ".csv"
# Rename after creation with a name meaningful to you.
SUMMARY_PATH = os.path.join(".", "Image_Summary" + TEXT_FORMAT) 


# Note for large datasets or limited space - visuals will be about same size as images
CREATE_VISUAL_IMAGE = True # only available on detection in this version
VISUAL_FORMAT = "image" # "image", ".png", ".jpg", ".pdf"
# Display text results of detection to screen
DISPLAY_TO_SCREEN = False


CONFIDENCE = 0.75
MAX_OBJECTS = 100


########################################################################
# Code
########################################################################
start_time = time.time()

# File management
if not os.path.exists(TEST_IMAGES):
    print("Error: Invalid test image folder path", TEST_IMAGES)
    exit()
if not os.path.exists(FROZEN_GRAPH):
    print("Error: I frozen graph found", FROZEN_GRAPH)
    exit()
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
  
# List test images
test_images = os.listdir(TEST_IMAGES)
# Load detection graph
detection_graph = tf_detections.load_frozen_graph(FROZEN_GRAPH)

# Start Tensorflow Session
with detection_graph.as_default():
    with tf.Session() as session:
        image_summary = []  
        image_count = 0
        for test_image in test_images:
            #image_start = dt.now()
            image_start = time.time()
            image_name, ext = os.path.splitext(test_image)
            if ext != ".jpg" and ext != ".png":
                continue # skip if not jpg or png (could be hidden files, folders or not image)
            image_count+=1

            image_path = os.path.join(TEST_IMAGES, test_image)
            text_path = os.path.join(OUTPUT_DIR, image_name + TEXT_FORMAT)

            # If text exists, collect txt for summary and skip detection
            if os.path.exists(text_path):
                # Read txt file, exclude first row and append to summary
                with open(text_path, "r") as f:
                    line_counter = 0
                    for line in f:
                        line_counter +=1
                        if line_counter > 1: # first line contains headings
                            line = line.strip()
                            image_summary.append(line) # + "\n" # string format
                continue # skip to next image
            # ORIG output_dict = run_inference_for_single_image(image_np, detection_graph)
            # PREV image = Image.open(image_path)
            # PREV image_np = load_image_into_numpy_array(image)
            # PREV output_dict = detect_numpy_for_boxes_session(image_np, session)
            image, output_dict = tf_detections.detect_filepath_for_boxes_session(
                image_path, session)
            
            #  Make text file
            imgX, imgY = image.size
            instances = [] # array
            instance_count = 0
            cat_count = [0] * 256 # quick array index = cat_id
            instance_total = len(output_dict['detection_classes'])
            
            #if CREATE_VISUAL_IMAGE: 
            #  draw = ImageDraw.Draw(image)
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
                cat_count[class_id] +=1
                # TODO python 3 format to 27
                instance_text = "{},{},{},{},{},{},{}".format(
                    test_image, instance_count, class_id, class_name, cat_count[class_id], score, box)
                instances.append(instance_text) # array
                
                if CREATE_VISUAL_IMAGE:
                    # Box
                    #l, r, t, b = box[1] * imgX, box[3] * imgX, box[0] * imgY, box[2] * imgY
                    #draw.line([(l, t), (l, b), (r, b), (r, t), (l, t)], width=2, fill='gray')
                    # Label
                    # prev box_label = str(instance_count) +" "+ class_name +" "+ str(cat_count[class_id])
                    label = str(instance_count) +" "+ class_name +" "+ str(cat_count[class_id])
                    #draw.text((l, t), box_label)
                    # or
                    draw = image_utils.draw_box_and_label_on_image(image, draw, box, label)
            # If no instances met criteria, ie nothing found, return a valid count of 0
            if instance_count == 0:
                instance_text = "{},{}".format(test_image, 0)
                instances.append(instance_text)
            #instance_lines = "\n".join(instances) # array to string
            # Save image text
            image_text_formatted = "ImageName,InstanceId,ClassId,ClassName,ClassCount,Score,Box\n"
            image_text_formatted += "\n".join(instances) # array to string
            with open(text_path, "w") as f:
                f.write(image_text_formatted)
            # Append instances to image summary
            for instance in instances:
                image_summary.append(instance)
              
            if CREATE_VISUAL_IMAGE:
                if VISUAL_FORMAT == "image":
                    visual_file = image_name + "_visual" + ext
                else:
                    visual_file = image_name + "_visual" + VISUAL_FORMAT 
                image.save(os.path.join(OUTPUT_DIR, visual_file))
            
            plt.close('all') # housekeeping
            # Display the results of each image to screen
            if DISPLAY_TO_SCREEN:
                print(image_text_formatted)
            # Timer
            print(image_count, test_image, tsf(time.time() - image_start))

print("Saving new image summary", SUMMARY_PATH)
image_summary_formatted = "ImageName,InstanceId,ClassId,ClassName,ClassCount,Score,Box\n"
image_summary_formatted += "\n".join(image_summary) # array to string
with open(SUMMARY_PATH, "w") as f:
    f.write(image_summary_formatted)

print('-'*50)
print("Processed", image_count, "images.  Total time", tsf(time.time() - start_time))
print('-'*50)
