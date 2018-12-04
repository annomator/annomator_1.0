# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Annomator
Copyright 2018 Arend Smits.
All rights reserved.  MIT Licence.  
"""


# Python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import json

import random # for train/val split

import time
start_time = time.time()

# Add sys
import sys
sys.path.append(os.path.join('..', 'anno_repo'))

import gen_functions
tsf = gen_functions.time_seconds_format
import tf_record

sys.path.append(os.path.join('..', 'tf_slim_obj_det'))
from object_detection.utils import label_map_util
# Going to use pbtxt for congruency with pipeline.config
PATH_TO_LABELS = os.path.join('.', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Translation index
# Experimental - Most useful when mixing datasets for training 
# It can interfere with training if used incorrectly
#translation_index = {
#    1: {'id': 1, 'name': 'person', 'display_name': 'Person', 'trans_id': 1, 'coco_id': 1},
#    2: {'id': 2, 'name': 'bird', 'display_name': 'Bird', 'trans_id': 2, 'coco_id': 16},
#    3: {'id': 3, 'name': 'cat', 'display_name': 'Cat', 'trans_id': 3, 'coco_id': 17},
#    4: {'id': 4, 'name': 'dog', 'display_name': 'Dog', 'trans_id': 4, 'coco_id': 18}}
translation_index = {}

COCO_DIR = os.path.join('..', '..', 'coco_panoptic')
IMAGES_DIR = os.path.join(COCO_DIR,  'train2017')

OUTPUT_DIR = os.path.abspath(os.path.join('.', 'tf_records'))
TRAIN_RECORD_PATH = os.path.join(OUTPUT_DIR, 'train.record')
VAL_RECORD_PATH = os.path.join(OUTPUT_DIR, 'val.record')

JSON_PATH = os.path.join(COCO_DIR, 'panoptic_train2017.json')

# Randomize image order
SHUFFLE_IMAGES = True

# All images, split or specify size
SAMPLE_IMAGES = True
# Use a ratio of all
VAL_SPLIT = 0.1 # NOTE setup for 10% val split
# or specify
IMAGE_TRAIN_NUM = 1000
IMAGE_VAL_NUM = 100

# Ballance categories
SAMPLE_CATEGORIES = True
CATEGORY_SAMPLE_NUM = 1000
CATEGORY_VAL_NUM = 100
# Screen categories
# List of category ids eg [1, 2, 7]
#CATEGORY_INCLUDE = [] # use include list
CATEGORY_EXCLUDE = [] # or exclude list

# Quality control by min pixels
PIXEL_AREA_MIN = 0


##################################################
# Code
##################################################


if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# load json
with open(JSON_PATH, 'r') as f:
    json_loaded = json.load(f)
    

# Check all json refs have an image file and prepare split list
sub_start = time.time()
images = []
image_ids = []
heights = []
widths = []
image_count = 0
skipped_images = 0

for image in json_loaded['images']:
    image_count +=1
    image_path = os.path.join(IMAGES_DIR, image['file_name'])
    if os.path.exists(image_path):
        images.append(image['file_name'])
        image_ids.append(image['id'])
        heights.append(image['height'])
        widths.append(image['width'])                     
    else:
        skipped_images +=1
        print("No image found for json reference", image['file_name'])
    if image_count % 10000 == 0:
        print("Checking json", image_count, tsf(time.time() - sub_start))

train_images = []
val_images = []
max_cat_id = max(category_index.keys()) + 1
total_cat_count_list = [0] * max_cat_id 


if SAMPLE_CATEGORIES:
    valid_cat_ids = []
    for key in category_index:
        if key in CATEGORY_EXCLUDE:
            continue
        if len(CATEGORY_INCLUDE) > 0:
            if key not in CATEGORY_INCLUDE:
                continue
        valid_cat_ids.append(key)
else:
    valid_cat_ids = category_index.keys()


if SHUFFLE_IMAGES:
    #random.seed(100) # replicate split with seed
    random.shuffle(images)

if SAMPLE_IMAGES:
    if VAL_SPLIT > 0:
        num_images = len(images)
        num_val = int(VAL_SPLIT * num_images)
        val_images = images[:num_val]
        train_images = images[num_val:]
    else:
        val_images = images[:IMAGE_VAL_NUM]
        train_images = images[IMAGE_VAL_NUM:IMAGE_TRAIN_NUM]
else:
    train_images = images

print('%d training and %d validation images' % (len(train_images), len(val_images)))

train_image_count = 0
val_image_count = 0

# Create tf records
train_image_count = tf_record.create_tf_record_from_json_panoptic(
    json_loaded, category_index,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN,
    valid_cat_ids, translation_index, 
    train_images, IMAGES_DIR, image_ids, heights, widths, 
    TRAIN_RECORD_PATH)

if SAMPLE_IMAGES:
    val_image_count = tf_record.create_tf_record_from_json_panoptic(
        json_loaded, category_index,
        SAMPLE_CATEGORIES, CATEGORY_VAL_NUM, PIXEL_AREA_MIN,
        valid_cat_ids, translation_index, 
        val_images, IMAGES_DIR, image_ids, heights, widths, 
        VAL_RECORD_PATH)

print('Finished', tsf(time.time() - start_time))
print('Images skipped as no image found for json reference', skipped_images)
print('%d train images %d validation images' % (train_image_count, val_image_count))
print('-'*50)


