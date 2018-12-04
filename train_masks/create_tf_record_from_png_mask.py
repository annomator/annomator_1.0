# Copyright 2018 Annomator Written by Arend Smits
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and limitations under the License.
# Python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import random
import time
start_time = time.time()

# Add sys
import sys
sys.path.append(os.path.join('..', 'anno_repo'))

import gen_functions
tsf = gen_functions.time_seconds_format
import tf_record


# Going to use tf labelmap as must match pipeline config
sys.path.append(os.path.join('..', 'tf_slim_obj_det'))

from object_detection.utils import label_map_util
PATH_TO_LABELS = os.path.join('.', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

COCO_DIR = os.path.join('..', '..', 'coco_panoptic')
IMAGES_DIR = os.path.join(os.path.abspath(COCO_DIR),  'train2017')
MASKS_DIR = os.path.join(os.path.abspath(COCO_DIR), 'png_metric_100_train2017') 
MASK_EXT = '_mask.png' # not used for binary_filename


OUTPUT_DIR = os.path.join(os.path.abspath('.'), 'tf_records')
TRAIN_RECORD_PATH = os.path.join(OUTPUT_DIR, 'train.record')
VAL_RECORD_PATH = os.path.join(OUTPUT_DIR, 'val.record')


# Use binary_filename, offset, centric, metric_100 or metric_offset
CODEC = "metric_100"
CODEC_OFFSET = 100

# Randomize image order
SHUFFLE_IMAGES = True

# All images, split or specify size
SAMPLE_IMAGES = True
# Use a ratio of all
VAL_SPLIT = 0.1
# or specify
IMAGE_TRAIN_NUM = 1000
IMAGE_VAL_NUM = 100

# Ballance categories - subsets of sampled images
SAMPLE_CATEGORIES = True
CATEGORY_SAMPLE_NUM = 100
# Note must have SAMPLE_IMAGES = True and VAL_SPLIT > 0 eg 0.1
CATEGORY_VAL_NUM = 10
# Screen categories
# List of category ids eg [1, 2, 7]
#CATEGORY_INCLUDE = [] # use include list
#CATEGORY_INCLUDE = [1, 16, 17, 18] # eg domestic animals
CATEGORY_EXCLUDE = [] # or exclude list
# Quality control by min pixels
PIXEL_AREA_MIN = 0

# Translation index
# Experimental - Most useful when mixing datasets for training 
# It can interfere with training if used incorrectly
translation_index = {} # default 
#translation_index = {
#    1: {'id': 1, 'name': 'person', 'display_name': 'Person', 'trans_id': 1, 'coco_id': 1},
#    2: {'id': 2, 'name': 'bird', 'display_name': 'Bird', 'trans_id': 2, 'coco_id': 16},
#    3: {'id': 3, 'name': 'cat', 'display_name': 'Cat', 'trans_id': 3, 'coco_id': 17},
#    4: {'id': 4, 'name': 'dog', 'display_name': 'Dog', 'trans_id': 4, 'coco_id': 18}}


##################################################
# Functions
##################################################


""" MOVED TO tf_record (prev tf_train)

def create_tf_example(
    built_dict, category_index,
    image_path, mask_path,
    skipped_masks, skipped_annos,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN, 
    valid_cat_ids, cat_count_list_total):
    

    
    " ""Converts image and annotations to a tf.Example proto.

    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      annotations_list:
        list of dicts with keys:
        [u'segmentation', u'area', u'iscrowd', u'image_id',
        u'bbox', u'category_id', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
      image_dir: directory containing the image files.
      category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    " ""
    # ORIG image_height = image['height']
    # ORIG image_width = image['width']
    # ORIG filename = image['file_name']
    # ORIG image_id = image['id']

    # ORIG full_path = os.path.join(image_dir, filename)
    # Load jpg image

    # Image name used as image id
    image_filename = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_filename)
    image_id = image_name 
    filename = image_filename
    
    # Encode image
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    image_width, image_height = image.size
    # vars
    category_ids = []
    Xmins = []
    Xmaxs = []
    Ymins = []
    Ymaxs = []
    pixel_areas = []
    binary_mask_list = []
    cat_count_list_image = [0] * (max(category_index.keys()) +1)
    # total handed in function
    total_built = len(built_dict['classes'])
    # check
    for i in range(total_built):
        cat_id = int(built_dict['classes'][i])
        area = (built_dict['masks'][i]).sum()
        if SAMPLE_CATEGORIES:
            if cat_id not in valid_cat_ids:
                #print("not valid", cat_id)
                skipped_annos +=1
                continue
            threshold_count = cat_count_list_total[cat_id] + cat_count_list_image[cat_id]
            if threshold_count >= CATEGORY_SAMPLE_NUM:
                #print("over threshold", threshold_count)
                skipped_annos +=1
                continue
            if area < PIXEL_AREA_MIN:
                print("less than area min", area)
                skipped_annos +=1
                continue

        # append
        pixel_areas.append(area)
        
        cat_count_list_image[cat_id] +=1
        category_ids.append(cat_id)
        
        left = built_dict['boxes'][i][1]
        Xmins.append(left)
        right = built_dict['boxes'][i][3]
        Xmaxs.append(right)
        top = built_dict['boxes'][i][0]
        Ymins.append(top)
        bottom = built_dict['boxes'][i][2]
        Ymaxs.append(bottom)

        binary_mask = built_dict['masks'][i]
        binary_image = PIL.Image.fromarray(binary_mask)
        output = io.BytesIO()
        binary_image.save(output, format='PNG')
        binary_mask_list.append(output.getvalue())
    # Return none if none, report, and skip
    if max(cat_count_list_image) == 0:
        skipped_masks +=1
        #print("Skipping mask - No instances found to add", os.path.basename(mask_path))
        return None, cat_count_list_total, skipped_masks, skipped_annos
    # Add to feature dict and return tf exampe
    for vid in valid_cat_ids:
        cat_count_list_total[vid] += cat_count_list_image[vid]
    #for vid in valid_cat_ids:
    #  print(vid, category_index[vid]['name'], cat_count_list_total[vid])

    feature_dict = {
        'image/height': tf_record.int64_feature(image_height),
        'image/width': tf_record.int64_feature(image_width),
        'image/filename': tf_record.bytes_feature(filename.encode('utf8')),
        'image/source_id': tf_record.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256': tf_record.bytes_feature(key.encode('utf8')),
        'image/encoded': tf_record.bytes_feature(encoded_jpg),
        'image/format': tf_record.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': tf_record.float_list_feature(Xmins),
        'image/object/bbox/xmax': tf_record.float_list_feature(Xmaxs),
        'image/object/bbox/ymin': tf_record.float_list_feature(Ymins),
        'image/object/bbox/ymax': tf_record.float_list_feature(Ymaxs),
        'image/object/class/label': tf_record.int64_list_feature(category_ids),
        'image/object/area': tf_record.float_list_feature(pixel_areas),
        'image/object/mask': tf_record.bytes_list_feature(binary_mask_list)}
    
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    return tf_example, cat_count_list_total, skipped_masks, skipped_annos
        
    # removed 'image/object/is_crowd': int64_list_feature(is_crowd),
    # removed key as not used when returned
  
def create_tf_record_from_png_masks(
    CODEC, CODEC_OFFSET, category_index,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN, valid_cat_ids, 
    images, IMAGES_DIR, MASKS_DIR, MASK_EXT,
    output_path):
    # Finds masks for a list of images
    print('Writing {}'.format(os.path.basename(output_path)))
    images_start = time.time()
    image_count = 0
    mask_count = 0
    skipped_images = 0
    skipped_masks = 0
    skipped_annos = 0
    cat_count_list_total = [0] * (max(valid_cat_ids) + 1)
    image_avg = 0
    built_dict = {}
    
    writer = tf.python_io.TFRecordWriter(output_path)
    for image in images:
        image_name, image_ext = os.path.splitext(image)
        if image_ext != ".jpg":
            # can store png masks and visuals 'all-in-one' with simple skip
            continue # only setup for jpgs
        
        image_path = os.path.join(IMAGES_DIR, image)
        image_count +=1
        mask_found = False
        if CODEC == "binary_filename":
            mask_path = (os.path.join(MASKS_DIR, image_name))
            #print(mask_path)
            if os.path.isdir(mask_path):
                mask_count +=1
                mask_found = True
                built_dict = png_masks.rebuild_from_binary_mask_dir(mask_path)

        else: # CODEC == "offset" or CODEC == "centric" or CODEC == "metric_100" or CODEC == "metric_offset":
            mask_path = os.path.join(MASKS_DIR, image_name + MASK_EXT)
            if os.path.isfile(mask_path):
                mask_count +=1
                mask_found = True
                # load mask_np and get built dict for image
                mask = PIL.Image.open(mask_path)
                mask_np = np.array(mask).astype(np.uint8)
                built_dict = png_masks.rebuild_from_mask(
                    mask_np, CODEC, CODEC_OFFSET, category_index)

        if not mask_found or len(built_dict) == 0:
            skipped_images +=1
            print('Skipping image: no mask found for', image)
            continue
        
        tf_example, cat_count_list_total, skipped_masks, skipped_annos = create_tf_example(
            built_dict, category_index, 
            image_path, mask_path,
            skipped_masks, skipped_annos,
            SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN, 
            valid_cat_ids, cat_count_list_total)

        if tf_example != None:
            writer.write(tf_example.SerializeToString())
        if SAMPLE_CATEGORIES:
            finished_sampling = True
            for vid in valid_cat_ids:
                if cat_count_list_total[vid] < CATEGORY_SAMPLE_NUM:
                    finished_sampling = False
            if finished_sampling:
                for vid in valid_cat_ids:
                    print(vid, category_index[vid]['name'], cat_count_list_total[vid])
                print("Category sampling complete - quoto achieved")
                print('-'*50)
                break
        
        # Reporting
        if image_count % 10 == 0:
            image_time_total = time.time() - images_start
            image_time_mean = image_time_total / image_count
            if SAMPLE_CATEGORIES:
                for vid in valid_cat_ids:
                    print(vid, category_index[vid]['name'], cat_count_list_total[vid])
            #TODO py27 tsf
            #print('Images {} Average {:0.03f} seconds'.format(image_count, image_time_mean))
            print('Images %d Average %s' % (image_count, tsf(image_time_mean)))
        
    writer.close()
    #TODO py27
    #print('{} images skipped as no mask'.format(skipped_images))
    #print('{} masks skipped due to sampling'.format(skipped_masks))
    #print('{} annotations skipped due to sampling'.format(skipped_annos))
    #print('Completed {} images and {} masks'.format(image_count, mask_count))
    print('%d images skipped as no mask' % (skipped_images))
    print('%d masks skipped due to sampling' % (skipped_masks))
    print('%d annotations skipped due to sampling' % (skipped_annos))
    print('Completed %d images and %d masks' % (image_count, mask_count))
    print('-'*50)
"""


##################################################
# Code
##################################################


if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
  
images = os.listdir(IMAGES_DIR)
train_images = []
val_images = []
#max_cat_id = max(category_index.keys()) + 1
#total_cat_count_list = [0] * max_cat_id # max cat_id



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
    #random.seed(128) # replicate split with seed
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

# Create tf records
tf_record.create_tf_record_from_png_masks(
    CODEC, CODEC_OFFSET, category_index,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN,
    valid_cat_ids, translation_index, 
    train_images, IMAGES_DIR, MASKS_DIR, MASK_EXT, 
    TRAIN_RECORD_PATH)

if SAMPLE_IMAGES or SAMPLE_CATEGORIES:
    tf_record.create_tf_record_from_png_masks(
        CODEC, CODEC_OFFSET, category_index,
        SAMPLE_CATEGORIES, CATEGORY_VAL_NUM, PIXEL_AREA_MIN,
        valid_cat_ids, translation_index, 
        val_images, IMAGES_DIR, MASKS_DIR, MASK_EXT, 
        VAL_RECORD_PATH)


print('Finished', tsf(time.time() - start_time))
print('-'*50)

