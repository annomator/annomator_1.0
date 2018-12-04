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

import hashlib
import io
import os
import numpy as np
import PIL.Image
import tensorflow as tf

#import json # NOT NEEDED
#from pycocotools import mask # NOT NEEDED

import time 
import gen_functions
tsf = gen_functions.time_seconds_format
import png_masks


# As specific io format, do not use image utils

# From TFOD/utils/dataset_utils.py
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def create_tf_example_for_boxes_area(
    boxes_dict, category_index, image_path, image_height,
    image_width, image_id, skipped_images, skipped_annos,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN, 
    valid_cat_ids, translation_index, cat_count_list_total):
    
    # Image name used as image id
    image_filename = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_filename) 
    filename = image_filename
    
    # Encode image
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    # vars
    category_ids = []
    Xmins = []
    Xmaxs = []
    Ymins = []
    Ymaxs = []
    pixel_areas = []
    cat_count_list_image = [0] * (max(category_index.keys()) +1)
    total_annos = len(boxes_dict['cat_ids'])
    # check
    for i in range(total_annos):
        cat_id = boxes_dict['cat_ids'][i]
        area = boxes_dict['areas'][i]
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
                #print("less than area min", area)
                skipped_annos +=1
                continue

        # append
        pixel_areas.append(area)
        cat_count_list_image[cat_id] +=1
        
        if translation_index == {}:
            category_ids.append(cat_id)
        else:
            for tid in translation_index.keys():
                if translation_index[tid]['coco_id'] == cat_id:
                    category_ids.append(tid)
        
        # Translate coco xywh pixels to yxyx norm
        left = boxes_dict['boxes'][i][0] / image_width
        Xmins.append(left)
        right = left + (boxes_dict['boxes'][i][2] / image_width)
        Xmaxs.append(right)
        top = boxes_dict['boxes'][i][1] / image_height
        Ymins.append(top)
        bottom = top + (boxes_dict['boxes'][i][3] / image_height)
        Ymaxs.append(bottom)
        
    # Return none if none, report, and skip
    if max(cat_count_list_image) == 0:
        skipped_images +=1
        #print("Skipping mask - No instances found to add", os.path.basename(mask_path))
        return None, cat_count_list_total, skipped_images, skipped_annos
    # Add to feature dict and return tf exampe
    for vid in valid_cat_ids:
        cat_count_list_total[vid] += cat_count_list_image[vid]

    feature_dict = {
        'image/height': int64_feature(image_height),
        'image/width': int64_feature(image_width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(Xmins),
        'image/object/bbox/xmax': float_list_feature(Xmaxs),
        'image/object/bbox/ymin': float_list_feature(Ymins),
        'image/object/bbox/ymax': float_list_feature(Ymaxs),
        'image/object/class/label': int64_list_feature(category_ids),
        'image/object/area': float_list_feature(pixel_areas)}
    
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    return tf_example, cat_count_list_total, skipped_images, skipped_annos
    # removed 'image/object/mask': bytes_list_feature(binary_mask_list)
    # removed 'image/object/is_crowd': int64_list_feature(is_crowd),
    # removed key as not used when returned


def create_tf_example_for_boxes_score(
    boxes_dict, category_index, image_path, image_height,
    image_width, skipped_images, skipped_annos,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, 
    valid_cat_ids, translation_index, cat_count_list_total):

    # Image name used as image id
    image_filename = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_filename)
    #old image_id = image_name 
    filename = image_filename
    
    # Encode image
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    # vars
    category_ids = []
    Xmins = []
    Xmaxs = []
    Ymins = []
    Ymaxs = []
    scores = []
    
    cat_count_list_image = [0] * (max(category_index.keys()) +1)
    
    total_annos = len(boxes_dict['cat_ids'])
    # check
    for i in range(total_annos):
        cat_id = boxes_dict['cat_ids'][i]
        score = boxes_dict['scores'][i]
        if SAMPLE_CATEGORIES:
            if cat_id not in valid_cat_ids:
                skipped_annos +=1
                continue
            threshold_count = cat_count_list_total[cat_id] + cat_count_list_image[cat_id]
            if threshold_count >= CATEGORY_SAMPLE_NUM:
                #print("over threshold", threshold_count)
                skipped_annos +=1
                continue

        scores.append(score)
        
        cat_count_list_image[cat_id] +=1
        
        if translation_index == {}:
            category_ids.append(cat_id)
        else:
            for tid in translation_index.keys():
                if translation_index[tid]['coco_id'] == cat_id:
                    category_ids.append(tid)
                    
        # Translate detection box from yxyx
        left = boxes_dict['boxes'][i][1]
        Xmins.append(left)
        right = boxes_dict['boxes'][i][3]
        Xmaxs.append(right)
        top = boxes_dict['boxes'][i][0]
        Ymins.append(top)
        bottom = boxes_dict['boxes'][i][2]
        Ymaxs.append(bottom)

    # Return none if none, report, and skip
    if max(cat_count_list_image) == 0:
        skipped_images +=1
        #print("Skipping mask - No instances found to add", os.path.basename(mask_path))
        return None, cat_count_list_total, skipped_images, skipped_annos
    # Add to feature dict and return tf exampe
    for vid in valid_cat_ids:
        cat_count_list_total[vid] += cat_count_list_image[vid]

    feature_dict = {
        'image/height': int64_feature(image_height),
        'image/width': int64_feature(image_width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(Xmins),
        'image/object/bbox/xmax': float_list_feature(Xmaxs),
        'image/object/bbox/ymin': float_list_feature(Ymins),
        'image/object/bbox/ymax': float_list_feature(Ymaxs),
        'image/object/class/label': int64_list_feature(category_ids),
        'image/object/score': float_list_feature(scores)}
    # removed 'image/source_id': bytes_feature(str(image_id).encode('utf8')),
    
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    return tf_example, cat_count_list_total, skipped_images, skipped_annos
    # removed 'image/object/mask': bytes_list_feature(binary_mask_list)
    # removed 'image/object/is_crowd': int64_list_feature(is_crowd),
    # removed key as not used when returned
  
def create_tf_record_from_json_panoptic(
    json_loaded, category_index,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN,
    valid_cat_ids, translation_index, 
    images, IMAGES_DIR, image_ids, heights, widths, 
    output_path):
    # Finds masks for a list of images
    print('Writing', os.path.basename(output_path))
    images_start = time.time()
    image_count = 0
    mask_count = 0
    skipped_images = 0
    skipped_annos = 0
    cat_count_list_total = [0] * (max(valid_cat_ids) + 1)
    image_avg = 0
    boxes_dict = {}
    images_used = 0
    
    writer = tf.python_io.TFRecordWriter(output_path)
    
    for i, image in enumerate(images):
        image_name, image_ext = os.path.splitext(image)
        if image_ext != ".jpg":
            # can store png masks and visuals 'all-in-one' with simple skip
            continue # only setup for jpgs
        
        image_path = os.path.join(IMAGES_DIR, image)
        image_count +=1

        image_id = image_ids[i]
        image_height = heights[i]
        image_width = widths[i]
        
        boxes_dict = {}
        cat_ids = []
        boxes = []
        areas = []

        annotations = json_loaded['annotations']
        
        # Not keyed so going to iterate
        anno = ""
        for annotation in annotations:
            if image_id == annotation['image_id']:
                anno = annotation['segments_info']
                break
        if anno == "":
            print("No annotation found for image", image)
            skipped_images +=1
            continue
        
        for a in anno:
            #print(a)
            if a['category_id'] > 90:
                # skip if not coco 'things'
                continue
            if a['iscrowd'] != 0:
                # skip if crowd
                continue
            cat_ids.append(a['category_id'])
            # coco is xy wh pixels
            # tf od is yxyx norm
            boxes.append(a['bbox'])
            areas.append(a['area'])
        boxes_dict['cat_ids'] = cat_ids
        boxes_dict['boxes'] = boxes
        boxes_dict['areas'] = areas
            
        if len(boxes_dict) == 0:
            skipped_images +=1
            print('Skipping image.  No json refs for', image)
            continue
        
        tf_example, cat_count_list_total, skipped_images, skipped_annos = create_tf_example_for_boxes_area(
            boxes_dict, category_index, image_path, image_height,
            image_width, image_id, skipped_images, skipped_annos,
            SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN, 
            valid_cat_ids, translation_index, cat_count_list_total)

        if tf_example != None:
            writer.write(tf_example.SerializeToString())
            # Should match total processed minus skipped at end
            images_used += 1
            
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
        if image_count % 1000 == 0:
            image_time_total = time.time() - images_start
            image_time_mean = image_time_total / image_count
            if SAMPLE_CATEGORIES:
                for vid in valid_cat_ids:
                    print(vid, category_index[vid]['name'], cat_count_list_total[vid])
            print('Images %d Average %s' % (image_count, tsf(image_time_mean)))
        
    writer.close()

    for vid in valid_cat_ids:
        print(vid, category_index[vid]['name'], cat_count_list_total[vid])
        
    print('%d images skipped as no annotations to add' % (skipped_images))
    print('%d annotations skipped due to sampling' % (skipped_annos))
    print('Processed %d images and %d masks' % (image_count, mask_count))
    print('Total Images to TF Record File %d' % (images_used))
    print('-'*50)
    return images_used


def create_tf_record_from_json_annotate(
    json_loaded, category_index,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, 
    valid_cat_ids, translation_index, 
    images, IMAGES_DIR, heights, widths, 
    output_path):
    # Finds masks for a list of images
    # Note category index will use pbtxt but also included in json
    # Scores used.  No area information included
    
    print('Writing', os.path.basename(output_path))
    images_start = time.time()
    image_count = 0
    mask_count = 0
    skipped_images = 0
    skipped_annos = 0
    cat_count_list_total = [0] * (max(valid_cat_ids) + 1)
    image_avg = 0
    boxes_dict = {}
    images_used = 0
    
    writer = tf.python_io.TFRecordWriter(output_path)
    
    for i, image in enumerate(images):
        image_name, image_ext = os.path.splitext(image)
        if image_ext != ".jpg":
            # can store png masks and visuals 'all-in-one' with simple skip
            continue # only setup for jpgs
        
        image_path = os.path.join(IMAGES_DIR, image)
        image_count +=1

        image_height = heights[i]
        image_width = widths[i]

        boxes_dict = {}
        cat_ids = []
        boxes = []
        scores = []

        for ref in json_loaded['images']:
            if ref['image'] == image:
                print(image)
                for instance in ref['instances']:
                    if instance['instance_id'] == 0:
                        # No instances for image
                        continue
                    cat_ids.append(instance['category_id'])
                    boxes.append(instance['box'])
                    scores.append(instance['score'])
        boxes_dict['cat_ids'] = cat_ids
        boxes_dict['boxes'] = boxes
        boxes_dict['scores'] = scores
        
        if len(boxes_dict) == 0:
            skipped_images +=1
            print('Skipping image.  No json refs for', image)
            continue
        
        tf_example, cat_count_list_total, skipped_images, skipped_annos = create_tf_example_for_boxes_score(
            boxes_dict, category_index, image_path, image_height,
            image_width, skipped_images, skipped_annos,
            SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, 
            valid_cat_ids, translation_index, cat_count_list_total)
        
        if tf_example != None:
            writer.write(tf_example.SerializeToString())
            # Should match total processed minus skipped at end
            images_used += 1
            
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
        if image_count % 1000 == 0:
            image_time_total = time.time() - images_start
            image_time_mean = image_time_total / image_count
            if SAMPLE_CATEGORIES:
                for vid in valid_cat_ids:
                    print(vid, category_index[vid]['name'], cat_count_list_total[vid])

            print('Images %d Average %s' % (image_count, tsf(image_time_mean)))
        
    writer.close()

    for vid in valid_cat_ids:
        print(vid, category_index[vid]['name'], cat_count_list_total[vid])
        
    print('%d images skipped as no annotations to add' % (skipped_images))
    print('%d annotations skipped due to sampling' % (skipped_annos))
    print('Processed %d images and %d masks' % (image_count, mask_count))
    print('Total Images to TF Record File %d' % (images_used))
    print('-'*50)
    return images_used



def create_tf_example_for_masks(
    built_dict, category_index,
    image_path, mask_path,
    skipped_masks, skipped_annos,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN, 
    valid_cat_ids, translation_index, cat_count_list_total):

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

        if translation_index == {}:
            category_ids.append(cat_id)
        else:
            for tid in translation_index.keys():
                if translation_index[tid]['coco_id'] == cat_id:
                    category_ids.append(tid)
        
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


    feature_dict = {
        'image/height': int64_feature(image_height),
        'image/width': int64_feature(image_width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(Xmins),
        'image/object/bbox/xmax': float_list_feature(Xmaxs),
        'image/object/bbox/ymin': float_list_feature(Ymins),
        'image/object/bbox/ymax': float_list_feature(Ymaxs),
        'image/object/class/label': int64_list_feature(category_ids),
        'image/object/area': float_list_feature(pixel_areas),
        'image/object/mask': bytes_list_feature(binary_mask_list)}
    
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    return tf_example, cat_count_list_total, skipped_masks, skipped_annos
    # removed 'image/object/is_crowd': int64_list_feature(is_crowd),
    # removed key as not used when returned

def create_tf_example_from_masks_for_boxes(
    built_dict, category_index,
    image_path, mask_path,
    skipped_masks, skipped_annos,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN, 
    valid_cat_ids, translation_index, cat_count_list_total):

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

        if translation_index == {}:
            category_ids.append(cat_id)
        else:
            for tid in translation_index.keys():
                if translation_index[tid]['coco_id'] == cat_id:
                    category_ids.append(tid)
        
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

    feature_dict = {
        'image/height': int64_feature(image_height),
        'image/width': int64_feature(image_width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(Xmins),
        'image/object/bbox/xmax': float_list_feature(Xmaxs),
        'image/object/bbox/ymin': float_list_feature(Ymins),
        'image/object/bbox/ymax': float_list_feature(Ymaxs),
        'image/object/class/label': int64_list_feature(category_ids),
        'image/object/area': float_list_feature(pixel_areas)}
    # removed 'image/object/mask': bytes_list_feature(binary_mask_list)
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    return tf_example, cat_count_list_total, skipped_masks, skipped_annos
    # removed 'image/object/is_crowd': int64_list_feature(is_crowd),
    # removed key as not used when returned


def create_tf_record_from_png_masks(
    CODEC, CODEC_OFFSET, category_index,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN,
    valid_cat_ids, translation_index, 
    images, IMAGES_DIR, MASKS_DIR, MASK_EXT,
    output_path):
    # Finds masks for a list of images
    print('Writing', os.path.basename(output_path))
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
            # can store png masks and png visuals ('all-in-one') with simple skip
            # - or add ext ".png", add io and screen potential _visual and _mask
            continue # only setup for jpg images, all masks are png
        
        image_path = os.path.join(IMAGES_DIR, image)
        image_count +=1
        mask_found = False
        if CODEC == "binary_filename":
            mask_path = (os.path.join(MASKS_DIR, image_name))
            if os.path.isdir(mask_path):
                mask_count +=1
                mask_found = True
                built_dict = png_masks.rebuild_from_binary_mask_dir(mask_path) 
        else: 
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
        
        tf_example, cat_count_list_total, skipped_masks, skipped_annos = create_tf_example_for_masks(
            built_dict, category_index, 
            image_path, mask_path,
            skipped_masks, skipped_annos,
            SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN, 
            valid_cat_ids, translation_index, cat_count_list_total)

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
            print('Images %d Average %s' % (image_count, tsf(image_time_mean)))
        
    writer.close()
    # Reporting
    print('%d images skipped as no mask' % (skipped_images))
    print('%d masks skipped due to sampling' % (skipped_masks))
    print('%d annotations skipped due to sampling' % (skipped_annos))
    print('Completed %d images and %d masks' % (image_count, mask_count))
    print('-'*50)


def create_tf_record_from_png_masks_for_boxes(
    CODEC, CODEC_OFFSET, category_index,
    SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN,
    valid_cat_ids, translation_index, 
    images, IMAGES_DIR, MASKS_DIR, MASK_EXT,
    output_path):
    # Finds masks for a list of images
    print('Writing', os.path.basename(output_path))
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
            # can store png masks and png visuals ('all-in-one') with simple skip
            # - or add ext ".png", add io and screen potential _visual and _mask
            continue # only setup for jpg images, all masks are png
        
        image_path = os.path.join(IMAGES_DIR, image)
        image_count +=1
        mask_found = False
        if CODEC == "binary_filename":
            mask_path = (os.path.join(MASKS_DIR, image_name))
            if os.path.isdir(mask_path):
                mask_count +=1
                mask_found = True
                built_dict = png_masks.rebuild_from_binary_mask_dir(mask_path)
        else: 
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
        
        tf_example, cat_count_list_total, skipped_masks, skipped_annos = create_tf_example_from_masks_for_boxes(
            built_dict, category_index, 
            image_path, mask_path,
            skipped_masks, skipped_annos,
            SAMPLE_CATEGORIES, CATEGORY_SAMPLE_NUM, PIXEL_AREA_MIN, 
            valid_cat_ids, translation_index, cat_count_list_total)

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
            print('Images %d Average %s' % (image_count, tsf(image_time_mean)))
        
    writer.close()
    # Reporting
    print('%d images skipped as no mask' % (skipped_images))
    print('%d masks skipped due to sampling' % (skipped_masks))
    print('%d annotations skipped due to sampling' % (skipped_annos))
    print('Completed %d images and %d masks' % (image_count, mask_count))
    print('-'*50)
