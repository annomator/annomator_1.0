"""
Annomator
Copyright 2018 Arend Smits.
All rights reserved.  MIT Licence.  
"""
"""
Credit goes to 
"""

# Python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import json

import PIL.Image as Image

import time

import sys
sys.path.append(os.path.join('..', 'anno_repo'))
import png_masks
import gen_functions
tsf = gen_functions.time_seconds_format

COCO_IMAGE_DIR = os.path.join('.', 'convert_input', 'input_images')
COCO_PNG_DIR = os.path.join('.', 'convert_input', 'panoptic_examples')
COCO_JSON_FILE = os.path.join('.', 'convert_input', 'panoptic_examples.json')

OUTPUT_PNG_DIR = os.path.join('.', 'convert_output')

# MSCOCO options
# Exclude crowds and panoptic for easy training
INCLUDE_CROWDS = False
# True is all categories, False is COCO 90 'Thing' classes
INCLUDE_PANOPTIC = False

# Encode options 
MASK_CODEC = "metric_100"
CODEC_OFFSET = 100

CHECK_JSON = True
CHECK_IMAGES = False

start_time = time.time()

if not os.path.exists(COCO_IMAGE_DIR):
    print("Error: COCO_IMAGE_DIR", COCO_IMAGE_DIR)
    exit()
if not os.path.exists(COCO_PNG_DIR):
    print("Error: COCO_PNG_DIR", COCO_PNG_DIR)
    exit()
if not os.path.exists(COCO_JSON_FILE):
    print("Error: COCO_JSON_FILE", COCO_JSON_FILE)
    exit()

if not os.path.exists(OUTPUT_PNG_DIR):
    os.mkdir(OUTPUT_PNG_DIR)

with open(COCO_JSON_FILE, 'r') as f:
    panoptic_coco = json.load(f)

if CHECK_JSON:
    # Check all json refs have an image file
    sub_start = time.time()
    anno_count = 0
    json_passed = True
    for annotation in panoptic_coco['annotations']:
        anno_count +=1
        if anno_count % 10000 == 0:
            print("Checking json", anno_count, tsf(time.time() - sub_start))
        img_path = os.path.join(COCO_PNG_DIR, annotation['file_name'])
        if not os.path.exists(img_path):
            json_passed = False
            print("No image found for json reference", annotation['file_name'])

if CHECK_IMAGES:
    # Check all png masks have a json ref
    coco_panoptic_images = os.listdir(COCO_PNG_DIR)
    img_count = 0
    sub_start = time.time()
    passed_checks = True
    for image in coco_panoptic_images:
        name, ext = os.path.splitext(image)
        if ext != ".png":
            continue # may be hidden files (eg mac)
        img_count +=1
        if img_count % 100 == 0:
            print("Checking image", img_count, tsf(time.time() - sub_start))
        anno_found = False
        for annotation in panoptic_coco['annotations']:
            if image in annotation['file_name']:
                anno_found = True
        if not anno_found:
            img_passed = False
            print("No json reference found for", image)
            
# Collect panoptic tag - 1 if 'coco 90' else 0 if panoptic or no id
cat_pan = [0] * 256
for cat in panoptic_coco['categories']:
    cat_pan[cat['id']] = cat['isthing']

image_counter = 0
skipped = 0
annotations = []
images_start = time.time()

start_conversion = True
if CHECK_IMAGES and not images_passed:
    start_conversion = False
    print("Some png masks did not have json references")
if CHECK_JSON and not json_passed:
    start_conversion = False
    print("Some json references did not have png masks")
    
if start_conversion:
    print("Starting conversion")
    for annotation in panoptic_coco['annotations']:

        image = annotation['file_name']
        name, ext = os.path.splitext(image)
        if ext != ".png":
            continue
        
        image_counter +=1
        mask_path = os.path.join(OUTPUT_PNG_DIR, name + "_mask.png")
        if os.path.exists(mask_path):
            if image_counter % 1000 == 0:
                print("Skipping as found complete", image_counter)
            skipped +=1
            continue
        
        segments = []
        mask_json = []
        
        # Load image
        image_array = np.array(Image.open(os.path.join(COCO_PNG_DIR, image)), dtype=np.uint8)
        R_pan = image_array[:,:,0]
        G_pan = image_array[:,:,1]
        B_pan = image_array[:,:,2]
        
        color_id = R_pan + 256 * G_pan + 256 * 256 * B_pan

        unique, counts = np.unique(color_id, return_counts=True)

        segment_count = 0
        category_count_list = [0] * 256

        for u in unique:
            if u == 0:
                continue
            # Set color to black to scrub by default
            # - this can remove crowd and stuff (or other criteria)
            # - this will also remove png color combinations (color id) that have no reference
            color = [0,0,0]
            for segment in annotation['segments_info']:
                scrub_segment = False
                anno_cat_id = segment['category_id']
                if segment['iscrowd'] == 1:
                    if not INCLUDE_CROWDS:
                        #print("scrubbing crowd", segment['iscrowd'])
                        scrub_segment = True
                if cat_pan[anno_cat_id] == 0:
                    if not INCLUDE_PANOPTIC:
                        #print("scrubbing stuff", segment['category_id'])
                        scrub_segment = True
                if segment['id'] == u and not scrub_segment:
                    
                    category_count_list[anno_cat_id] +=1
                    category_count = category_count_list[anno_cat_id]
                    segment_count +=1

                    R, G, B = png_masks.codec(
                        MASK_CODEC, "encode",
                        anno_cat_id, category_count, segment_count,
                        CODEC_OFFSET)
                    color = [R, G, B]

                # Add segment to mask
                mask = (color_id == u)
                image_array[mask] = color
            # end of segments
        Image.fromarray(image_array).save(os.path.join(OUTPUT_PNG_DIR, name + "_mask.png"))

        if image_counter % 100 == 0:
            print("Images %d Time %s" % (image_counter, tsf(time.time() - start_time)))

print('-'*50)
print("Total time:", tsf(time.time() - start_time))
print('-'*50)

