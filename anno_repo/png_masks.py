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

# Modules
import numpy as np
import os # binary folder only
import image_utils # binary folder only

##################################################
# Binary Codec - Files in folder of image name
##################################################

def encode_binary_filename(
    image_name, instance_count, category_id,
    category_name, category_count):
    image = str(image_name)
    total = str(instance_count)
    cat_id = str(category_id)
    name = str(category_name)
    count = str(category_count)
    #binary_filename = image
    binary_filename = "_".join([image, total, cat_id, name, count])
    binary_filename += ".png"
    return binary_filename

def decode_binary_filename_v1(binary_filename):
    # Naming convention for binary file encoding/decoding
    binary_mask_data = binary_filename.split('_')
    image_name = binary_mask_data[0]
    instance_count = binary_mask_data[1]
    category_id = binary_mask_data[2]
    category_name = binary_mask_data[3]
    category_count = binary_mask_data[4]
    mask_ext = binary_mask_data[5]
    return category_id, category_count, instance_count

def decode_binary_filename(binary_filename):
    binary_dict = create_binary_dict(binary_filename)
    category_id  = binary_dict['category_id']
    category_count = binary_dict['category_count']
    instance_count = binary_dict['instance_count']
    return category_id, category_count, instance_count

# def encode_binary_folder - UNTESTED - Use decode binary file
def create_binary_dict(binary_filename):
    # given a list (os.listdir) of binary filenames
    # returns a list binary_dicts - built_dict['category_id'][i]
    #total_masks = len(list_dir)
    binary_dict = {}
    #binary_dicts = []
    #image_names = []
    #instance_counts = []
    #category_ids = []
    #category_names = []
    #category_counts = []
    #mask_exts = []
    #for i in range(total_masks):
    binary_data = binary_filename.split('_')
    #print(binary_data)
    image_name = binary_data[0]
    binary_dict['image_name'] = image_name
    instance_count = binary_data[1]
    binary_dict['instance_count'] = instance_count
    category_id = binary_data[2]
    binary_dict['category_id'] = category_id
    category_name = binary_data[3]
    binary_dict['category_name'] = category_name
    split_ext = binary_data[4].split('.')
    category_count, mask_ext = split_ext[0], split_ext[1]
    binary_dict['category_count'] = category_count
    #mask_ext = binary_data[5]
    binary_dict['mask_ext'] = mask_ext
    #binary_dicts.append(binary_dict)
    #print(binary_dict)
    
    return binary_dict
    #return binary_dicts
  
##################################################
# Condensed Mask RGB 8-bit Codecs
##################################################

# Codec - Offset - "offset"

def encode_offset(category_id, category_count, instance_count, offset):
    # Simple.  0 offset is raw data
    R = offset + category_id
    G = offset + category_count
    B = offset + instance_count
    return R, G, B
def decode_offset(R, G, B, offset):
    category_id = R - offset
    category_count = G - offset
    instance_count = B - offset
    return category_id, category_count, instance_count


# Codec - Centric - "centric"


def encode_centric(category_id, category_count, instance_count):
    # Designed for use with dense or complex datasets like mscoco panoptic data
    # Helps to distinguish complex foreground and background scenes
    # Coding panoptic, it looks good with pleasant grays/natural and coding is fairly simple
    # Manual panoptic - harder to encode, decode or annotate manually than other codecs
    # Red < 128 reserved for background (128 - (cat_id - 90))
    # This translates Stuff (sky, grass) being more grays to blue/green due to low red
    # Things from grays to reds (never near black), mixed with some med blue and green
    if category_id > 90: # 'Stuff' 
        R = 128 - (cat_id - 90)
    else: # 'Things'
        R = 128 + category_id
    # Simple counts
    # Easier to encode manually and slightly easier to see category by color
    G = 128 + category_count
    B = 128 + instance_count
    # Oscilate odd counts
    # Easier to see each instance, particularly in complex images
    if G % 2 != 0:
        G = 128 - (G-128) 
    if B % 2 != 0:
        B = 128 - (B-128)
    return R, G, B
    
def decode_centric(R, G, B):
    # MSCOCO Panoptic 128+ = 'Things' and 128- = 'Stuff'-90
    if R > 128:
        category_id = R - 128
    else:
        category_id = 128 - (R - 90)
    category_count = abs(128 - G)
    instance_count = abs(128 - B)
    return category_id, category_count, instance_count


# "metric_100" uses encode ((cat_id, count, total)) decode ((R, G, B))
# A subset of metric_offset allows for reliable string encoding and decoding

def single_encode_metric_100(code_input):
    flip = str(code_input + 100)
    return int(flip[0] + flip[-1] + flip[-2])

def single_decode_metric_100(code_input):
    flip = str(code_input)
    return int(flip[-1] + flip[-2])

def encode_decode_metric_100(R_cat, G_count, B_total):
    if R_cat < 100:
        catR = single_encode_metric_100(R_cat)
    else:
        catR = single_decode_metric_100(R_cat)
    if G_count < 100:
        countG = single_encode_metric_100(G_count)
    else:
        countG = single_decode_metric_100(G_count)
    if B_total < 100:
        totalB = single_encode_metric_100(B_total)
    else:
        totalB = single_decode_metric_100(B_total)
    return catR, countG, totalB


# "metric_offset" uses encode ((cat_id, count, total), offset) decode ((R, G, B), offset)
# "metric_offset", set to 100 offset, will have the same math result as "metric_100" string

def single_encode_decode_metric(code_input):
    # aka def ende_metric(code_input):
    # Encode and decode "metric_offset"
    code_output = (code_input //100) *100
    code_output += (code_input %10) *10
    code_output += (code_input %100) //10
    return code_output

def encode_metric_offset(cat, count, total, offset):
    R = single_encode_decode_metric(cat) + offset
    G = single_encode_decode_metric(count) + offset
    B = single_encode_decode_metric(total) + offset
    return R, G, B

def decode_metric_offset(R, G, B, offset):
    category_id = single_encode_decode_metric(R) - offset
    category_count = single_encode_decode_metric(G) - offset
    instance_count = single_encode_decode_metric(B) - offset
    return category_id, category_count, instance_count



def codec(codec, encode_decode, R_cat, G_count, B_total, offset):
    # Encode or decode any codec - 2 strings, 4 integers
    # eg codec("offset", "encode", cat_id, cat_count, inst_count, 100)
    # eg codec("offset", "decode", R, G, B, 100)
    if codec == "offset":
        if encode_decode == "encode":
            R, G, B = encode_offset(R_cat, G_count, B_total, offset)
            return R, G, B
        else:
            cat, count, total = decode_offset(R_cat, G_count, B_total, offset)
            return cat, count, total
      
    elif codec == "centric":
        if encode_decode == "encode":
            R, G, B = encode_centric(R_cat, G_count, B_total)
            return R, G, B
        else:
            cat, count, total = decode_centric(R_cat, G_count, B_total)
            return cat, count, total

    elif codec == "metric_100":
        catR, countG, totalB = encode_decode_metric_100(R_cat, G_count, B_total)
        return catR, countG, totalB
    
    elif codec == "metric_offset":
        if encode_decode == "encode":
            R, G, B = encode_metric_offset(R_cat, G_count, B_total, offset)
            return R, G, B
        else:
            cat, count, total = decode_metric_offset(R_cat, G_count, B_total, offset)
            return cat, count, total
      
    elif codec == "binary_filename":
        if encode_decode == "encode":
            # does not translate - use binary dict
            pass 
        else:
            cat, count, total = decode_binary_filename(filename)
            return cat, count, total
      
    else:
        print("Error: No codec matches", codec)
        return
  

##################################################
# Condensed Mask Creation and Rebuild
##################################################


def create_mask_from_detection(
    image_np, output_dict,
    MAX_OBJECTS, CONFIDENCE, MASK_ENCODE, CODEC_OFFSET):
    # Create 8-bit rbg condensed png mask from tf detection
    
    instance_total = len(output_dict['detection_masks'])
    instance_count = 0
    cat_count_list = [0] * 256 # (index=class_id)
    mask_np = np.zeros_like(image_np)
    masks = []
    classes = []
    boxes = []
    codecs = []
    built_dict = {}
    for i in range(instance_total):
        if i == MAX_OBJECTS:
            break
        if output_dict['detection_scores'][i] < CONFIDENCE:
            continue

        instance_count +=1
        class_id = int(output_dict['detection_classes'][i])
        classes.append(class_id)
        cat_count_list[class_id] +=1
        category_count = cat_count_list[class_id]
        mask = output_dict['detection_masks'][i]
        masks.append(mask)
        R, G, B = codec(
          MASK_ENCODE, "encode",
          class_id, category_count, instance_count,
          CODEC_OFFSET)
        
        mask_np[:,:,0][mask == 1] = R 
        mask_np[:,:,1][mask == 1] = G
        mask_np[:,:,2][mask == 1] = B
        
        box = output_dict['detection_boxes'][i]
        boxes.append(box)
        
        codec_dict = {} # keep or code same
        codec_dict['cat_id'] = class_id
        codec_dict['count'] = category_count
        codec_dict['total'] = instance_count
        codec_dict['color'] = (R, G, B)
        codecs.append(codec_dict)
    built_dict['masks'] = np.asarray(masks).astype(np.uint8)
    built_dict['classes'] = np.asarray(classes).astype(np.uint8)
    built_dict['boxes'] = np.asarray(boxes).astype(np.float32)
    built_dict['codecs'] = codecs 

    return mask_np, built_dict


def rebuild_from_mask(mask_np, MASK_DECODE, CODEC_OFFSET, category_index):
    # Rebuild from Mask
    # Returns data in a similar format to detections - scores removed, codec_dict added
    # Scores can be inferred by instance count and category count (car 1 is highest score car)
    unique = np.unique(mask_np.reshape(-1, 3), axis=0) 
    binary_masks = []
    classes = []
    boxes = []
    codecs = [] 
    built_dict = {}

    new_cat_count_list = [0] *256
    new_cc = []
    new_instance_count = 0
    new_ic = []

    unique_count = len(unique)
    max_masks = 256
    if unique_count > max_masks:
        print("Mask error: Skipped as", unique_count, "masks more than", max_masks)
        print("Common causes are external manipulations: antialising and 'image (part) in mask'")
        return

    for color in unique:
        R = int(color[0])
        G = int(color[1])
        B = int(color[2])
        if color.all() == 0:
            continue # skip black

        class_id, cat_count, inst_count = codec(
            MASK_DECODE, "decode",
            R, G, B,
            CODEC_OFFSET)

        # Skip if no valid categegory, set to 0 for counts
        if class_id < 1 or class_id > 255:
            print("Skipping instance as no valid category id for codec", MASK_DECODE)
            continue
        if cat_count < 0 or cat_count > 255:
            cat_count = 0
        if inst_count < 0 or inst_count > 255:
            inst_count = 0
          
        classes.append(class_id)
        # Create new counts in case any are invalid
        new_cat_count_list[class_id] +=1
        new_cc.append(new_cat_count_list[class_id])
        new_instance_count +=1
        new_ic.append(new_instance_count)

        codec_dict = {} # keep here or code same
        codec_dict['cat_id'] = class_id
        codec_dict['count'] = cat_count
        codec_dict['total'] = inst_count
        codec_dict['color'] = (R, G, B)
        codecs.append(codec_dict)
        
        binary_mask = np.zeros_like(mask_np[:,:,0]) 
        binary_mask[
            (mask_np[:,:,0] == R) &
            (mask_np[:,:,1] == G) &
            (mask_np[:,:,2] == B)] = 1
        binary_masks.append(binary_mask)

        mask_pixels = np.where(binary_mask == 1)
        bbox_top = np.min(mask_pixels[0])
        bbox_bottom = np.max(mask_pixels[0])
        bbox_left = np.min(mask_pixels[1])
        bbox_right = np.max(mask_pixels[1])

        
        image_height = binary_mask.shape[0]
        image_width = binary_mask.shape[1]
        ymin = (float(bbox_top / image_height))
        ymax = (float(bbox_bottom / image_height))
        xmin = (float(bbox_left / image_width))
        xmax = (float(bbox_right / image_width))
        box = np.asarray([ymin, xmin, ymax, xmax])

        boxes.append(box)

    # Check if all images have a valid category count and instance count
    # Now have all the counts, need to check if all are not zero.
    codec_count_valid = True
    for codec_dict in codecs:
        if 0 in codec_dict['color']:
            codec_count_valid = False
    if not codec_count_valid:
        print("Using generated category and instance counts")
        for i in range(len(codecs)):
            codecs[i]['count'] = new_cc[i]
            codecs[i]['total'] = new_ic[i]
    
    built_dict['boxes'] = np.asarray(boxes).astype(np.float32)
    built_dict['classes'] = np.asarray(classes).astype(np.uint8)
    built_dict['masks'] = np.asarray(binary_masks).astype(np.uint8)
    built_dict['codecs'] = codecs

    return built_dict

def rebuild_from_binary_mask(binary_np, binary_filename):
    # Convert binary numpy and filename to one element of built_dict
    binary_dict = create_binary_dict(binary_filename)
    building_dict = {}
    
    mask_pixels = np.where(binary_np == 1)
    bbox_top = np.min(mask_pixels[0])
    bbox_bottom = np.max(mask_pixels[0])
    bbox_left = np.min(mask_pixels[1])
    bbox_right = np.max(mask_pixels[1])
    
    image_height = binary_np.shape[0]
    image_width = binary_np.shape[1]
    ymin = (float(bbox_top / image_height))
    ymax = (float(bbox_bottom / image_height))
    xmin = (float(bbox_left / image_width))
    xmax = (float(bbox_right / image_width))
    box = np.asarray([ymin, xmin, ymax, xmax])

    codec_dict = {} # keep here or code same
    codec_dict['cat_id'] = binary_dict['category_id']
    codec_dict['count'] = binary_dict['category_count']
    codec_dict['total'] = binary_dict['instance_count']
    codec_dict['color'] = ([1])
    
    building_dict['box'] = box
    building_dict['class'] = binary_dict['category_id']
    building_dict['mask'] = binary_np
    building_dict['codec'] = codec_dict

    return building_dict

def rebuild_from_binary_mask_dir(dir_path):
    binary_images = os.listdir(dir_path)
    built_dict = {}
    masks = []
    classes = []
    boxes = []
    codecs = []
    for filename in binary_images:
        binary_file_path = os.path.join(dir_path, filename)
        binary_image = image_utils.pil_image_open(binary_file_path)
        binary_np = image_utils.numpy_from_image(binary_image)
        building_dict = rebuild_from_binary_mask(binary_np, filename)
        masks.append(building_dict['mask'])
        classes.append(building_dict['class'])
        boxes.append(building_dict['box'])
        codecs.append(building_dict['codec'])                        
    
    built_dict['masks'] = masks
    built_dict['classes'] = classes
    built_dict['boxes'] = boxes
    built_dict['codecs'] = codecs

    return built_dict
