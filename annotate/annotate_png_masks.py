
# coding: utf-8
#from __future__ import print_function # Python 2.7
#from __future__ import division
"""
Mission Objective:
Make it easy for Windows, Mac or Linux users to run Mask RCNN

Target Audiences:
Researchers: Image to text - classify and count
Personal use: Image to text - classify contents using 80 labels
You can then search for things like people, car, dog, boat directly to be able to search for text photos of interest
Small business: stock-take to staff management - mrcnn can do it all
Photo and Video professionals: Apart from the ability to look through all photos or video for descriptions (shot list or rushes list), colorizing, special effects and green screen are all possible with a little tweaking.
Big business: not for you... well... if your motto is 'do some good' then you should know tensorflow runs everything from amazon robots to 'ok google'.  

Remove the need to compile
"""

# # Welcome to Masking for All
# ### A cross-platform version of Tensorflow Object Detection
# ### The fastest and lightest version of Mask RCNN - InceptionV2 - setup and ready to start detecting
# 
# If you think linux may be a cross between a lion and a lynx, that python and anaconda are snakes,
# If you have never compiled anything, this is for you.  This is written primarily for Windows and Mac with no prior AI or ML experience.  
# imho, this shit is the new 0,0,0
# It is designed for easy end-user encoding
# A designer can choose from a broad spectrum of colors
# A researcher can code precisely which, how much, where in instance is
# All this 'head-maths' is automated in the encoder but designed for humans
# Simply red is category id, green is instance count and blue is category count
# These are encoded 100 + 10*count  (110, 120, 130)
# This keeps the colors produced visibly separate
# - 0 on 1 binaries are black on near black
# - coding around 128+ is all grey
# - oscilating out limited count to 127 but works ok for panoptic
# - working in negative, binary multiplication, all too hard
# - the head-maths, explanation and training would not work

# The bottom line is it was way
# But doesn't that leave a count of 10? 25, at best?  No.
# Without straying into distracting / loud colors ?
# - coding where 100 is 0, where 110 is 1 and 209 is 0.
# - Easy head-maths? Easy for most cases as can read number backwards
# - Could always use a color chart for use with dropper but last resort
# The encode is simple for the first 15: 100 + 10*count
# encode 1-10: 1-110, 2-120, 3-130 ... 9-190, 10-101
# You could decode simply with count = (color - 100) / 10
# This would not suit some tasks as 100 should be seen as entirely possible
# The formula is fairly easy to learn, calculate and infer directly from code
# I know that red 129, green 110, blue 120 is the first person of two objects
# That is 128+coco(1), (the middle number), (the middle number)
# You can't avoid noticing if it has a 1 or 2 to the left, and a 0, 1, 2 to the right
# The formula becomes more complicated at the back end...
# (python % the remainder,  // no remainder)
# encode count = 100 + ((10*count) % 99) 
# decode count =((color % 10)*10) + ((color - 100) // 10) 
# encode 9-190, 10-101, 11-111, 12-121, 3-131 ... 19-191, 20-102
### Gives a range of 0-98, 99 total, without getting more complicated
### Just ignore the 1 (would start in black otherwise)
### the number is now just backwards 191 becomes 19. easy peasy maskineasy!
# You can then alter the mask.  Scrub, add and alter easily!!!
# You can even just start with a copy of a pic and start masking
# Red has been reserved for categories
# - 128+ for objects, 128- for environment
# This is also new but works well with coco panoptic
# A zero index for coco category ends up with dark objects, red sky etc
# If start at 128 and go out (I am calling mapped split index)
# - Ideally, if no category index already exists for mapping...
# - map supercategories directly to red scheme at stepped increments
# - The absence of red, given neutral green, gives blue
# - - ie objects will be increasingly red, background will increasingly blue
# - A simple version is indoor objects mapped 129-192, outdoor mapped 193-255
# - This leaves us with 63 slots for each... too hard
# - A simpler version is just to map to 100+
# - - gives a better range of colors as dips below 128 and ranges to 'not gaudy'
# - - this would allow for 10 slots for people, 130 animals, 120 for vehicles, 
# The best color scheme can be produced using a coded color chart
# I have created one that puts a spectrum of blue (instance count)
# - on top of an oscilating count of green
# - This gives every category a good color range to choose from
# - 
# use a color chart to code 
# aka easy encoder
# aka Mighty Mini Masker
# aka Masky McMask Mask or Masky Mask or Masky McMask Face

# Suggested workflow
# 1 auto-annotate
# 2 Visually check, correct and improve
# 3 Train using condensed mask or export to binary format

# Python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print("Loading modules...")

#from datetime import datetime as dt
#start_time = dt.now()
import time


# Import first to prevent warning messages
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements


import os # for os file paths

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np # for arrays and data manipulation
#import six.moves.urllib as urllib # for downloading
#import sys # for folder reference
#import tarfile # for zip files
import tensorflow as tf
#import zipfile # for zip files

from collections import defaultdict # text - storing
from io import StringIO # text - translating
from matplotlib import pyplot as plt # for image display
from PIL import Image # for image import

from matplotlib import patches as patches # for visual only - mask is all numpy

import sys
ANNO_REPO_DIR = os.path.join('..', 'anno_repo')
sys.path.append(ANNO_REPO_DIR)

#import default_category_names
#category_index = default_category_names
# or
import category_names # contains category_index
category_index = category_names.category_index
import image_utils
import tf_detections
import png_masks
from gen_functions import time_seconds_format as tsf


# The tf_slim folder is a copy of tf slim with object_detection within
# tf_slim
# - deployment
# - nets etc
# - object detection
# Not all are needed but easy to replicate when tf slim or object changes
# You can also just copy the tf_slim folder and run scripts within
# Any folder structure that works for you is ok, just ensure sys.path.append to tf slim

# A copy of necessary files from object_detection so demo works
# see tensorflow/research/object_detection for full code

# Object detection files
##########from object_detection.utils import ops as utils_ops
##########from object_detection.utils import visualization_utils as vis_util

#python -V
#print("Tensorflow", tf.__version__)


# 'http://download.tensorflow.org/models/object_detection/'

FROZEN_GRAPH = os.path.join(os.path.abspath('./'), 'frozen_graph', 'frozen_inference_graph.pb')

# ORIG - List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
#NUM_CLASSES = 90


TEST_IMAGES = os.path.join(os.path.abspath('./'), 'test_images')
COCO_2017_DIR = os.path.join('../', '../', 'MSCOCO_2017_PANOPTIC')
#TEST_IMAGES = os.path.join(COCO_2017_DIR, 'train2017')
OUTPUT_DIR = os.path.join(os.path.abspath('./'), 'output_masks')
#OUTPUT_DIR = os.path.join(COCO_2017_DIR, 'png_metric_100_train2017')
BINARY_IMAGES_DIR = os.path.join(OUTPUT_DIR, "_binary_masks")
#BINARY_IMAGES_DIR = os.path.join(COCO_2017_DIR, "png_binary_train2017")

CONFIDENCE = 0.75
MAX_OBJECTS = 100

# Codec Options - offset, centric, metric_100, metric_offset
MASK_ENCODE = "metric_100"
MASK_DECODE = MASK_ENCODE 
CODEC_OFFSET = 100

# Resize all images
# Speed up detection and resize ready for training with clean scaling
# A copy will be made and all the masks, visuals and binaries will match
# Detects faster when all images are smaller and same
IMAGE_RESIZE = False # Longest XY if scale=0 (antialiased)
RESIZE_PADDING = True # Pad to XY - detection speed boost if all same
RESIZE_BORDER = 0 # pixels
SAVE_IMAGE_RESIZE = True # Needed for training and re-runs to match mask
RESIZE_SCALE = 0.0 # > 0.0 will override resize X Y - eg 0.5 all half size
# or set longest side max - X, Y generally the same
RESIZE_IMAGE_X = 512
RESIZE_IMAGE_Y = RESIZE_IMAGE_X


#CREATE_BLEND_IMAGE = False # Changed to visual


# Too keep visual image size down use VISUAL_MAX to cap the size
# To enlarge or make all visuals the same size, use VISUAL_MIN = VISUAL_MAX
# If you 
# It will also adjust text size for better viewing depending on the source images



# The png condensed mask will be made.  Choose which extra files to create.
# Although mask pngs are small, large images will create large visual files.
# The visual setting will not effect the mask or the detection.
# It can be used to dramatically save disk space
# - but doesn't dramatically change speed unless extreme.
# They are purely used to see the result in a way that works for the dataset/setup/you
# They can also be remade from the mask on the next execution so write notes or delete at will.
# Hell... it's the only place you can have some artistic licence so go nuts.
CREATE_VISUAL_IMAGE = True # From Tensorflow Object Detection
VISUAL_BLEND = 0.5 # Mask visibility
# Proportion option.  Overides visual min/max if >0. 
VISUAL_RESIZE = 0.0 # >0.0 # 1 will return same size as original eg 0.5 = half size
# or set max to reduce, min to enlarge or set min to max to for same size
VISUAL_MAX = 10000 # pixels # Default 10000 (off bar huge) 1000 (on)
VISUAL_MIN = VISUAL_MAX # pixels # Default 0 (off), number=enlarge or VISUAL_MAX


# Export to another system
# Stay synchronized with mask and visual or export afterwards
# Note: you can use the condensed masks directly for training
# This will build binary pngs from compatible condensed masks, auto, human or both.
# Annotated by filename: image, instance id, category id, category name and category count
# Searchable labels by score rank order with no external text or json needed. ("aza out" mike drop).  
CREATE_BINARY_IMAGES = False
BINARY_IMAGES_DIR = os.path.join('.', 'binary_masks')
#BINARY_IMAGES_DIR = os.path.join(OUTPUT_DIR, "_binary_masks")
#BINARY_IMAGES_DIR = os.path.join(COCO_2017_DIR, "_binary_masks")




################################################
# Code
################################################

start_time = time.time()

test_images = os.listdir(TEST_IMAGES)

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if CREATE_BINARY_IMAGES:
    if not os.path.exists(BINARY_IMAGES_DIR):
        os.mkdir(BINARY_IMAGES_DIR)
# status report
mask_count = 0
visual_count = 0
binary_count = 0
complete_count = 0

# Create image_dict
# Lists which files are complete or still need to be detected or rebuilt from mask
# It also checks the status of requested output files depending on settings.
# Allows:
# - Delete anything and re-run anytime.
# - Fix the mask just delete the visual to remake with your changes.
# - Delete the masks from one model, upgrade and rerun on difficult images
# - Keep all the visuals up to date or batch binary export - All at speed
# - image_dict is also used to attach every instance found for reporting/analysis/json
image_count = 0
image_dicts = []
for test_image in test_images:
    #image_start = dt.now()
    #image_start = time.time()
    image_dict = {}
    image_path = os.path.join(TEST_IMAGES, test_image)
    image_name, ext = os.path.splitext(test_image)
    if ext != ".jpg" and ext != ".png":
        continue # skip if not jpg or png (could be folder, hidden or not image)
    if image_name[-5:] == "_mask":
        continue # skip masks if images and masks in one folder
    if image_name[-7:] == "_visual":
        continue # skip visuals if images and visuals in one folder
      
    image_count +=1
    
    mask_file_path = os.path.join(OUTPUT_DIR, image_name + "_mask.png")
    mask_visual_path = os.path.join(OUTPUT_DIR, image_name + "_visual.png")
    binary_masks_image_dir = os.path.join(BINARY_IMAGES_DIR, image_name)
    # Use image_id if you have an external image id.  Only used for reporting. 
    image_dict['image_id'] = image_count # count / reporting id / external id
    image_dict['image_name'] = image_name 
    image_dict['image_complete'] = True
    image_dict['image_path'] = image_path
    image_dict['mask_path'] = mask_file_path
    image_dict['mask_exists'] = False
    image_dict['visual_path'] = mask_visual_path
    image_dict['visual_exists'] = False
    image_dict['binary_dir'] = binary_masks_image_dir
    image_dict['binary_exists'] = False

    # Check and count paths of possible creations
    if os.path.exists(image_dict['mask_path']):
        mask_count +=1
        image_dict['mask_exists'] = True
    else:
        image_dict['image_complete'] = False
    if CREATE_VISUAL_IMAGE:
        if os.path.exists(image_dict['visual_path']):
            visual_count += 1
            image_dict['visual_exists'] = True
        else:
            image_dict['image_complete'] = False
    if CREATE_BINARY_IMAGES:
        # Checks for folder of binary masks only
        # Folder may be incomplete on interrupt
        if os.path.exists(image_dict['binary_dir']):
            bimgs = os.listdir(image_dict['binary_dir'])
            binary_count += 1
            image_dict['binary_exists'] = True
        else:
            complete_count = False
            image_dict['image_complete'] = False
    image_dicts.append(image_dict)


# Status report simple
#print("Status: Images", image_count, "Masks", mask_count)
# Status report string
report_string = "Status: Images " + str(image_count)
report_string += " Masks " + str(mask_count)
if CREATE_VISUAL_IMAGE:
    report_string += " Visuals " + str(visual_count)
if CREATE_BINARY_IMAGES:
    report_string += " Binaries " + str(binary_count)
print(report_string)

detection_graph = tf_detections.load_frozen_graph(FROZEN_GRAPH)

with detection_graph.as_default():
    with tf.Session() as session:
        cache_image_size = (0,0)
        tensor_dict_cache = {}
        for image_dict in image_dicts:
            if image_dict['image_complete']:
                continue
            #image_start_time = dt.now()
            image_start = time.time()
            #print("Processing", image_dict['image_name'])
            image = Image.open(image_dict['image_path'])
            
            if IMAGE_RESIZE:
                image = image_utils.resize_image(
                    image, "image", 
                    RESIZE_SCALE, RESIZE_IMAGE_X, RESIZE_IMAGE_Y,
                    RESIZE_PADDING, RESIZE_BORDER)
                if SAVE_IMAGE_RESIZE:
                    resized_filename = os.path.basename(image_dict['image_path'])
                    image.save(os.path.join(OUTPUT_DIR, resized_filename))
            if image.size != cache_image_size:
                #print("resetting tensor dict cache")
                cache_image_size = image.size
                tensor_dict_cache = {}
            image_np = image_utils.numpy_from_image(image)#np.array(image).astype(np.uint8)
            if image_dict['mask_exists']:
                mask = Image.open(image_dict['mask_path'])
                mask_np = image_utils.numpy_from_image(mask)#np.array(mask).astype(np.uint8)
                built_dict = png_masks.rebuild_from_mask(
                    mask_np, MASK_DECODE, CODEC_OFFSET, category_index)
            else:
                output_dict, tensor_dict_cache = tf_detections.detect_numpy_for_cached_session(
                    image_np, session, tensor_dict_cache)
                if output_dict.get('detection_masks') is None:
                    print("No masks found for current graph")
                    break
                mask_np, built_dict = png_masks.create_mask_from_detection(
                    image_np, output_dict,
                    MAX_OBJECTS, CONFIDENCE, MASK_ENCODE, CODEC_OFFSET)
                # Safer handover and access to image and numpy
                mask = image_utils.image_from_numpy(mask_np)#Image.fromarray(np.uint8(mask_np)).convert('RGB')
                mask.save(image_dict['mask_path']) # not tf. but neater
            # Now should have image, image_np, mask, mask_np and built_dict
            if CREATE_VISUAL_IMAGE:
                image_utils.create_visual_from_built(
                    image_dict['visual_path'], built_dict, image_np, mask_np, category_index, 
                    VISUAL_MIN, VISUAL_MAX, VISUAL_BLEND, VISUAL_RESIZE)
            if CREATE_BINARY_IMAGES:
                image_utils.create_binaries_from_built(
                    BINARY_IMAGES_DIR, image_dict['image_name'], built_dict, category_index)
            plt.close('all') # clear images
            image_dict['codecs'] = built_dict['codecs']
            image_name = os.path.basename(image_dict['image_path'])
            print(image_dict['image_id'], image_name, "complete", tsf(time.time() - image_start))

# image_dict now contains codec_dict for each image processed
# No checks have been made so 'exists' flags are initial state
# From here you can check the files, detections or rebuilds
# You can also export reports to json or text file. 
print('-'*50)
print("Processed", image_count, "images.  Total time", tsf(time.time() - start_time))
print('-'*50)

