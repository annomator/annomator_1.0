
# coding: utf-8
#from __future__ import print_function # Python 2.7
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
# If you think linux may be a cross between a lion and a lynx, that python and anaconda are snakes, if you have never compiled anything, this is for you.  This is written primarily for Windows and Mac with no prior AI or ML experience.  
# 
# Everything is evolving fast in the world of Aritificial Inteligence and Machine Learning.  Google has open-sourced Tensorflow with has been widely adopted.  New to the collection of cutting edge models released is Mask RCNN.  It is a little hard to find and get working, particularly on Windows or Mac, but if you are confident please use the official installation instructions.  
# 
# The problem at the moment is it is not very user friendly.  You cannot run any of the code directly on Mac or Windows without fixing many errors.  Many of the bigger models also require a lot of computing power and will fail because it assumes you have a good linux computer.  The other option of buying cloud time, or uploading gigs of photos are simply not practical for the other 99% percent of us.  
# 
# I am currently running 1.5.0 as it is the only version that will run on Windows, Mac and Linux.  You should be able to detect without compiling anything.  You should be able to train too but the setup is more complicated.
# 
# If you have Windows 10, Mac Sierra, or Ubuntu 14 (ref) you should be able to run the lightest version.  I have also gone through the pain of installing and upgrading a graphics card, CUDA on linux and every time I do I want a week of my life back...  We are often talking about whether something will run in 4 hours or 8.  I sleep for 8 and most work computers are idle for 16 hours a day.  The question is how much can I get done overnight?  
# 
# Considering it would take a few days to go through every picture I have ever taken, how much power do you need?  You won't be able to take photos fast enough to keep up with it.  I have also tested all the other models (the resnet 50, resnet 101 and inception resnet V2) and found little difference.  In fact, the inceptionV2 often scored better in some situations.  It also runs at least times 4 faster (tf says 8x) than any other.   
# 
# Most of us are on Mac, Windows or Linux.  Most Macs and Linux come with python 2.7.  It is everywhere but will phased out in 2020.  I have coded to the latest python version at the time of writing (3.6.5) to make it easy for new users.  It is supposed to work with python 2.7 and tensorflow 1.4 but save yourself the trouble.  Update to python 3.6.5 and (possiblly downgrade) to tensorflow 1.5.0
# 
# A case in point is "pip install tensorflow" on Mac will give you 1.8.0 at the moment.  It will not work with object detection scripts.  Neither will 1.5.1.  The models were all trained in tf 1.5.0 so all good.  I had errors with 1.4.0 even though the demo notebook proudly adds 2 more lines of code to check, it is wrong in most cases.  You may be able to run it on linux with 2.7 but I assume most end-users can't.  
# 
# It is complicated.  Don't be put off.  It is expected that no more than 10 lines of code and 10 minutes to get you started.  
# 
# there is one non-python-3 line of code that will make it break for some.
# 
# I have simply tried to make a version of Tensorflow Object Detection API that is OS agnostic.  Lost you already?  I have written a version that will work on Mac, Windows and Linux.  You do not need to compile anything and I will try to keep it simple.  
# 
# It is primarily written with researchers in mind.  The ability to identify and count ojects has been around for a while.  The ability mask (colour in) is relatively new.  To do this, every pixel is coded.  Now you can itendify and mask every object you can get the information out.  
# You can finally do some amazing things automagically:
# researcher: count every individual bird, person, car, dog, boat
# video: can now do smart green-screen (no green background needed)
# images: make a text file for each image for analysis in excel etc
# 
# The best thing is you can avoid painful hours of making masks manually.  You can then use the pre-made masks to train.  Booya!
# 
# Displaying to screen is good for a few pictures but what if you have thousands of pictures?
# 
# The other target audience 

# In[ ]:


# Note changes to original demo
# - disable auto-download
# - disable use zip files
# - disable use of pbtxt as needs protoc install etc
# - add coco class_index - for MSCOCO label names
# - - note can use skip_labels=True and the original demo will work (minus label names)
# - add ouput.py for command line execution - TODO
# - add pdf, txt/csv and jpg/png output - TODO
# - add matplotlib backend - fixes backend macos, multiple warnings


# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

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


#sys.path.append("../code")


# The code folder is a copy of tf slim with object_detection within
# code
# - deployment
# - nets etc
# - object detection
# Not all are needed but easy to replicate when tf slim or object changes
# You can also just copy the code folder and run scripts within
# The code is kept separate and a few working examples
# Any folder structure that works for you is ok, just ensure sys.path.append to code

# A copy of necessary files from object_detection so demo works
# see tensorflow/research/object_detection for full code
# Object detection files
##########from object_detection.utils import ops as utils_ops
# Moved up
##########from object_detection.utils import visualization_utils as vis_util

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
#FROZEN_GRAPH = os.path.join(os.path.abspath('./'), 'frozen_graph', 'frozen_inference_graph.pb')
FROZEN_GRAPH = os.path.join(os.path.abspath('.'), 'frozen_graph', 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

#NUM_CLASSES = 90


# ## Download Model

# In[5]:


#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())


# Built with for researchers, students and personal use
# Make no money, pay no money - all open-sourced for you
# Student - Easy introduction tensorflow and object detection
# - Plain code and comments for beginners
# Researchers - Customize to your needs with outputs you can use directly
# - From analizing satelite imagery to counting cancer cells, this is a powerful tool
# Personal - Catalogue all your old photos to a searchable list of contents
# - Find all images of 10+ people, boats, trains, dogs, cats etc
# If you want more features see the masker


# Built with a restarter you can stop and start anytime.
# It will collect any text files and detect on any new images
# You can mix and match images (with text) and create a new image summary
# It is not as pretty as the annotater but the output can be 'image' png jpg or pdf
# The principle being to make sure all platforms and end-users can see and use results

# Want better, faster?
# - Just drop in any coco model from tensorflow zoo and it should work
# (incV2 Mask RCNN included) - Works with resnet50, resnet10, incresV2
# Compatible with single shot, faster rcnn, nasnet - from fastest to best
# The default model contains masks as well incV2 Mask RCNN
# It is a great light-weight all-rounder that compares well to others



# I have tested on Linux, Windows and Mac and Python 2.7 - 3.6
# I have designed to be tf 1.5.0 compatible so older machines can use it.
# There is still some testing to be done 

# I have included the box data but I do not recommend trying to alter it.
# It is there to give reference to which instance is "that one"
# Possibly more use when working with others but a number and box makes it easier
# The format is 0-1 so is tolerant to rescaling.  It is not pixels so seems user unfriendly.
# In context of all looking at a screen or projecter, what relavance is pixels?
# It makes more sense to refer to say half way, left top and these numbers could be read
# - as percent/proportion from top left. 
# eg ie x .25, y .25, (top-left-corner), x .5 y .5 (bottom-right-corner)
# describes a box that start 25% right, 25% down and opposite corner is middle of image
# It does not need to be used at all but it can be handy and also useful for training data

# # Detection
# Format for linux, windows and mac safe paths
TEST_IMAGES = os.path.join(os.path.abspath('.'), 'test_images')
OUTPUT_DIR = os.path.join(os.path.abspath('.'), 'ouput_text')

# Click to open on most platforms and aimed at low-tech end-user
# We should all love linux, compiling, pngs, json, run length encoding but I have probably lost half of you on the left and half on the right...
# Half switched off at linux coders and that coders can't see more than 80+ chars/line (oops) characters...? letters... my bad
# The other half can't believe I let that line of code hang out there and want to fix it...
# I want to reassure the newbs on Windows and Mac I have done my best to make it usable as is.

# It is set to go.  No compiling, no protos, no coding at all.
# It is a gentle introduction to the awesome Tensorflow Object Detection and Tensorflow Slim
# Many where lost compiling protos to get the pbtxt to get the category index.  I have just popped it in.
# Try switching show_labels=False on the demo and if it finally works... you probably didn't protos the protoc.  Then you have count them? index them... highest id them (NUM_CLASSES)...  Point made.  Let's all move on.
# I have a quick visual image (mainly to check text) but avoids other dependancies as well
# Please use the official instructions and try the official demo.  It has the 

# Use stackoverflow for questions.  Use github for reporting replicable bugs.
# Assume the gods are busy but will respond if you are respectful, cogent and complete.
# It is coded in Python 3 circa 3.6 - Testing for 2.7+ only on linux, windows and mac
# I am losing people... ok...quick geek speak 1.5.0 for all platforms - see setup for more

# I have also coded as plainly as possible so that it can be easily modified.  
# For the geeks it still stores boxes for training and can use any cutting-edge tf zoo coco model
# Geeks can probably whip up a quick detection-to-json or RLE quicker and better.  So what I am I am trying to say is this is prabably not for you... lost you both again :-)
# Try the masker out if you want training data.  More sophisticated with png semantic encoding

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



# Although possible, this version makes no attemp to recreate visuals from text files
# It is only used to check the validity of the annotations and make minor corrections
# Simply choose to make a visual or not at the time of creation and keep it with the text file
# This allows a level or correction to text files but no way to visually check corrections - see masker

# The proposed workflow would be add images to test_images folder
# Run output.py.   Adjust confidence and max objects to suit data and purpose.
# Check each visual image for accuracy
# You can edit the text file but do so with caution
# - It is used to recompile summary so format must be identical
# - You can easily delete a false positive.
# - Altering is relatively easy most text editers or spreadsheets should work with txt or csv
# - Adding can be done for image summary but care should be taken with format
# To keep large datasets from getting too large you can:
# - You could delete each image after checking, leaving only the text file
# - When happy that the image summary is correct you can archive or delete the rest

# You can keep the text and visual with the image and combine them for new summary
# The box data will enable a reconstruction of the visual image or used in analysis
# The instance number and category count represent confidence order
# Open the image summary directly from text editor, spreadsheet or database


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
            # Not standard time formatting but easy to understand and modify
            # Should cater for a spectrum of hardware and system setups - Change to your needs
            #image_time = dt.now() - image_start
            #image_time_string = str(image_time.seconds) + "."
            #image_time_string += str(round(image_time.microseconds/1000)) + "s"
            #print(image_count, test_image, '-', image_time_string)
            print(image_count, test_image, tsf(time.time() - image_start))

print("Saving new image summary", SUMMARY_PATH)
image_summary_formatted = "ImageName,InstanceId,ClassId,ClassName,ClassCount,Score,Box\n"
image_summary_formatted += "\n".join(image_summary) # array to string
with open(SUMMARY_PATH, "w") as f:
    f.write(image_summary_formatted)

print('-'*50)
print("Processed", image_count, "images.  Total time", tsf(time.time() - start_time))
print('-'*50)
