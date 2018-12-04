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

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
from matplotlib import patches 
from PIL import Image
from PIL import ImageDraw
import numpy as np

import os # only for binary
import png_masks # only for binary

# Functions to access PIL, matplotlib, pyplot
# Allows numpy, no general functions or mask
# Allows os but file management should be external

# Any image manipulation
def numpy_from_image(image):
    # Convert RGBA to RGB numpy uint8
    return np.array(image.convert("RGB")).astype(np.uint8) # Y, X
def image_from_numpy(numpy):
    # R, G, B to binary (single channel gray)
    return Image.fromarray(np.uint8(numpy)) # X, Y

# External Access
def pil_image_open(image_path):
    pil_image = Image.open(image_path)
    return pil_image
def save_numpy_as_image(filename, numpy):
    # Save png, jpg, pdf from numpy array
    image = image_from_numpy(numpy)
    image.save(filename)
def load_image_as_numpy(filename):
    image = pil_image_open(filename)
    return numpy_from_image(image)

# Visual blend
def blend_numpys(image_np, mask_np, blend):
    numpy_ratio = 1 - blend
    blend_np = (image_np * numpy_ratio) + (mask_np * blend)
    return blend_np

def resize_image(image, method, scale, resizeX, resizeY, padding, border):
    # Image resizer for jpg images, png masks or making quick thumbnails
    # Resize by scale or longest side, with border and padding
    # Note resize uses bilinear by default, thumbnail uses lanczos (antialias)
    # I include bilinear as "other" as it is faster than lanczos yet not as blocky as bilinear
    # Scale first, resize then borders and pad last, else simple resize
    if method == "image" or method == "images":
        # Nice for images but slower and will break png masks
        # Note Image.ANTIALIAS is an alias for Image.LANCZOS
        resample_method = Image.LANCZOS 
    elif method == "mask" or method == "masks":
        # Clean scaling for masks but blocky for images
        resample_method = Image.NEAREST 
    elif method == "fast":
        # Faster than "image", good if you just need a working copy fast
        resample_method = Image.BILINEAR
        
    origX, origY = image.size
    # Scale
    if scale > 0:
        resizeX = int(origX * scale)
        resizeY = int(origY * scale)
    # Scale X Y
    if origX > origY:
        newX = resizeX
        newY = int((resizeX/origX) * origY)
    else:
        newX = int((resizeY/origY) * origX)
        newY = resizeY
    # Border and padding
    if border > 0 or padding:
        padded_image = Image.new("RGB",(resizeX,resizeY))
        newX = newX - border
        newY = newY - border
        # prev image = image.resize((newX,newY), Image.ANTIALIAS)
        image = image.resize((newX,newY), resample_method)
        marginX = (resizeX - newX) // 2
        marginY = (resizeY - newY) // 2
        padded_image.paste(image, (marginX,marginY))
        rescaled_image = padded_image
    else:
        # Simple resize
        # prev rescaled_image = image.resize((newX,newY), Image.ANTIALIAS)
        rescaled_image = image.resize((newX,newY), resample_method)
    return rescaled_image




##################################################
# Outputs - visual and binary
##################################################



def draw_box_and_label_on_image(image, draw, box, label):
    # requires from PIL import ImageDraw
    # Runs faster for multiple if given 'draw image' or auto-makes new
    if draw == "" or draw is None:
        draw = ImageDraw.Draw(image)
    imgX, imgY = image.size
    Xmin, Xmax, Ymin, Ymax = box[1], box[3], box[0], box[2]
    l, r, t, b = Xmin * imgX, Xmax * imgX, Ymin * imgY, Ymax * imgY
    #l, r, t, b = left, right, top, bottom # use short or long version
    box_to_lines = [(l, t), (l, b), (r, b), (r, t), (l, t)]
    draw.line(box_to_lines, width=2, fill=(128, 128, 128, 128))
    draw.text((l, t), label, fill='white') # color can be rgba, rgb, name or abbreviation
    
"""
def draw_box_and_label_on_draw(imgX, imgY, draw, box, label):
    l, r, t, b = box[1] * imgX, box[3] * imgX, box[0] * imgY, box[2] * imgY
    draw.line([(l, t), (l, b), (r, b), (r, t), (l, t)], width=2, fill=(128, 128, 128, 128))
    draw.text((l, t), label, fill='w') # color can be rgba, rgb, name or abbreviation
"""


def create_visual_from_built(
    # requires from matplotlib import patches
    visual_path, built_dict, image_np, mask_np, category_index, 
    VISUAL_MIN, VISUAL_MAX, VISUAL_BLEND, VISUAL_RESIZE):
    #print("Creating visual image")
    if VISUAL_RESIZE < 0 or VISUAL_RESIZE > 10:
        print("Error: VISUAL_RESIZE out of range 0.0-10.0")
        return
    if VISUAL_MIN > VISUAL_MAX:
        print("Error: VISUAL_MIN > VISUAL MAX")
        return
    if VISUAL_MIN < 100 or VISUAL_MIN > 10000:
        print("Error: VISUAL_MIN out of range 100-10000")
        return
    if VISUAL_MAX < 100 or VISUAL_MAX > 10000:
        print("Error: VISUAL_MAX out of range 100-10000")
        return
    #VISUAL_BLEND = 0.5 # 0-1
    #VISUAL_RESIZE = 1
    #image_ratio = 1 - VISUAL_BLEND
    #blend_np = (image_np * image_ratio) + (mask_np * VISUAL_BLEND)
    #print("blend_np", blend_np.shape)
    #blend_image = Image.fromarray(np.uint8(blend_np))
    #imgY, imgX = blend_np.shape[0], blend_np.shape[1] # NTS order is YX
    # or
    blend_np = blend_numpys(image_np, mask_np, VISUAL_BLEND)
    blend_image = image_from_numpy(blend_np)
    imgX, imgY = blend_image.size
    
    boxes = built_dict['boxes']
    codecs = built_dict['codecs']

    # Pixels to inches to pixels.  Have to use inches for plt.figure(figsize)
    # Given 550x550 image - dpi=100, 550 pixels would be 5.5 inches
    # This is then translated back to 550 pixels (same size as original)
    # This is all fairly seemless but a lot of coding for a nicer visual
    # See the text versions for a few lines of code using ImageDraw
    
    # Start with simple(ish) no resize default
    dpi_inches = 100 # do not alter
    blend_inches = (imgX / dpi_inches, imgY / dpi_inches)
    dpi_save = dpi_inches
    
    # Scale by ratio or enlarge/reduce to longest edge
    scalor = 0
    if VISUAL_RESIZE > 0:
        dpi_save = int(dpi_inches * VISUAL_RESIZE)
    elif imgX > VISUAL_MAX or imgY > VISUAL_MAX:
        scalar = VISUAL_MAX
    elif imgX < VISUAL_MIN or imgY < VISUAL_MIN:
        scalar = VISUAL_MIN

    if scalor > 0:
        if imgX > imgY: # landscape
            img_ratio = scalor / imgX
            scaleX = scalar / dpi_inches
            scaleY = (imgY * img_ratio) / dpi_inches
        else: # portrait or square
            img_ratio = scalor / imgY
            scaleX = (imgX * img_ratio) / dpi_inches
            scaleY = scalar / dpi_inches
        blend_inches = (scaleX, scaleY)

    # For this we will use blend_inches... figures and axes... bottom up indexing...
    # Imagefont needs to locate font files locally, and a font_dict, but may suit some users.
    # So... a chart it is... more options but essentially for control of fonts
    fig = plt.figure(figsize=blend_inches)
    ax = fig.add_axes([0, 0, 1, 1]) # Set x, y to be 0-1
    ax.imshow(blend_image) 
    
    # rebuilt boxes is normalised 0-1 from top left yxyx
    # patches expects from bottom left norm ...
    for i, box in enumerate(boxes):
        left = box[1]
        right = box[3]
        top = 1 - box[0] # flip y axis to bottom up
        bottom = 1 - box[2] # flip y axis to bottom up
        width = right - left
        height = top - bottom
        # box
        p = patches.Rectangle((left, bottom), width, height,
                              fill=False, transform=ax.transAxes, clip_on=False,
                              linewidth=2, edgecolor='gray', facecolor='none')
        ax.add_patch(p)
        # label
        category_id = codecs[i]['cat_id']
        if category_id in category_index.keys():
            if 'display_name' in category_index[category_id].keys():
                category_name = str(category_index[category_id]['display_name'])
            else:
                category_name = str(category_index[category_id]['name'])
        else:
            category_name = "NO ID"
        category_count = str(codecs[i]['count'])
        instance_count = str(codecs[i]['total'])
        patch_label = " ".join([instance_count, category_name, category_count])
 
        # Default alignment is top left outside box
        # Shift alignment and position for edges
        # If close to top, shift label inside box
        # If close to right edge, use right alignment
        if left > 0.8:
            hor_alignment = 'right'
            label_x = right - 0.005
        else:
            hor_alignment = 'left'
            label_x = left + 0.005
        if top < 0.95:
            ver_alignment = 'bottom'
            label_y = top + 0.005
        else:
            ver_alignment = 'top'
            label_y = top - 0.005
        # Nice modern Tohoma standard with alignments, weights etc
        # Lots of options that should work and appear similar accross platforms
        ax.text(label_x, label_y, patch_label, transform=ax.transAxes,
                color='w', ha=hor_alignment, va=ver_alignment)
                # ref weight='medium', fontsize='medium', 
                # ref family='sans-serif', name='Verdana', 
    # Along with figure size scaling, you can alter outputs you want for your dataset
    # With a numpy yx pixel twist, to inverted normalized inches, to pixels, I wish ImageFont worked
    ax.set_axis_off()
    plt.savefig(visual_path, dpi=dpi_save) # last reference to inches
    plt.close()
    
    
def create_binaries_from_built(binary_dir, image_name, built_dict, category_index):
    # Make both the directory and image names have the root with info join _
    # Designed as export after condensed mask creation or detection
    # You can also use it directly from detections or batch them at end
    #print("Creating binary images")
    binary_image_dir = os.path.join(binary_dir, image_name)
    if not os.path.exists(binary_image_dir):
        os.mkdir(binary_image_dir)

    masks = built_dict['masks']
    codecs = built_dict['codecs']
    
    for i, codec in enumerate(codecs):
        #print("binary codec",i, codec)
        # Making the name...
        ##    cat_id = codec['cat_id'] # need the int for name
        ##    cat_name = str(category_index[cat_id]['name']) # just in case name is number
        ##    cat_id = str(cat_id) # now to string
        ##    cat_count = str(codec['count'])
        ##    instance_count = str(codec['total'])
        ##    # Folder and root of name is image name
        ##    binary_mask_path = os.path.join(folder, root_name) 
        ##    binary_mask_path +=  "_" + "_".join([instance_count, cat_id, cat_name, cat_count])
        ##    binary_mask_path +=  ".png"
        # OR
        #cat_id = codec['cat_id']
        cat_name = category_index[codec['cat_id']]['name']
        binary_filename = png_masks.encode_binary_filename(
            image_name, codec['total'], codec['cat_id'], cat_name, codec['count'])
        binary_mask_path = os.path.join(binary_image_dir, binary_filename)
        #print(binary_mask_path)
        # Mask and save
        #Image.fromarray(np.uint8(masks[i])).save(binary_mask_path)
        #or
        #binary_mask_image = image_from_numpy(masks[i])
        #binary_mask_image.save(binary_mask_path)
        #or
        image_from_numpy(masks[i]).save(binary_mask_path)
        #or
        #save_numpy_as_image(binary_mask_path, masks[i])
        #plt.close()
        # The tf way
        #binary_image = Image.fromarray(np.uint8(binary_mask))
        #with tf.gfile.Open(binary_mask_path, 'w') as fid:
        #  binary_image.save(fid, 'PNG')
