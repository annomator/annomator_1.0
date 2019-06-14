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
import tensorflow as tf
import numpy as np
from PIL import Image

import image_utils

# Detecting

def load_frozen_graph(frozen_graph):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def detect_filepath_for_boxes_session(image_path, session):
    # Assumed - with graph.as_default(): ;  with tf.Session() as session:
    # Load image
    image = image_utils.pil_image_open(image_path)
    image_np = image_utils.numpy_from_image(image)
    
    # Get handles to input and output tensors (Get detection graph info)
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)

    # Run inference
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    output_dict = session.run(
        tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})
    # All outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    # Return image and output dict
    return image, output_dict



def detect_image_for_boxes_session(image, session):
  # Assumed - with graph.as_default(): ;  with tf.Session() as session:
  image_np = numpy_from_image(image)
  # Get handles to input and output tensors (Get detection graph info)
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

  # Run inference (Detect image)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
  output_dict = session.run(
      tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})
  # All outputs are float32 numpy arrays, so convert types as appropriate (All #.# array)
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
      output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict




# From TFOD/utils/ops.py
def reframe_box_masks_to_image_masks(
    box_masks, boxes, image_height, image_width):
    # Transforms the box masks back to full image masks
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
        boxes = tf.reshape(boxes, [-1, 2, 2])
        min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
        max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
        transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
        return tf.reshape(transformed_boxes, [-1, 4])

    box_masks = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    image_masks = tf.image.crop_and_resize(image=box_masks,
                                           boxes=reverse_boxes,
                                           box_ind=tf.range(num_boxes),
                                           crop_size=[image_height, image_width],
                                           extrapolation_value=0.0)
    return tf.squeeze(image_masks, axis=3)


def detect_image_for_session(image, session):
    # Assumed - with graph.as_default(): ;  with tf.Session() as session:
    image_np = image_utils.numpy_from_image(image)
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    if 'detection_masks' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        # ORIG utils_ops.reframe_box_masks_to_image_masks
        detection_masks_reframed = reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        
    # Run inference
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    output_dict = session.run(
        tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})
    # All outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def detect_numpy_for_cached_session(image_np, session, tensor_dict_cache):
    # Assumed - with graph.as_default(): ;  with tf.Session() as session:
    # Can make a significant difference to mask inference speed having cache
    # - when all images are the same (or pad resized to same)
    # Use tensor dict cache or build a new one
    if tensor_dict_cache != {}:
        tensor_dict = tensor_dict_cache
    else:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            # ORIG utils_ops.reframe_box_masks_to_image_masks
            detection_masks_reframed = reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    # Run inference
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    output_dict = session.run(
        tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})
    # All outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    # Return output dict and tensor dict for cache
    return output_dict, tensor_dict


