import argparse
import os
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from yolo_utils import scale_boxes

# matplotlib inline

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes thanks to the max boxes_scores,
    # keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

    filtering_mask = box_class_scores >= threshold

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes

def iou(box1, box2):
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (xi1 - xi2) * (yi1 - yi2)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou_res = inter_area / union_area

    return iou_res

def yolo_non_max_suppresion(scores, boxes, classes, max_boxes = 1.0, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Step:
        1. Select the box that has the highest score
        2. Compute its overlap with all other boxes, and remove
            boxes that overlap it more then iou_threshold
        3. Go back step 1 and iterate until there's no more boxes
            with a lower score than the current selected box

    :param scores:
    :param boxes:
    :param classes:
    :param max_boxes:
    :param iou_threshold:
    :return:
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor use in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initializer

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold=iou_threshold)

    # Use L.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppresion(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes



