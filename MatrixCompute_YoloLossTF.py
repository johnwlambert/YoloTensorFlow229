# John Lambert, Matt Vilim, Konstantine Buhler
# Acknowledgment to Xuerong Xiao for helping write much of the code
# Dec. 14, 2016


import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import os, sys
from os import listdir
from os.path import isfile, join
import cPickle

IMAGE_SIZE = 448
NUM_GRID = 7
GRID_SIZE = 64 # IMAGE_SIZE / NUM_GRID
NUM_BOX = 2 # why did Xuerong have it as 5?
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5
THRESHOLD = 0.2
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
NUM_CLASSES = len(CLASSES)


def sqrt_wh(box):
  """
  Take the square root of wh regardless of pred or true boxes given
  INPUTS:
  - box: 
  OUTPUTS:
  - box_new: 
  """
  if len(box.get_shape().as_list()) == 4:
    box_new = tf.concat(3, [box[:, :, :, : 2], tf.sqrt(box[:, :, :, 2 :])])
  else:
    print "LABELS HAVE WRONG SHAPE !!!"
  return box_new


def square_wh(box):
  """
  Take the square of wh regardless of pred or true boxes given
  INPUTS:
  - 
  OUTPUTS:
  - 
  """
  if len(box.get_shape().as_list()) == 4:
    box_new = tf.concat(3, [box[:, :, :, : 2], tf.square(box[:, :, :, 2 :])])
  else:
    print "BOXES HAVE WRONG SHAPE !!!"
  return box_new


def compute_iou(box_pred, box_true):
  """
  tensorflow version of computing iou between the predicted box and the gt box
  INPUTS:
  - box_pred: Tensor of shape [?, 7, 7, 4], xy norm to image, wh as is
  - box_true: Tensor of shape [?, 1, 1, 4], xy norm to image, wh as is
  OUTPUTS:
  - iou: Tensor of shape [?,7,7,1], intersection over unit with each predicted bounding box
  """
  lr = tf.minimum(box_pred[:, :, :, 0] + 0.5 * box_pred[:, :, :, 2],   \
                  box_true[:, :, :, 0] + 0.5 * box_true[:, :, :, 2]) - \
       tf.maximum(box_pred[:, :, :, 0] - 0.5 * box_pred[:, :, :, 2],   \
                  box_true[:, :, :, 0] - 0.5 * box_true[:, :, :, 2])
  tb = tf.minimum(box_pred[:, :, :, 1] + 0.5 * box_pred[:, :, :, 3],   \
                  box_true[:, :, :, 1] + 0.5 * box_true[:, :, :, 3]) - \
       tf.maximum(box_pred[:, :, :, 1] - 0.5 * box_pred[:, :, :, 3],   \
                  box_true[:, :, :, 1] - 0.5 * box_true[:, :, :, 3])
  lr = tf.maximum(lr, lr * 0)
  tb = tf.maximum(tb, tb * 0)
  intersection = tf.mul(tb, lr)
  union = tf.sub(tf.mul(box_pred[:, :, :, 2], box_pred[:, :, :, 3]) +  \
                 tf.mul(box_true[:, :, :, 2], box_true[:, :, :, 3]), intersection)
  iou = tf.div(intersection, union)
  iou = tf.reshape(iou, [-1, NUM_GRID, NUM_GRID, 1])  # otherwise [?, 7, 7]
  return iou


def compute_ious(pred_boxes, gtbox):
  """
  computing ious between two predicted boxes and the ground truth box
  INPUTS:
  - pred_boxes: list of TF Tensors of shape [?, 7, 7, 4], xy norm to image, wh as is
  - gtbox:  Tensor of shape [?, 1, 1, 4], xy norm to image, wh as is
  OUTPUTS:
  result: a list of IOUs for each of the pred_boxes with gt box. Shape=[?, 7, 7, NUM_BOX]
  """
  ious = [compute_iou(pb, gtbox) for pb in pred_boxes]
  result = tf.concat(3, ious)
  return result


def compute_single_loss(pred_labels, gt_labels):
  """
  INPUTS:
  - pred_labels: [batch=1, 7, 7, 20+2*5], batch has to be 1?!
              xy norm to grid, wh square root (out of network)
      PRETRAINED WEIGHTS REQUIRE: [20 CLASS PROBS , C1,C2, X Y W H, X Y W H]
      CURRENTLY, IN LOSS FN, TF MODEL REQUIRES: [20 CLASSES, XYWH,XYWH,C1,C2] I THINK?

  - gt_labels: xy norm to image, wh square root (read from file)
              [num_gtbox, 73=NUM_CLASSES+4+49]
  when compute iou, need to have pred boxes norm to image, wh as is
  when compute coord loss, need to have true boxes norm to grid, wh square root

  Confidence computed as Pr(Object) * IOUtruth_pred
  If no object exists in that cell, the confidence scores should be zero. 
  Otherwise we want the confidence score to equal the 
  intersection over union (IOU) between the predicted box and the ground truth.
  Finally the confidence prediction represents the IOU between the 
  predicted box and any ground truth box.

  We assign one predictor to be "responsible" for predicting an object 
  based on which prediction has the highest current IOU with the ground truth. 
  """
  gt_classes = tf.reshape(gt_labels[:, 0 : NUM_CLASSES], [-1, 1, 1, NUM_CLASSES]) # [num_gt_box,1,1,NUM_CLASSES]
  gt_pr_object = tf.reshape(gt_labels[:, NUM_CLASSES + 4 : ], [-1, 7, 7, 1]) # [num_gt_box,1,1,4]
  gt_boxes = tf.reshape(gt_labels[:, NUM_CLASSES : NUM_CLASSES + 4], [-1, 1, 1, 4]) # [num_gt_box,7,7,1]

  pred_classes = pred_labels[:, :, :, 0 : NUM_CLASSES]
  # CONSTRUCT A LIST OF PREDICTED BOXES

  pred_boxes_arr = pred_labels[:, :, :, NUM_CLASSES : NUM_CLASSES + NUM_BOX * 4]
  print 'Predicted Box Arr Shape: ', pred_boxes_arr.get_shape().as_list()
  pred_boxes = [pb for pb in tf.split(3, NUM_BOX, pred_boxes_arr)]
  pred_p_obj = pred_labels[:, :, :, NUM_CLASSES + NUM_BOX * 4 : NUM_CLASSES + NUM_BOX * 5]

  # Threshold the object probabilities
  mask_coord = tf.greater(pred_p_obj, THRESHOLD * tf.ones_like(pred_p_obj))
  # Select either the predicted confidence or zero
  pred_masked = tf.select(mask_coord, pred_p_obj, tf.zeros_like(pred_p_obj))

  for pb in pred_boxes:
    print 'Predicted Box Shape: ', pb.get_shape().as_list()

  print 'GT Box Shape: ', gt_boxes.get_shape().as_list()
  with tf.variable_scope('coord_loss'):
    box_loss = [tf.reduce_sum(tf.square(pb - gt_boxes), reduction_indices=3, keep_dims=True) \
                for pb in pred_boxes]
    # BOX LOSS IS A LIST OF LOSSES FOR EACH BOX
    squared_pred_list = [square_wh(pb) for pb in pred_boxes]
    ious = compute_ious(squared_pred_list, square_wh(gt_boxes))

    obj_loss_tmp = tf.square(pred_p_obj - gt_pr_object)

    # For each of the predicted boxes in each grid cell, we check if B_1 or B_2 has a higher IOU 
    # Find which predicted boxes have largest IOU with the ground truth?
    # AT THIS POINT WE COMPLETELY FORGET ABOUT THE GRID CELLS
    temp_max_iou = ious[:, :, :, 0 : 1]
    temp_coord_loss = box_loss[0]
    temp_obj_loss = obj_loss_tmp[:, :, :, 0 : 1]
    for i in range(NUM_BOX - 1):
      mask_temp = tf.greater(temp_max_iou, ious[:, :, :, i + 1 : i + 2])
      temp_max_iou = tf.select(mask_temp, temp_max_iou, ious[:, :, :, i + 1 : i + 2])
      temp_coord_loss = tf.select(mask_temp, temp_coord_loss, box_loss[i + 1])
      print "temp_coord_loss: ", temp_coord_loss
      temp_obj_loss = tf.select(mask_temp, temp_obj_loss, obj_loss_tmp[:, :, :, i + 1 : i + 2])

    # pick largest value along the last dimension of ious, use the corresp. loss

    coord_loss = LAMBDA_COORD * tf.mul(gt_pr_object, temp_coord_loss)

  with tf.variable_scope('obj_loss'):
    obj_loss = LAMBDA_COORD * tf.mul(gt_pr_object, temp_obj_loss)

  with tf.variable_scope('noobj_loss'):
    noobj_loss = LAMBDA_NOOBJ * tf.mul(1 - gt_pr_object, temp_obj_loss)    # (?, 7, 7, 1)

  with tf.variable_scope('class_loss'):
    classes_diff = tf.mul(gt_pr_object, pred_classes - gt_classes)
    class_loss = tf.reduce_sum(tf.square(classes_diff), reduction_indices=3)
    class_loss = tf.reshape(class_loss, [-1, 7, 7, 1])

  with tf.variable_scope('total_loss'):
    total_loss = coord_loss + obj_loss + noobj_loss + class_loss
    # One loss per grid cell, i.e. total_loss has shape [:,7,7,1]
    loss = tf.reduce_mean(tf.reduce_sum(total_loss, reduction_indices=[1,2,3]), reduction_indices=0)

  print loss.get_shape().as_list()
  return loss








#   # Each grid cells also predicts conditional class probabilities, Pr(Classi |Object). 
#   # These probabilities are conditioned on the grid cell containing an object.
#   # Why not just at test time do cross product?
#   classes_diff = tf.mul(true_prob_obj, pred_classes - gt_classes)
#   class_loss = tf.reduce_sum(tf.square(classes_diff), reduction_indices=3)
#   # Why not turn into a scalar below?
#   class_loss = tf.reshape(class_loss, [-1, 7, 7, 1])




