# John Lambert, Matt Vilim, Konstantine Buhler
# Acknowledgment to Xuerong Xiao for suggestions on how to write the loss function
# Dec. 15, 2016


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
NUM_BOX = 2
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


def square_wh(boxes):
  """
  Take the square of wh regardless of pred or true boxes given
  INPUTS:
  - boxes: n-d array of shape ?????
  OUTPUTS:
  - 
  """
  print boxes.get_shape().as_list()
  #if len(box.get_shape().as_list()) == 4:
  boxes_wh_squared = tf.concat(1, [boxes[:, :2], tf.square(boxes[:, 2:]) ])
  #else:
  #  print "BOXES HAVE WRONG SHAPE !!!"
  return boxes_wh_squared


def compute_iou(box_pred,box_true):
  """
  tensorflow version of computing iou between the predicted box and the gt box
  INPUTS:
  - box_pred: Tensor of shape [?, 7, 7, 4], xy norm to image, wh as is
  - box_true: Tensor of shape [?, 1, 1, 4], xy norm to image, wh as is
  OUTPUTS:
  - iou: Tensor of shape [?,7,7,1], intersection over unit with each predicted bounding box
  """
  # print 'PRED BOX SHAPE: ', box_pred.get_shape().as_list()
  # print 'GT BOX SHAPE: ', box_true.get_shape().as_list()
  lr = tf.minimum(box_pred[ :, 0 ] + 0.5 * box_pred[ :,2 ],   \
                  box_true[ :, 0 ] + 0.5 * box_true[ :,2 ]) - \
       tf.maximum(box_pred[ :, 0 ] - 0.5 * box_pred[ :,2 ],   \
                  box_true[ :, 0 ] - 0.5 * box_true[ :,2 ])
  tb = tf.minimum(box_pred[ :, 1 ] + 0.5 * box_pred[ :,3 ],   \
                  box_true[ :, 1 ] + 0.5 * box_true[ :,3 ]) - \
       tf.maximum(box_pred[ :, 1 ] - 0.5 * box_pred[ :,3 ],   \
                  box_true[ :, 1 ] - 0.5 * box_true[ :,3 ])
  lr = tf.maximum(lr, lr * 0)
  tb = tf.maximum(tb, tb * 0)
  intersection = tf.mul(tb, lr)
  union = tf.sub(tf.mul(box_pred[ :, 2], box_pred[ :, 3 ]) +  \
                 tf.mul(box_true[ :, 2], box_true[ :, 3 ]), intersection)
  iou = tf.div(intersection, union)

  return iou


def compute_ious(box_pred,box_true):
  all_ious = compute_iou(box_pred,box_true)
  # can reshape if necessary
  #iou = tf.reshape(iou, [4, 1])  # otherwise [?, 7, 7]
  return all_ious


# def compute_ious(pred_boxes, gtbox):
#   """
#   computing ious between two predicted boxes and the ground truth box
#   INPUTS:
#   - pred_boxes: list of TF Tensors of shape [?, 7, 7, 4], xy norm to image, wh as is
#   - gtbox:  Tensor of shape [?, 1, 1, 4], xy norm to image, wh as is
#   OUTPUTS:
#   result: a list of IOUs for each of the pred_boxes with gt box. Shape=[?, 7, 7, NUM_BOX]
#   """
#   ious = [compute_iou(pb, gtbox) for pb in pred_boxes]
#   result = tf.concat(3, ious)
#   return result


def computeYoloLossTF( pred_classes, pred_conf, pred_boxes, gt_conf, gt_classes, ind_obj_i, gt_boxes_j0):
  """
  As a simplification, right now I only match one bounding box predictor
  to 1 ground truth, in each grid cell

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

  NEW INPUTS:
  - pred_classes: 49 x 20
  - pred_conf: 49 x 2 array
  - pred_boxes: 49 x 8 array
  - gt_boxes_j0; 49 x 4
  - I'M IGNORING THIS FOR NOW IN THE SIMPLIFICATION -- gt_boxes_j1: 49 x 4
  - gt_conf: 49 x 4, values in each of 4 columns are identical (tiled/repmatted)
  - gt_classes: 49 x 20
  - ind_obj_i: 49 x 20, indicating if that grid cell contains any object
  """
  pred_conf_j0 = pred_conf[:,0]
  pred_conf_j1 = pred_conf[:,1]
  ############ BOX LOSS ##################################################
  pred_boxes = tf.reshape( pred_boxes, shape=[49,2,4] )
  pred_boxes_j0 = pred_boxes[:,0,:] # 49 x 4 array
  pred_boxes_j1 = pred_boxes[:,1,:] # 49 x 4 array
  pred_boxes_j0 = tf.mul( pred_boxes_j0 , gt_conf ) # multiply by 1s or 0s
  pred_boxes_j1 = tf.mul( pred_boxes_j1 , gt_conf ) # multiply by 1s or 0s 
  # NOW the predictions in wrong cells are zeroed out
  j0_coord_loss = tf.reduce_sum(tf.square(pred_boxes_j0 - gt_boxes_j0), reduction_indices=[1] )
  squared_gt_boxes_j0 = square_wh(gt_boxes_j0)
  squared_pred_boxes_j0 = square_wh(pred_boxes_j0)
  squared_pred_boxes_j1 = square_wh(pred_boxes_j1)
  # For each of the predicted boxes in each grid cell, we check if B_1 or B_2 has a higher IOU with GT
  #pbs_j0 = [pb for pb in tf.split(0, 49, squared_pred_boxes_j0 )] # BREAK INTO chunks of 4
  ious_j0 = compute_ious( squared_pred_boxes_j0 , squared_gt_boxes_j0 )
  #pbs_j1 = [pb for pb in tf.split(0, 49, squared_pred_boxes_j1 )] # BREAK INTO chunks of 4
  ious_j1 = compute_ious( squared_pred_boxes_j1 , squared_gt_boxes_j0 )
  mask_temp = tf.greater( ious_j0, ious_j1 )
  final_ious = tf.select(mask_temp, ious_j0, ious_j1 )
  j1_coord_loss = tf.reduce_sum(tf.square(pred_boxes_j1 - gt_boxes_j0), reduction_indices=[1] )
  box_loss = tf.select(mask_temp, j0_coord_loss, j1_coord_loss )
  box_loss = LAMBDA_COORD * tf.reduce_sum(box_loss, reduction_indices=[0])
  ##############################################################################

  ############ OBJECT LOSS ##################################################
  # NOW ZERO OUT THE predicted_CONFIDENCES if no object there
  pred_conf_j0 *= gt_conf[:,0]
  pred_conf_j1 *= gt_conf[:,0]
  # Now only one of those predictors is doing the work. Mask out the other
  j0_mask = tf.logical_and( mask_temp, tf.greater( pred_conf_j0, tf.zeros_like(pred_conf_j0)) )
  j1_mask = tf.logical_and( tf.greater( pred_conf_j1, tf.zeros_like(pred_conf_j1)), tf.logical_not(mask_temp))
  
  pred_conf_j0 = tf.select(j0_mask, pred_conf_j0, tf.zeros_like(pred_conf_j0) )
  pred_conf_j1 = tf.select(j1_mask, pred_conf_j1, tf.zeros_like(pred_conf_j1) )

  obj_ious = final_ious * gt_conf[:,0]
  gt_conf_j0 = tf.select(j0_mask, obj_ious, tf.zeros_like(pred_conf_j0) )
  gt_conf_j1 = tf.select(j1_mask, obj_ious, tf.zeros_like(pred_conf_j0) )

  j0_obj_loss = tf.square( pred_conf_j0 - gt_conf_j0 )
  j1_obj_loss = tf.square( pred_conf_j1 - gt_conf_j1 )

  obj_loss = tf.reduce_sum( j0_obj_loss + j1_obj_loss, reduction_indices=[0])
  ##############################################################################

  ############ NO-OBJECT LOSS ##################################################
  gt_conf_noobj_mask = tf.greater( gt_conf, tf.zeros_like(gt_conf) )
  gt_conf_noobj = tf.select( gt_conf_noobj_mask, tf.zeros_like(gt_conf), tf.ones_like(gt_conf)) # opposite of gt_conf

  j0_mask_noobj = tf.logical_not( j0_mask ) 
  j1_mask_noobj = tf.logical_not( j1_mask )

  pred_conf_j0_noobj = tf.select(j0_mask_noobj, pred_conf[:,0], tf.zeros_like(pred_conf_j0) )
  pred_conf_j1_noobj = tf.select(j1_mask_noobj, pred_conf[:,1], tf.zeros_like(pred_conf_j1) )

  noobj_ious = final_ious * gt_conf_noobj[:,0]
  gt_conf_j0_noobj = tf.select(j0_mask_noobj, noobj_ious, tf.zeros_like(pred_conf_j0) )
  gt_conf_j1_noobj = tf.select(j1_mask_noobj, noobj_ious, tf.zeros_like(pred_conf_j0) )

  noobj_loss_j0 = tf.square( pred_conf_j0_noobj - gt_conf_j0_noobj )
  noobj_loss_j1 = tf.square( pred_conf_j1_noobj - gt_conf_j1_noobj )

  noobj_loss = LAMBDA_NOOBJ * tf.reduce_sum(noobj_loss_j0 + noobj_loss_j1, reduction_indices=[0])
  ##############################################################################

  ############ COMPLETED CLASS LOSS ############################################
  class_loss = gt_classes - pred_classes # both are 49 x 20
  class_loss = tf.square(class_loss)
  class_loss = tf.reduce_sum(class_loss, reduction_indices=[1]) # along all classes
  class_loss = tf.mul( ind_obj_i, class_loss) # both are 49 x 1
  class_loss = tf.reduce_sum(class_loss, reduction_indices=[0]) # along all boxes
  #############################################################################

  return box_loss +class_loss #+ noobj_loss#obj_loss # + 
