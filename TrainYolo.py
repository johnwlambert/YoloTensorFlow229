# John Lambert, Matt Vilim, Konstantine Buhler
# CS 229, Stanford University
# December 15, 2016


# TO-DO:
# Verify Indicator Probabilities
# Compute p(c) = p(c | obj) * p(obj) before loss calc
# Compute mean image, subtract by it, Divide by standard deviation.
import pdb
from PIL import Image
import tensorflow as tf
import math
import numpy as np

import matplotlib.pyplot as plt

import matplotlib
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.patches as patches

from preprocess_data import *
from YOLO_TrainingNetwork import YOLO_TrainingNetwork
from YOLO_PlottingUtils import *
from YOLO_DataUtils import *
from YOLO_mAP_Evaluation import * # mean average precision utils

####### HYPERPARAMETERS ###########################################
TRAIN_DROP_PROB = 0.8
TEST_DROP_PROB = 1.0
#ARBITRARY_STOP_LOADING_IMS_NUMBER_FOR_DEBUGGING = 5011
NUM_VOC_IMAGES = 5011
TRAIN_SET_SIZE = int( math.floor( NUM_VOC_IMAGES * 0.8 )) 
BATCH_SIZE = 1
NUM_EPOCHS = 100
plot_yolo_grid_cells = False
plot_bbox_centerpoints = False
plot_im_bboxes = False # True
getPickledData = True
vocImagesPklFilename = 'VOC_AnnotatedImages.pkl'
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
NUM_CLASSES = len(CLASSES)
NUM_GRID = 7
NUM_BOX = 2
voc_data_path = '/Users/johnlambert/Documents/Stanford_2016-2017/CS_229/229CourseProject/YoloTensorFlow229/VOCdevkit_2012/'
numVar = 5

VAL_SET_SIZE = 501
TEST_SET_SIZE = 502
THRESH = 0.2
#######################################################################



def runTrainStep(yoloNet, annotatedImages,sess, step):
	"""
	INPUTS:
	-	yoloNet: Instance of YOLO_TrainingNetwork class (this is a CNN)
	-	annotatedImages: python list of annotated_image class objects
	-	sess: TensorFlow Session
	-	step: int, current training iteration
	OUTPUTS:
	-	N/A
	"""
	minibatchIms, minibatchGT = sampleMinibatch(annotatedImages, plot_yolo_grid_cells, plot_bbox_centerpoints)
	minibatchIms = np.expand_dims( minibatchIms, 0)
	gt_conf,gt_classes,ind_obj_i,gt_boxes_j0 = minibatchGT
	gt_conf = gt_conf.astype(np.float32)
	gt_classes = gt_classes.astype(np.float32)
	ind_obj_i = ind_obj_i.astype(np.float32)
	gt_boxes_j0 = gt_boxes_j0.astype(np.float32)

	## FEED DROPOUT 0.5 AT TRAIN TIME, 1.0 AT TEST TIME #######
	feed = { yoloNet.input_layer: minibatchIms , yoloNet.gt_conf : gt_conf, \
	yoloNet.gt_classes : gt_classes, yoloNet.ind_obj_i: ind_obj_i, \
	yoloNet.dropout_prob: TRAIN_DROP_PROB, yoloNet.gt_boxes_j0 : gt_boxes_j0 }

	trainLossVal = sess.run( yoloNet.loss ,feed_dict=feed ) # yoloNet.train_op
	#print 'Length of train loss: ', len(trainLossVal)
	print 'Training loss at step %d: %f' % (step,trainLossVal)

def runEvalStep( splitType, yoloNet, annotatedImages ,sess, epoch, saver, best_val_mAP ):
	"""
	INPUTS:
	-	yoloNet: Instance of YOLO_TrainingNetwork class (this is a CNN)
	-	annotatedImages: python list of annotated_image class objects
	-	sess: TensorFlow Session
	-	epoch: int, current training epoch
	OUTPUTS:
	-	N/A
	"""
	detections = []
	print '====> Evaluating: %s at epoch %d =====>' % (splitType, epoch)
	if splitType == 'val':
		data_set_size = VAL_SET_SIZE
	elif splitType == 'test':
		data_set_size = TEST_SET_SIZE
	for i in range( 500): #data_set_size ):
		minibatchIms, minibatchGT = sampleMinibatch(annotatedImages, plot_yolo_grid_cells, plot_bbox_centerpoints)
		minibatchIms = np.expand_dims( minibatchIms, 0)

		gt_conf,gt_classes,ind_obj_i,gt_boxes_j0 = minibatchGT
		gt_conf = gt_conf.astype(np.float32)
		gt_classes = gt_classes.astype(np.float32)
		ind_obj_i = ind_obj_i.astype(np.float32)
		gt_boxes_j0 = gt_boxes_j0.astype(np.float32)

		## FEED DROPOUT 0.5 AT TRAIN TIME, 1.0 AT TEST TIME #######
		feed = { yoloNet.input_layer: minibatchIms , yoloNet.gt_conf : gt_conf, \
		yoloNet.gt_classes : gt_classes, yoloNet.ind_obj_i: ind_obj_i, \
		yoloNet.dropout_prob: TEST_DROP_PROB, yoloNet.gt_boxes_j0 : gt_boxes_j0 }

		class_probs_giv_obj,confidences,boxes, lossVal = sess.run( [yoloNet.class_probs,yoloNet.confidences,yoloNet.bboxes, yoloNet.loss],feed_dict=feed )
		print 'Loss in %s split at epoch%d , iter %d : %f' % (splitType, epoch, i, lossVal)
		# NOW PROCESS THE PREDICTIONS HERE
		boxes = np.reshape(boxes, [NUM_GRID*NUM_GRID,NUM_BOX,4] )
		boxes = unnormalizeBoxes(boxes, minibatchIms)
		gt_boxes_j0 = unnormalizeGTBoxes(gt_boxes_j0,minibatchIms)
		#boxes = convertWH_to_xMax_yMax( boxes )
		class_probs = np.zeros((NUM_GRID*NUM_GRID,NUM_BOX,NUM_CLASSES))
		# We use a law of probability: prob(class) = prob(class|object) * prob(object)
														# 49 x 20 			49 x 2
		for i in range(NUM_BOX):
			for j in range(NUM_CLASSES):
				class_probs[:,i,j] = np.multiply(class_probs_giv_obj[:,j],confidences[:,i])

		class_probs = np.reshape( class_probs, [-1,20] )
		confidences = np.reshape( confidences, [-1,1] )
		boxes = np.reshape( boxes, [-1,4] )

		class_indices = np.argmax( class_probs, axis = 1)
		valid_detection_indices = np.where( confidences > 0.2 )[0]

		class_indices = class_indices[valid_detection_indices]
		confidences = confidences[valid_detection_indices]
		boxes = boxes[valid_detection_indices,:]
		#print 'Class Indices have shape: ', class_indices.shape
		# pdb.set_trace()
		# gt_nonzero_indices = []
		# for gridIdx in range(49):
		# 	if np.sum(gt_boxes_j0[gridIdx]) > 0.9:
		# 		gt_nonzero_indices.append( gridIdx)
		# gt_nonzero_indices = np.asarray( gt_nonzero_indices )
		# print 'NON_ZERO_INDICES: ', gt_nonzero_indices
		gt_nonzero_indices = np.where( np.sum(gt_classes, axis = 1) > 0.9 )[0]
		gt_boxes_j0 = gt_boxes_j0[gt_nonzero_indices]
		#print 'GT_classes looks like: ', gt_classes
		gt_classes = gt_classes[gt_nonzero_indices]
		#print 'Shrunken gt_classes: ', gt_classes
		gt_classes = np.argmax( gt_classes, axis = 1)
		#print 'argmaxed gt_classes: ', gt_classes

		detections.append( { 'pred_classes':class_indices, 'confidences':confidences,\
			'bboxes':boxes, 'gt_boxes_j0':gt_boxes_j0, 'gt_classes': gt_classes, 'im':minibatchIms   })

	# BB, BBGT = convertPredsAndGTs(bboxes, class_probs, confidences )
	mAP = computeMeanAveragePrecision(detections, splitType)
	# save some of the plots just as a sanity check along the way
	print '==> Current mAP: ', mAP, ' ===>'
	if (splitType == 'val') and (mAP > best_val_mAP):
		#saver.save(sess, './YOLO_Trained.weights' )
		best_val_mAP = mAP
	# plot_detections_on_im( imread(self.image_path),probs,confidences,bboxes,classes)

	# Each grid cells also predicts conditional class probabilities, Pr(Classi |Object). 
	# These probabilities are conditioned on the grid cell containing an object.
	# Why not just at test time do cross product?


def unnormalizeBoxes(boxes, im):
	"""
	INPUTS:
	-	boxes: NUM_GRID*NUM_GRID,NUM_BOX,4
	OUTPUTS:
	-	boxes: but scaled to image, and 

		x_cent,y_cent,w,h
	"""
	im = np.squeeze( im )
	imWidth = im.shape[1]
	imHeight = im.shape[0]
	grid_width = int(np.floor(imWidth / 7.))
	grid_height = int(np.floor(imHeight / 7.))

	for row in range(7):
		for col in range(7):
			gridIdx = (row*7) + col
			grid_x_offset = col * grid_width
			grid_y_offset = row * grid_height
			for boxIdx in range(2):
				# box_x_offset = boxes[gridIdx,j,0] * grid_width
				# box_y_offset = boxes[gridIdx,j,1] * grid_height
				# boxes[gridIdx,j,0] = box_x_offset + grid_x_offset
				# boxes[gridIdx,j,1] = box_y_offset + grid_y_offset
				# box_sqrt_w
				x_cent = (boxes[gridIdx,boxIdx,0] + col) / 7.0 * imWidth
				y_cent = (boxes[gridIdx,boxIdx,1] + row) / 7.0 * imHeight
				w = (boxes[gridIdx,boxIdx,2]**2) * imWidth * 1.0
				h = (boxes[gridIdx,boxIdx,3]**2) * imHeight * 1.0
				xmin = max(0, x_cent - w/2. )
				xmax = min( x_cent + w/2., imWidth-1 )
				ymin = max(0, y_cent - h/2. )
				ymax = min( y_cent + h/2., imHeight-1 )
				boxes[gridIdx,boxIdx,:] = np.array([xmin,ymin,xmax,ymax])
	# now have xcent,ycent,w,h
	# convert to xmin,xmax,ymin,ymax

	return boxes


def unnormalizeGTBoxes(boxes,im):
	im = np.squeeze( im )
	imWidth = im.shape[1]
	imHeight = im.shape[0]
	grid_width = int(np.floor(imWidth / 7.))
	grid_height = int(np.floor(imHeight / 7.))

	for row in range(7):
		for col in range(7):
			gridIdx = (row*7) + col
			x_cent = (boxes[gridIdx,0] + col) / 7.0 * imWidth
			y_cent = (boxes[gridIdx,1] + row) / 7.0 * imHeight
			w = (boxes[gridIdx,2]**2) * imWidth * 1.0
			h = (boxes[gridIdx,3]**2) * imHeight * 1.0
			xmin = max(0, x_cent - w/2. )
			xmax = min( x_cent + w/2., imWidth-1 )
			ymin = max(0, y_cent - h/2. )
			ymax = min( y_cent + h/2., imHeight-1 )
			boxes[gridIdx,:] = np.array([xmin,ymin,xmax,ymax])
	return boxes

# def convertWH_to_xMax_yMax( boxes ):
# 	"""
# 	INPUTS:
# 	-	boxes: n-d array contains [xmin,ymin,w,h] columns
# 	OUTPUTS:
# 	-	boxes: n-d array contains [xmin,ymin,xmax,ymax] columns
# 	"""
# 	xmin = boxes[:,:,0]
# 	ymin = boxes[:,:,1]
# 	boxes[:,:,2] += xmin # xmin + w = xmax
# 	boxes[:,:,3] += ymin # ymin + h = ymax
# 	return boxes


if __name__ == '__main__':

	checkpoint_path = '/Users/johnlambert/Documents/Stanford_2016-2017/CS_229/229CourseProject/YoloTensorFlow229/yolo.ckpt'

	best_val_mAP = -1 * float('inf')
	annotatedImages = getData(getPickledData,vocImagesPklFilename)
	trainData, valData, testData = separateDataSets(annotatedImages)
	if plot_im_bboxes == True:
		plotGroundTruth(annotatedImages)
	yoloNet = YOLO_TrainingNetwork( use_pretrained_weights = False)
	numItersPerEpoch = TRAIN_SET_SIZE / BATCH_SIZE

	# t = 1
	# beta1 = tf.convert_to_tensor(0.9)
	# beta2 = tf.convert_to_tensor(0.999)
	# beta1_power = beta1**t
	# beta2_power = beta2**t

	saver = tf.train.Saver() # {'beta2_power':beta2_power} )
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		saver.restore(sess, checkpoint_path)
		for epoch in range(20):#NUM_EPOCHS):
			print '====> Starting Epoch %d =====>' % (epoch)
			for step in range( 1): # numItersPerEpoch):
				runTrainStep(yoloNet, annotatedImages ,sess, step)
			runEvalStep( 'val', yoloNet, annotatedImages ,sess, epoch, saver, best_val_mAP)
		# After all training complete
		saver.restore(sess, './YOLO_Trained.weights' )
		runEvalStep( 'test', yoloNet, annotatedImages ,sess, epoch, None, None)


