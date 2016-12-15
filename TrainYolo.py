# John Lambert, Matt Vilim, Konstantine Buhler
# CS 229, Stanford University
# December 15, 2016


# TO-DO:
# Verify Indicator Probabilities
# Compute p(c) = p(c | obj) * p(obj) before loss calc
# Compute mean image, subtract by it, Divide by standard deviation.

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
voc_data_path = '/Users/johnlambert/Documents/Stanford_2016-2017/CS_229/229CourseProject/YoloTensorFlow229/VOCdevkit_2012/'
numVar = 5

VAL_SET_SIZE = 501
TEST_SET_SIZE = 502
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

	trainLossVal, _ = sess.run( [yoloNet.loss, yoloNet.train_op],feed_dict=feed )
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
	for i in range( data_set_size ):
		minibatchIms, minibatchGTs = sampleMinibatch(annotatedImages, plot_yolo_grid_cells, plot_bbox_centerpoints)
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

		class_probs,confidences,bboxes = sess.run( [self.class_probs,self.confidences,self.bboxes],feed_dict=feed )
		detections.append( { class_given_obj:class_probs, confidences:confidences,\
			bboxes:bboxes, gt_boxes_j0:gt_boxes_j0, gt_classes: gt_classes   })
		# NOW PROCESS THE PREDICTIONS HERE
	# BB, BBGT = convertPredsAndGTs(bboxes, class_probs, confidences )
	mAP = computeMeanAveragePrecision(detections, splitType)
	# save some of the plots just as a sanity check along the way

	if (splitType == 'val') and (mAP > best_val_mAP):

		saver.save(session, './%s/YOLO_Trained.weights' % (new_dir_path ) )
		best_val_mAP = mAP
	# plot_detections_on_im( imread(self.image_path),probs,confidences,bboxes,classes)

	# Each grid cells also predicts conditional class probabilities, Pr(Classi |Object). 
	# These probabilities are conditioned on the grid cell containing an object.
	# Why not just at test time do cross product?


if __name__ == '__main__':
	"""
	"""
	best_val_mAP = -1 * float('inf')
	annotatedImages = getData(getPickledData,vocImagesPklFilename)
	trainData, valData, testData = separateDataSets(annotatedImages)
	if plot_im_bboxes == True:
		plotGroundTruth(annotatedImages)
	yoloNet = YOLO_TrainingNetwork( use_pretrained_weights = False)
	numItersPerEpoch = TRAIN_SET_SIZE / BATCH_SIZE
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(NUM_EPOCHS):
			print '====> Starting Epoch %d =====>' % (epoch)
			for step in range(numItersPerEpoch):
				runTrainStep(yoloNet, annotatedImages ,sess, step)
			runEvalStep( 'val', yoloNet, annotatedImages ,sess, epoch, saver, best_val_mAP)
		# After all training complete
		saver.restore
		runEvalStep( 'test', yoloNet, annotatedImages ,sess, epoch, None, None)


