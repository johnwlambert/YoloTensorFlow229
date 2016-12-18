# John Lambert, Matt Vilim, Konstantine Buhler
# Dec 15, 2016
# For Training YOLO

import math
import cPickle as pickle
import numpy as np
from scipy.misc import imread, imresize

from YOLO_PlottingUtils import *
from YOLO_CoverageMap import *
from preprocess_data import *
###### HYPERPARAMETERS #######
voc_data_path = 'home/johnwl/YoloTensorFlow229/VOCdevkit/'
PERCENT_TRAIN_SET = 0.8
PERCENT_VAL_SET = 0.1
PERCENT_TEST_SET = 0.1
S = 7
B = 2.
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
NUM_CLASSES = len(CLASSES) # 20 for VOC, later will change for MSCOCO, ImageNet, etc.
classnameToIdxDict = {}
for i,classname in enumerate(CLASSES):
	classnameToIdxDict[classname] = i

### FOR DECIDING WHICH BBOXES CORRESPOND TO WHICH GRID CELLS ##########
NO_IMAGE_FLAG = 0 # COULD MAKE THIS -9999, ETC.
CONTAINS_IMAGE_FLAG = 1
#######################################################################


def getData(getPickledData,vocImagesPklFilename):
	"""
	INPUTS:
	-	N/A
	OUTPUTS:
	-	annotatedImages: Python list of annotated_image class objects
	##################################################################
	We retrieve data of the following format:
	class annotated_image:
	    def __init__(self, image_path):
	        self.image_path = image_path
	        # list of class bounding boxes
	        self.bounding_boxes = []
	class bounding_box:
	    def __init__(self, x_min, y_min, w, h, category):
	        self.x_min = x_min
	        self.y_min = y_min
	        self.w = w
	        self.h = h
	        self.category = category
	"""
	if getPickledData:
		with open( vocImagesPklFilename, 'rb') as f:
			annotatedImages = pickle.load(f)
	else:
		annotatedImages = preprocess_data(voc_data_path)
		with open( vocImagesPklFilename, 'wb') as f:
			pickle.dump(annotatedImages,f)
	return annotatedImages


def sampleMinibatch(annotatedImages, plot_yolo_grid_cells, plot_bbox_centerpoints):
	"""
	If center of a bounding box falls into a grid cell, that grid cell is 
	responsible for detecting that bounding box. So I store that bbox info
	for that particular grid cell.
	INPUTS:
	-	annotatedImages: a Python list of annotated_image class objects
	OUTPUTS:
	-	minibatchIms: n-d array
	-	minibatchGTs: n-d array, shape [num_gtbox, 73=NUM_CLASSES+4+49] (gt labels)
	"""
	# print '=====> Sampling Ground Truth tensors for %d images ====>' % (len(annotatedImages))

	batch_size = 1
	mask = np.random.choice( len(annotatedImages) , batch_size ) # RANDOM SAMPLE OF THE INDICES
	imNum = mask[0] 
	annotatedImage = annotatedImages[imNum]
	image_path = annotatedImage.image_path
	img = imread(image_path)
	img = imresize(img, (448, 448))
	img = img[...,::-1]
	# darknet scales color values from 0 to 1
	# https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/src/image.c#L469
	img = (img / 255.0)
	   
	# if imNum > ARBITRARY_STOP_LOADING_IMS_NUMBER_FOR_DEBUGGING:
	# 	break

	gt_classes = np.zeros((49,20))
	gt_conf = np.zeros((49,4))
	ind_obj_i = np.zeros((49))
	gt_boxes_j0 = np.zeros((49,4))
	
	im = imread(image_path)
	if plot_yolo_grid_cells or plot_bbox_centerpoints:
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.imshow(im, aspect='equal')
		plotGridCellsOnIm(im,ax)
	gt = []
	occupiedSlot = np.zeros((S,S,B))
	# We limit to two bounding boxes per grid cell. 
	# For each image, tell which grid cells are responsible for which bboxes
	for i, bbox in enumerate(annotatedImage.bounding_boxes):
		x_cent = bbox.x_min + bbox.w / 2.
		y_cent = bbox.y_min + bbox.h / 2.
		normalizedXCent, normalizedYCent, gridCellRow, gridCellCol = normXYToGrid(x_cent,y_cent,im)
		normalizedW = bbox.w * 1.0 / im.shape[1] # dividing by im width
		normalizedH = bbox.h * 1.0 / im.shape[0] # dividing by im height
		gridCellRow = int(gridCellRow)
		gridCellCol = int(gridCellCol)
		classIdx = classnameToIdxDict[ bbox.category ] # convert string to int
		coverageMap = computeCoverageMap(im, bbox) # Returns 49x1 coverage map

  		# indicating if that grid cell contains any object
		ind_obj_i = np.logical_or( ind_obj_i, coverageMap).astype(np.int64)

		if occupiedSlot[gridCellRow,gridCellCol,0] == NO_IMAGE_FLAG:
			j = 0 # 2nd box slot for this grid cell
			gt_classes[gridCellRow * 7 + gridCellCol, classIdx ] = 1
			xywh = np.array([ normalizedXCent, normalizedYCent, math.sqrt(normalizedW), math.sqrt(normalizedH) ])
			bboxGT = xywh # coverage map is the confidence
			gt_boxes_j0[ gridCellRow * 7 + gridCellCol] = bboxGT
			occupiedSlot[gridCellRow,gridCellCol,j] = CONTAINS_IMAGE_FLAG

			# values in each of 4 columns are identical (tiled/repmatted). Object here at cell i!
			gt_conf[gridCellRow * 7 + gridCellCol,:] = np.ones((1,4))

  		# IGNORING J=1 IN SIMPLIFIED, CURRENT CASE
		# elif occupiedSlot[gridCellRow,gridCellCol,1] == NO_IMAGE_FLAG:
		# 	j = 1 # 2nd box slot for this grid cell
		# 	classGTs = np.zeros(NUM_CLASSES)
		# 	classGTs[classIdx] = 1
		# 	xywh = np.array([ normalizedXCent, normalizedYCent, math.sqrt(normalizedW), math.sqrt(normalizedH) ])
		# 	gt = np.hstack(( classGTs , xywh, coverageMap)) # coverage map is the confidence
		# 	occupiedSlot[gridCellRow,gridCellCol,j] = CONTAINS_IMAGE_FLAG
		else:
			#print 'In Image %d, no more room in some grid cell for this bbox.' % (imNum)
			pass
	if plot_bbox_centerpoints == True:
		plt.scatter(x_cent,y_cent)
	if plot_bbox_centerpoints or plot_yolo_grid_cells:
		plt.tight_layout()
		plt.show()
		plt.gcf().set_size_inches(15, 12)



	minibatchIms = img
	return minibatchIms, (gt_conf,gt_classes,ind_obj_i,gt_boxes_j0)


def normXYToGrid(x_cent,y_cent,im):
	"""
	I normalize the x,y coordinates to be the offset from top-left grid corner,
	normalized to size of grid.
	The (x, y) coordinates represent the center of the box relative to the 
	bounds of the grid cell. 
	The width and height are predicted relative to the whole image. 
	In contrast, w and h from the GTs are normalized to the image.
	INPUTS:
	-	x_cent: float, x-coordinate of center of a bounding box
	-	y_cent: float, y-coordinate of center of a bounding box
	-	im: n-d array, representing an image of shape [None,None,3]
	OUTPUTS:
	-	normalizedXCent: float, x-coordinate of center of bbox, normalized to grid cell size
	-	normalizedYCent: float, y-coordinate of center of bbox, normalized to grid cell size
	-	gridCellRow: int, index of row in which we find the bbox center
	-	gridCellCol: int, index of column in which we find the bbox center
	"""
	gridCellWidth = im.shape[1] / 7.
	gridCellHeight = im.shape[0] / 7.
	gridCellRow = math.floor( y_cent / gridCellHeight )
	gridCellCol = math.floor( x_cent / gridCellWidth )
	normalizedXCent=(x_cent-(gridCellCol * gridCellWidth))/gridCellWidth
	normalizedYCent=(y_cent-(gridCellRow * gridCellHeight))/gridCellHeight
	return normalizedXCent, normalizedYCent, gridCellRow, gridCellCol



def separateDataSets(annotatedImages):
	"""
	INPUTS:
	-	annotatedImages: python list of annotated_image class objects
	OUTPUTS:
	-	trainData: python list of annotated_image class objects
	-	valData: python list of annotated_image class objects
	-	testData: python list of annotated_image class objects
	"""
	trainNum = int( math.floor( PERCENT_TRAIN_SET * len(annotatedImages) ) )
	valNum = int( math.floor( PERCENT_VAL_SET * len(annotatedImages) ) )
	testNum = int( len(annotatedImages) - trainNum - valNum )
	print '===> Placing %f %% into training set, %f %% into val set, %f %% into test set' % (PERCENT_TRAIN_SET,PERCENT_VAL_SET,PERCENT_TEST_SET)
	print '===> %d ims in training set, %d ims in val set, %d ims in test set' % (trainNum,valNum,testNum)
	trainEndIdx = trainNum
	valEndIdx = trainNum + valNum
	testIndIdx = trainNum + valNum + testNum
	trainData = annotatedImages[0:trainEndIdx]
	valData = annotatedImages[trainEndIdx:valEndIdx]
	testData = annotatedImages[valEndIdx:testIndIdx]
	return trainData, valData, testData
