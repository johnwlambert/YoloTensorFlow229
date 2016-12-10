# John Lambert, Matt Vilim, Konstantine Buhler
# CS 229, Stanford University
# December 2016


# TO-DOs
# HOW TO CALCULATE GT CONFIDENCES? What is GT_{PRED}^{IOU} ?

import math
import numpy as np
from preprocess_data import *
from scipy.misc import imread
import matplotlib.pyplot as plt
import cPickle as pickle

import matplotlib
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.patches as patches

### FOR DECIDING WHICH BBOXES CORRESPOND TO WHICH GRID CELLS ##########
NO_IMAGE_FLAG = 0 # COULD MAKE THIS -9999, ETC.
CONTAINS_IMAGE_FLAG = 1
#######################################################################
plot_yolo_grid_cells = False
plot_bbox_centerpoints = False
plot_im_bboxes = False # True
getPickledData = True
vocImagesPklFilename = 'VOC_AnnotatedImages.pkl'
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
voc_data_path = '/Users/johnlambert/Documents/Stanford_2016-2017/CS_229/229CourseProject/YoloTensorFlow229/VOCdevkit/'
S = 7
numClasses = len(classes) # 20 for VOC, later will change for MSCOCO, ImageNet, etc.
B = 2.
numVar = 5
#######################################################################
classnameToIdxDict = {}
for i,classname in enumerate(classes):
	classnameToIdxDict[classname] = i



def getData():
	"""
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




def plotGroundTruth(annotatedImages):
	for imIdx, annotatedImage in enumerate(annotatedImages):
		bboxes = annotatedImage.bounding_boxes
		im = imread( annotatedImage.image_path )
		plotBBoxes(bboxes,im, imIdx)
		if imIdx > 100:
			quit()

def plotBBoxes(bboxes, im, imIdx):
	imWidth = im.shape[1]
	imHeight = im.shape[0]
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.imshow(im, aspect='equal')
	for bbox in bboxes:
		x = bbox.x_min
		y = bbox.y_min
		w = bbox.w
		h = bbox.h
		class_name = bbox.category
		x_cent = x + w/2.
		y_cent = y + h/2.
		x = x_cent
		y = y_cent
		left = max(0, x - w/2. )
		right = min( x + w/2., imWidth-1 )
		top = max(0, y - h/2. )
		bot = min( y + h/2., imHeight-1 )
		ax.add_patch(
	        plt.Rectangle((left, top),
	                      right-left,
	                      bot-top, fill=False,
	                      edgecolor='red', linewidth=3.5)
	        )
		ax.text(left, top-2,
	            '{:s}'.format(class_name ),
	            bbox=dict(facecolor='blue', alpha=0.5),
	            fontsize=14, color='white')
	plt.draw()
	plt.tight_layout()
	plt.savefig( 'Image_%d.png' % imIdx)
	plt.gcf().set_size_inches(15, 12)


def plotGridCellsOnIm(im,ax):
	"""
	Show 7x7 YOLO detection grid on image.
	"""
	for row in range(S):
		for col in range(S):
			imHeight = im.shape[0]
			imWidth = im.shape[1]
			left = row * (1. * imWidth /S )
			right = (row+1) * (1. * imWidth /S)
			top = col * (1. * imHeight/S)
			bot = (col + 1) * (1. * imHeight/S)
			ax.add_patch(
		        plt.Rectangle((left, top),
		                      right-left,
		                      bot-top, fill=False,
		                      edgecolor='red', linewidth=3.5)
		        )
		plt.draw()


def makeGroundTruthTensors(annotatedImages):
	"""
	If there's a bbox center in that grid cell, process and store bbox info there.
	20 CLASS PROBS , X Y W H C1 , X Y W H C2
	"""
	listOfImPaths = []
	listOfGTs = []
	print 'Num. Images: ', len(annotatedImages)
	for imNum, annotatedImage in enumerate(annotatedImages):
		print 'Image: ', imNum
		image_path = annotatedImage.image_path
		im = imread(image_path)
		### PLOTTING ########
		if plot_yolo_grid_cells or plot_bbox_centerpoints:
			fig, ax = plt.subplots(figsize=(8, 8))
			ax.imshow(im, aspect='equal')
			plotGridCellsOnIm(im,ax)
		gt = np.ones( (S,S,numClasses+(B*numVar) )) * NO_IMAGE_FLAG
		occupiedSlot = np.zeros((S,S,B))
		# We limit to two bounding boxes per grid cell. 
		# For each image, tell which grid cells are responsible for which bboxes
		for i, bbox in enumerate(annotatedImage.bounding_boxes):
			x_cent = bbox.x_min + bbox.w / 2.
			y_cent = bbox.y_min + bbox.h / 2.
			gridCellWidth = im.shape[1] / 7.
			gridCellHeight = im.shape[0] / 7.
			gridCellRow = math.floor( y_cent / gridCellHeight )
			gridCellCol = math.floor( x_cent / gridCellWidth )
			normalizedXCent=(x_cent-(gridCellCol * gridCellWidth))/gridCellWidth
			normalizedYCent=(y_cent-(gridCellRow * gridCellHeight))/gridCellHeight
			normalizedW = bbox.w * 1.0 / im.shape[1] # dividing by im width
			normalizedH = bbox.h * 1.0 / im.shape[0] # dividing by im height
			gridCellRow = int(gridCellRow)
			gridCellCol = int(gridCellCol)
			CONF = 99999 ## UNKNOWN HOW TO DO THIS PART
			classIdx = classnameToIdxDict[ bbox.category ] # convert string to int
			if occupiedSlot[gridCellRow,gridCellCol,0] == NO_IMAGE_FLAG:
				j = 0 # 2nd box slot for this grid cell
				gt[gridCellRow,gridCellCol,classIdx] = 1
				gt[gridCellRow,gridCellCol,numClasses+(j*numVar)+0 ] = normalizedXCent
				gt[gridCellRow,gridCellCol,numClasses+(j*numVar)+1 ] = normalizedYCent
				gt[gridCellRow,gridCellCol,numClasses+(j*numVar)+2 ] = normalizedW
				gt[gridCellRow,gridCellCol,numClasses+(j*numVar)+3 ] = normalizedH
				gt[gridCellRow,gridCellCol,numClasses+(j*numVar)+4 ] = CONF
				occupiedSlot[gridCellRow,gridCellCol,j] = CONTAINS_IMAGE_FLAG
			elif occupiedSlot[gridCellRow,gridCellCol,1] == NO_IMAGE_FLAG:
				j = 1 # 2nd box slot for this grid cell
				gt[gridCellRow,gridCellCol,classIdx] = 1
				gt[gridCellRow,gridCellCol,numClasses+(j*numVar)+0 ] = normalizedXCent
				gt[gridCellRow,gridCellCol,numClasses+(j*numVar)+1 ] = normalizedYCent
				gt[gridCellRow,gridCellCol,numClasses+(j*numVar)+2 ] = normalizedW
				gt[gridCellRow,gridCellCol,numClasses+(j*numVar)+3 ] = normalizedH
				gt[gridCellRow,gridCellCol,numClasses+(j*numVar)+4 ] = CONF
				occupiedSlot[gridCellRow,gridCellCol,j] = CONTAINS_IMAGE_FLAG
			else:
				print 'No more room in grid cell for this bbox.'

		if plot_bbox_centerpoints == True:
			plt.scatter(x_cent,y_cent)
		if plot_bbox_centerpoints or plot_yolo_grid_cells:
			plt.tight_layout()
			plt.show()
			plt.gcf().set_size_inches(15, 12)

		listOfImPaths.append( image_path )
		listOfGTs.append( gt )
	return listOfImPaths, listOfGTs


def lossFunction():
	"""
	How to change to batch processing, so not processing one by one?
	"""

if __name__ == '__main__':
	annotatedImages = getData()
	listOfImPaths, listOfGTs = makeGroundTruthTensors(annotatedImages)
	if plot_im_bboxes == True:
		plotGroundTruth(annotatedImages)
	#minibatch = sampleMinibatch(annotatedImages)

	# Compute mean image, subtract by it
	# Divide by standard deviation 



