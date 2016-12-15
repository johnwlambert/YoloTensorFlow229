# John Lambert, Matt Vilim, Konstantine Buhler
# YOLO_CoverageMap
# Compute ground truth object coverage map across 7x7 grid cells, indicator variable
# Dec 15, 2016

import numpy as np
from scipy.misc import imread, imresize
import matplotlib
import matplotlib.pyplot as plt

######## HYPERPARAMETERS ########
NUM_GRID = 7.
SHOW_COVERAGE_MAP = False # True # 
#################################

def computeCoverageMapSimplified(im,gtBbox):
	"""
	INPUTS:
	-	im: n-d array containing resized [448,448,3] image
	-	gtBbox: (4x1) n-d array containing bounding box info [xmin,ymin,xmax,ymax]
	OUTPUTS:
	-	coverageMap: (7x7) n-d array containing pr(object)=1 if portion of obj lies in that grid cell
	"""
	GRID_SIZE = im.shape[0] / NUM_GRID

	xmin = gtBbox[0]
	ymin = gtBbox[1]
	xmax = gtBbox[2]
	ymax = gtBbox[3]

	leftGridEdge = (xmin - (xmin % GRID_SIZE))/ GRID_SIZE
	topGridEdge = (ymin - (ymin % GRID_SIZE))/ GRID_SIZE
	rightGridEdge = (xmax + ( (448-xmax) % GRID_SIZE) )/GRID_SIZE
	botGridEdge = (ymax + ( (448-ymax) % GRID_SIZE) )/GRID_SIZE

	leftGridEdge = max( leftGridEdge, 0 )
	topGridEdge = max( topGridEdge, 0 )
	rightGridEdge = min( rightGridEdge, 7 )
	botGridEdge = min( botGridEdge, 7 )

	coverageMap = np.zeros((7,7))
	for col in range(7):
		for row in range(7):
			if (col >= leftGridEdge) and (col<rightGridEdge):
				if (row >= topGridEdge) and (row<botGridEdge):
					coverageMap[row,col] = 1

	return coverageMap


def computeCoverageMap(im, bbox):
	"""
	I use xmin,ymin,xmax,ymax to find an object coverage map for each ground truth box.
	INPUTS:
	-	im: n-d array, NOT-yet-resized-to-448-448-3 image
	-	bbox: 
	OUTPUTS:
	-	coverageMap: n-d array, 49x1
	"""
	im = im.astype(np.float32)
	gtBbox = np.array([bbox.x_min,bbox.y_min,bbox.x_min+bbox.w,bbox.y_min+bbox.h]) 
	xScale = 448.0 / im.shape[1]
	yScale = 448.0 / im.shape[0]
	gtBbox[0] *= xScale
	gtBbox[2] *= xScale
	gtBbox[1] *= yScale
	gtBbox[3] *= yScale
	im = imresize(im, [448,448])
	coverageMap = computeCoverageMapSimplified(im,gtBbox)
	if SHOW_COVERAGE_MAP:
		fig, ax = plt.subplots(figsize=(8, 8))
		plotGridCellsOnIm(im,ax)
		im = Image.fromarray(im,'RGB')
		ax.imshow(im, aspect='equal')
		print coverageMap
		plt.scatter(gtBbox[0],gtBbox[1])
		plt.scatter(gtBbox[2],gtBbox[3])
		plt.tight_layout()
		plt.show()

	coverageMap = np.reshape( coverageMap, [-1])
	return coverageMap
