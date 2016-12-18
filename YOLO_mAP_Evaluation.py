# John Lambert, Matt Vilim, Konstantine Buhler
# Dec 15, 2016

import os
import numpy as np
import pdb
import matplotlib.pyplot as plt

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

EPSILON = 1e-5

def computeMeanAveragePrecision(detections, splitType):
	"""
	INPUTS:
	-	detections: python list of objects with fields: class_given_obj, confidences, bboxes
	OUTPUTS:
	-	mAP: float
	For each class, we compute average precision (AP)
	This score corresponds to the area under the precision-recall curve.
	The mean of these numbers is the mAP.
	"""

	#plotDets(detections)
	aps = []
	for classIdx, cls in enumerate(CLASSES):
		pdb.set_trace()
		#print 'Evaluate class: ', cls, ' with class index: ', classIdx
		rec, prec, ap = matchGTsAndComputePrecRecallAP(classIdx,detections,iouthresh=0.5)
		if ap is not None:
			aps += [ap]
			print('AP for {} = {:.4f}'.format(cls, ap))
	print('Mean AP = {:.4f}'.format(np.mean(aps)))
	print('~~~~~~~~')
	print('Results:')
	for ap in aps:
	    print('{:.3f}'.format(ap))
	print('{:.3f}'.format(np.mean(aps)))
	mAP = np.mean(aps)
	return mAP

def naiveAPCalculation(rec,prec):
	"""
	Take sum of P(k) * \Delta recall(k)
	"""
	deltaRecall = []
	rec = np.insert(rec,0,0) # SYNTAX: np.insert(Arr,idxToInsertAt,valuesToInsert)
	for i in range(1,rec.shape[0]):
		deltaRecall.append( rec[i] - rec[i-1] ) # find differences
	deltaRecall = np.array(deltaRecall)
	ap = np.dot( deltaRecall,prec)
	return ap


def voc_ap(rec, prec):
    """ 
    ap = voc_ap(rec, prec)
    Compute VOC AP given precision and recall.
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py#L31
    """
    # first append sentinel values at the end
    # Integrating from 0 to 1
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def intersection(BBGT,bb):
	"""
	Compute Intersection
	Why do they inflate numbers by adding +1.? I don't
	https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py#L168
	"""
	#print BBGT[ 0], bb[0]
	ixmin = np.maximum(BBGT[ 0], bb[0])
	iymin = np.maximum(BBGT[ 1], bb[1])
	ixmax = np.minimum(BBGT[ 2], bb[2])
	iymax = np.minimum(BBGT[ 3], bb[3])
	iw = np.maximum(ixmax - ixmin , 0.)
	ih = np.maximum(iymax - iymin , 0.)
	inters = iw * ih
	return inters

def union(BBGT,bb,inters):
	"""
	Why do they inflate numbers by adding +1.? I don't
	https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py#L173
	INPUTS:
	-	
	OUTPUTS:
	-	
	"""
	union = ((bb[2] - bb[0] ) * (bb[3] - bb[1] ) + \
		(BBGT[ 2] - BBGT[ 0] ) * \
		(BBGT[ 3] - BBGT[ 1] ) - inters)
	return union

#Returns the intersection over union of two rectangles, a and b, where each is an array [x,y,w,h]
def computeIOU(BBGT,bb): 
	maxIOU = 0
	for i,gtBbox in enumerate(BBGT):
		gtBbox = np.squeeze(gtBbox)
		#print gtBbox
		inters = intersection(gtBbox,bb) * 1.0
		#print 'Intersection: ', inters
		uni = union(gtBbox,bb,inters)
		#print 'Union: ', uni
		iou =   inters/uni
		maxIOU = max(iou,maxIOU)
	return maxIOU


def unitTestAP():
	prec = np.array([1.,1.,.67,.75,.6,0.67,4/7., .5, 4/9., 0.5])
	rec = np.array([.2,.4,.4,.6,.6,.8,.8,.8,.8,1.])
	assert (voc_ap(rec, prec) - naiveAPCalculation(rec,prec)) < EPSILON

def plotRectangle(bbox,ax,class_name, edgecolor):
	"""
	"""
	xmin = bbox[0]
	ymin = bbox[1]
	xmax = bbox[2]
	ymax = bbox[3]
	left = xmin
	right = xmax
	top = ymin
	bot = ymax
	ax.add_patch(
        plt.Rectangle((left, top),
                      right-left,
                      bot-top, fill=False,
                      edgecolor=edgecolor, linewidth=3.5)
        )
	ax.text(left, top-2,
            '{:s}'.format(class_name ),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14, color='white')


def plotDets(detections):

	for imIdx,im in enumerate(detections):

		image = im['im']
		image = np.squeeze(image)
		imWidth = image.shape[1]
		imHeight = image.shape[0]
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.imshow(image, aspect='equal')

		# look at each of the predictions
		for j in range( len(im['bboxes'])):
			pred_class = im['pred_classes'][j]
			bbox = im['bboxes'][j] 
			plotRectangle(bbox,ax,CLASSES[pred_class], 'red')
		for j in range( len(im['gt_boxes_j0'])):
			gt_class = im['gt_classes'][j]
    		gt_bbox = im['gt_boxes_j0'][j]
    		plotRectangle(bbox,ax,CLASSES[gt_class], 'green')

		plt.draw()
		plt.tight_layout()
		#plt.show()
		plt.savefig( 'Image_%d.png' % imIdx)
		plt.gcf().set_size_inches(15, 12)


def matchGTsAndComputePrecRecallAP(cls,detections,iouthresh=0.5):
	"""
	INPUTS:
	-	BB: predicted bounding boxes
	-	BBGT: predicted bounding boxes, BBGT = R['bbox'].astype(float)
	OUTPUTS:
	-	rec: recall
	-	prec: precision
	-	ap: average precision
	A bounding box reported by an algorithm is considered
	correct if its area intersection over union with a ground 
	truth bounding box is beyond 50%. If a lot of closely overlapping 
	bounding boxes hitting on a same ground truth, only one of
	them is counted as correct, and all the others are treated as false alarms
	"""
	BBGT = []
	confidence = []
	BB = []
	BB_im_ids = []

	imIdToNumGtInsideImDict = {}
	# FIND THE PREDICTIONS FOR JUST --ONE CLASS--, E.G. AIRPLANE
	for imIdx,im in enumerate(detections):
		# look at each of the predictions
		for j in range( len(im['bboxes'])):
			if im['pred_classes'][j] == cls:
				BB.append( im['bboxes'][j] )
				confidence.append( im['confidences'][j] )
				BB_im_ids.append( imIdx )
		for j in range( len(im['gt_boxes_j0'])):
			num_of_this_class_in_this_im = 0
    		if im['gt_classes'][j] == cls:
    			BBGT.append( im['gt_boxes_j0'][j] )
    			num_of_this_class_in_this_im += 1
    		imIdToNumGtInsideImDict[imIdx] = num_of_this_class_in_this_im
	BBGT = np.asarray(BBGT)
	if BBGT.shape[0] == 0:
		return None,None,None
	confidence = np.asarray(confidence)
	BB = np.asarray(BB)
	if BB.shape[0] > 0:
		print 'At least one detection made.'
	#print 'BBGT has shape: ', BBGT.shape
	BB_im_ids = np.asarray(BB_im_ids)

	# sort by confidence
	sorted_ind = np.argsort(confidence) # sort according to highest confidence
	sorted_scores = np.sort(confidence) # sort according to highest confidence

	prec = []
	rec = []
	if BB.shape[0] > 0:
		print 'One Valid Detection.'
		#pdb.set_trace()
		BB = BB[sorted_ind]
		BB_im_ids = BB_im_ids[sorted_ind]
		#print 'BB.shape: ', BB.shape
		BB = np.squeeze(BB,axis=1)
		#print 'BB.shape: ', BB.shape
		BB_im_ids = np.squeeze(BB_im_ids, axis=1)
		#get k highest rank (confidence score) things
		for k in range( 1,BB.shape[0]+1 ):
			# then choose next most confident thing
			#get unique imIds inside current rank list
			total_cls_reports = k
			#print 'BB_im_ids has shape: ', BB_im_ids.shape
			uniqueImIds = np.unique( BB_im_ids[:k])
			total_cls_groundtruth = 0
			for uniqueImId in uniqueImIds:
				total_cls_groundtruth += imIdToNumGtInsideImDict[uniqueImId]
			# go down detections and mark TPs and FPs
			fp = 0
			tp = 0
			for j in range( k ):
				bb = BB[j, :].astype(float)
				bb = np.squeeze(bb)
				iou = computeIOU(BBGT,bb)
				print 'IOU: ', iou
		        if iou > iouthresh:
		        	tp += 1.
		        else:
		        	fp += 1.
			# compute precision recall
			prec.append( tp / max(total_cls_reports, np.finfo(np.float64).eps)) # how accurate are reports
			rec.append( tp / max(total_cls_groundtruth, np.finfo(np.float64).eps)) # how many ground truth can be found by the algorithm
	else:
		# no detections
		rec = [0]
		prec = [0]

	print 'Recall: ', rec
	print 'Precision: ', prec
	# avoid divide by zero
	ap = voc_ap(rec, prec )
	# My implementation vs. Ross Girshick's implementation -- should be same area under curve
	assert (voc_ap(rec, prec) - naiveAPCalculation(rec,prec)) < EPSILON
	return rec, prec, ap
