# John Lambert, Matt Vilim, Konstantine Buhler
# Dec 15, 2016

import os
import numpy as np



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
	aps = []
	for i, cls in enumerate(CLASSES):
	    rec, prec, ap = matchGTsAndComputePrecRecallAP(cls,detections,ovthresh=0.5)
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
	print BBGT[ 0], bb[0]
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
		print gtBbox
		inters = intersection(gtBbox,bb) * 1.0
		print 'Intersection: ', inters
		uni = union(gtBbox,bb,inters)
		print 'Union: ', uni
		iou =   inters/uni
		maxIOU = max(iou,maxIOU)
	return maxIOU


def unitTestAP():
	prec = np.array([1.,1.,.67,.75,.6,0.67,4/7., .5, 4/9., 0.5])
	rec = np.array([.2,.4,.4,.6,.6,.8,.8,.8,.8,1.])
	assert (voc_ap(rec, prec) - naiveAPCalculation(rec,prec)) < EPSILON


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
	predBB = []
	BB_im_ids = []

	imIdToNumGtInsideImDict = {}

	# FIND THE PREDICTIONS FOR JUST --ONE CLASS--, E.G. AIRPLANE
	for imIdx,im in enumerate(detections):
		# look at each of the predictions
		for j in range( len(im['bboxes'])):
			if im['pred_classes'][j] == cls:
				BB.append( im['bboxes'][j] )
				confidence.append( im['confidences'][j] )
				im_ids.append( imIdx )
		for j in range( len(im['gt_boxes_j0'])):
			num_of_this_class_in_this_im = 0
    		if im['gt_classes'][j] == cls:
    			BBGT.append( im['gt_boxes_j0'][j] )
    			num_of_this_class_in_this_im += 1
    		imIdToNumGtInsideImDict[imIdx] = num_of_this_class_in_this_im

	BBGT = np.asarray(BBGT)
	confidence = np.asarray(confidence)
	BB = np.asarray(BB)

	# sort by confidence
	sorted_ind = np.argsort(confidence) # sort according to highest confidence
	sorted_scores = np.sort(confidence) # sort according to highest confidence
	BB = BB[sorted_ind, :]
	BB_im_ids = BB_im_ids[sorted_ind,:]

	#get k highest rank (confidence score) things
	for k in range( BB.shape[0] ):
		# then choose next most confident thing
		#get unique imIds inside current rank list
		total_cls_reports = k
		uniqueImIds = np.unique( BB_im_ids[:k])
		for uniqueImId in uniqueImIds:
			total_cls_groundtruth += imIdToNumGtInsideImDict[uniqueImId]

		# go down detections and mark TPs and FPs
		fp = 0
		tp = 0
		if BBGT_classes == BB_classes:
			bb = BB[d, :].astype(float)
			ovmax = -np.inf
			if BBGT.size > 0:
				iou = computeIOU(BBGT,bb)
		        if iou > iouthresh:
		            if BBGT_classes[d] == BB_classes[d]:
		                tp += 1.
		            else:
		                fp += 1.
		        else:
		            fp += 1.

	# compute precision recall
	precision = tp / maximum(tp + fp, np.finfo(np.float64).eps) # how accurate are reports
	recall = tp / maximum(tp + fn, np.finfo(np.float64).eps) # how many ground truth can be found by the algorithm

	# avoid divide by zero
	ap = voc_ap(rec, prec )
	# My implementation vs. Ross Girshick's implementation -- should be same area under curve
	assert (voc_ap(rec, prec) - naiveAPCalculation(rec,prec)) < EPSILON
	return rec, prec, ap
