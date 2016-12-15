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
	# SORT THEM FIRST
	BB = [BLAH BLAH for BLAH BLAH in BLAH BLAH]# for our favorite volleyball o
	BBGT [BLAH BLAH for BLAH BLAH in BLAH BLAH]

	aps = []
	for i, cls in enumerate(CLASSES):
	    rec, prec, ap = matchGTsAndComputePrecRecallAP(BB,BBGT,ovthresh=0.5)
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


	if splitType == 'val':
		data_set_size = VAL_SET_SIZE
	elif splitType == 'test':
		data_set_size = TEST_SET_SIZE
	BB = np.zeros((TEST_SET_SIZE,4))
	BBGT = []

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
	ixmin = np.maximum(BBGT[:, 0], bb[0])
	iymin = np.maximum(BBGT[:, 1], bb[1])
	ixmax = np.minimum(BBGT[:, 2], bb[2])
	iymax = np.minimum(BBGT[:, 3], bb[3])
	iw = np.maximum(ixmax - ixmin , 0.)
	ih = np.maximum(iymax - iymin , 0.)
	inters = iw * ih
	return inters

def union(BBGT,bb,inters):
	"""
	Why do they inflate numbers by adding +1.? I don't
	https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py#L173
	"""
	union = ((bb[2] - bb[0] ) * (bb[3] - bb[1] ) + \
		(BBGT[:, 2] - BBGT[:, 0] ) * \
		(BBGT[:, 3] - BBGT[:, 1] ) - inters)
	return union


def computeIOU(BBGT,bb):
	inters = intersection(BBGT,bb)
	# union
	uni = union(BBGT,bb,inters)
	overlaps = inters / uni
	return overlaps


def unitTestAP():
	prec = np.array([1.,1.,.67,.75,.6,0.67,4/7., .5, 4/9., 0.5])
	rec = np.array([.2,.4,.4,.6,.6,.8,.8,.8,.8,1.])
	assert (voc_ap(rec, prec) - naiveAPCalculation(rec,prec)) < EPSILON


def matchGTsAndComputePrecRecallAP(BB,BBGT,ovthresh=0.5):
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

	# confidence = np.array([float(x[1]) for x in splitlines])
    # BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]


	# go down detections and mark TPs and FPs
	nd = len(image_ids)
	tp = np.zeros(nd) # True positives
	fp = np.zeros(nd) # False positives
	for d in range(nd):
		R = class_recs[image_ids[d]]
		bb = BB[d, :].astype(float)
		ovmax = -np.inf
		if BBGT.size > 0:
			overlaps = computeIOU(BBGT,bb)
			ovmax = np.max(overlaps)
			jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

	# compute precision recall
	fp = np.cumsum(fp)
	tp = np.cumsum(tp)
	rec = tp / float(npos)
	# avoid divide by zero in case the first detection matches a difficult
	# ground truth
	prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
	ap = voc_ap(rec, prec, use_07_metric)
	# My implementation vs. Ross Girshick's implementation.
	assert (voc_ap(rec, prec) - naiveAPCalculation(rec,prec)) < EPSILON
	return rec, prec, ap
	


BBGT = np.array([[0,0,4,4]])
bb = np.array([1,1,4,4])
print computeIOU(BBGT,bb)


