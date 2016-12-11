# John Lambert, Matt Vilim, Konstantine Buhler
# CS 229 Course Project
# "You Only Look Once" YOLO Detection
# November 2016

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.patches as patches
import cv2

def plot_detections_on_im( im , probs_given_obj , prob_obj , bboxes, classes, thresh = 0.2 ):
    S = 7
    B = 2
    num_classes = 20
    probs = np.zeros((S,S,B,num_classes))
    # We use a law of probability: prob(class) = prob(class|object) * prob(object)
    for i in range(B):
        for j in range(num_classes):
            probs[:,:,i,j] = np.multiply(probs_given_obj[:,:,j],prob_obj[:,:,i])
    for row in range(S):
        for col in range(S):
            for boxIdx in range(B):
                score = np.amax( probs[row,col,boxIdx,:] )
                class_name = classes[np.argmax(probs[row,col,boxIdx,:])]
                if score > thresh:
                    imHeight = im.shape[0]
                    imWidth = im.shape[1]
                    x = (bboxes[row,col,boxIdx,0] + col) / 7.0 * imWidth
                    y = (bboxes[row,col,boxIdx,1] + row) / 7.0 * imHeight
                    w = (bboxes[row,col,boxIdx,2]**2) * imWidth * 1.0
                    h = (bboxes[row,col,boxIdx,3]**2) * imHeight * 1.0
                    left = max(0, x - w/2. )
                    right = min( x + w/2., imWidth-1 )
                    top = max(0, y - h/2. )
                    bot = min( y + h/2., imHeight-1 )
                    cv2.rectangle(im, (int(left), int(top)), (int(right), int(bot)), (0, 255, 255), 2)
                    cv2.putText(im, class_name, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2);
    cv2.imwrite('dog_out.png', im)

def plotSplitMetric( complete_train_loss_history, complete_val_loss_history, new_dir_path, metricName, iterNum, complete_test_loss_history=None ):
    """
    # lengthOfTrainLossHistory = len( complete_train_loss_history )
    # lengthOfValLossHistory = len( complete_val_loss_history )
    # lengthOfTestLossHistory = len( complete_test_loss_history )
    """
    train_loss_x = []
    train_loss_y = []

    val_loss_x = []
    val_loss_y = []

    if complete_test_loss_history is not None:
        test_loss_x = []
        test_loss_y = []

    for idx in xrange( len( complete_train_loss_history ) ):
        train_loss_x.append( complete_train_loss_history[idx][0] )
        train_loss_y.append( complete_train_loss_history[idx][1] )

    for idx in xrange( len( complete_val_loss_history ) ):
        val_loss_x.append( complete_val_loss_history[idx][0] )
        val_loss_y.append( complete_val_loss_history[idx][1] )

    if complete_test_loss_history is not None:
        for idx in xrange( len( complete_test_loss_history ) ):
            test_loss_x.append( complete_test_loss_history[idx][0] )
            test_loss_y.append( complete_test_loss_history[idx][1] )

    plt.subplot(1, 1, 1)

    plt.title('%s vs. Number of Iterations\n RED=TRAINING, BLUE=VALIDATION, GREEN=TEST' % (metricName ))
    line1, = plt.plot( train_loss_x, train_loss_y, '-ro', label = 'Training %s' % (metricName) )
    line2, = plt.plot( val_loss_x, val_loss_y, '-bo', label = 'Val %s' % (metricName) )
    if complete_test_loss_history is not None:
        line3, = plt.plot( test_loss_x, test_loss_y, '-go', label = 'Test %s' % (metricName) )

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

    plt.xlabel('Iteration')
    plt.ylabel( metricName )
    if metricName == 'Accuracy':
        plt.ylim([0.0, 1.0])
    elif metricName == 'Loss':
        plt.ylim([0,7])
    plt.gcf().set_size_inches(15, 12)
    lossPlotFileName = './%s/YOLO%sHistory_%d.png' % (new_dir_path, metricName, iterNum )
    plt.savefig( lossPlotFileName )
    plt.clf()
