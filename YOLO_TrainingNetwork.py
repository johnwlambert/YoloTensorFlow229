# John Lambert

import tensorflow as tf
import numpy as np
from functools import reduce

from YoloTensorFlowFunctions import *


class YOLO_TrainingNetwork:
    def __init__(self):
        self.num_classes = 20
        self.S = 7
        self.B = 2
        self.use_open_cv = False
        self.add_placeholders()

    def add_placeholders(self):
        self.input_layer = tf.placeholder(tf.float32, shape = [None, 448, 448, 3])
        self.gts = tf.placeholder(tf.float32, shape = [None, 7, 7, 30])

    def create_network(self):
        # network structure is based on darknet yolo-small.cfg
        # https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/cfg/yolov1/yolo-small.cfg
        conv_layer0 = self.create_conv_layer(self.input_layer, 7, 7, 64, 2, 0)
        maxpool_layer1 = self.create_maxpool_layer(conv_layer0, 2, 2, 2)
        conv_layer2 = self.create_conv_layer(maxpool_layer1, 3, 3, 192, 1, 2)
        maxpool_layer3 = self.create_maxpool_layer(conv_layer2, 2, 2, 2)
        conv_layer4 = self.create_conv_layer(maxpool_layer3, 1, 1, 128, 1, 4)
        conv_layer5 = self.create_conv_layer(conv_layer4, 3, 3, 256, 1, 5)
        conv_layer6 = self.create_conv_layer(conv_layer5, 1, 1, 256, 1, 6)
        conv_layer7 = self.create_conv_layer(conv_layer6, 3, 3, 512, 1, 7)
        maxpool_layer8 = self.create_maxpool_layer(conv_layer7, 2, 2, 2)
        conv_layer9 = self.create_conv_layer(maxpool_layer8, 1, 1, 256, 1, 9)
        conv_layer10 = self.create_conv_layer(conv_layer9, 3, 3, 512, 1, 10)
        conv_layer11 = self.create_conv_layer(conv_layer10, 1, 1, 256, 1, 11)
        conv_layer12 = self.create_conv_layer(conv_layer11, 3, 3, 512, 1, 12)
        conv_layer13 = self.create_conv_layer(conv_layer12, 1, 1, 256, 1, 13)
        conv_layer14 = self.create_conv_layer(conv_layer13, 3, 3, 512, 1, 14)
        conv_layer15 = self.create_conv_layer(conv_layer14, 1, 1, 256, 1, 15)
        conv_layer16 = self.create_conv_layer(conv_layer15, 3, 3, 512, 1, 16)
        conv_layer17 = self.create_conv_layer(conv_layer16, 1, 1, 512, 1, 17)
        conv_layer18 = self.create_conv_layer(conv_layer17, 3, 3, 1024, 1, 18)
        maxpool_layer19 = self.create_maxpool_layer(conv_layer18, 2, 2, 2)
        conv_layer20 = self.create_conv_layer(maxpool_layer19, 1, 1, 512, 1, 20)
        conv_layer21 = self.create_conv_layer(conv_layer20, 3, 3, 1024, 1, 21)
        conv_layer22 = self.create_conv_layer(conv_layer21, 1, 1, 512, 1, 22)
        conv_layer23 = self.create_conv_layer(conv_layer22, 3, 3, 1024, 1, 23)
        conv_layer24 = self.create_conv_layer(conv_layer23, 3, 3, 1024, 1, 24)
        conv_layer25 = self.create_conv_layer(conv_layer24, 3, 3, 1024, 2, 25)
        conv_layer26 = self.create_conv_layer(conv_layer25, 3, 3, 1024, 1, 26)
        conv_layer27 = self.create_conv_layer(conv_layer26, 3, 3, 1024, 1, 27)
        # flatten layer for connection to fully connected layer
        conv_layer27_flatten_dim = int(reduce(lambda a, b: a * b, conv_layer27.get_shape()[1:]))
        conv_layer27_flatten = tf.reshape(tf.transpose(conv_layer27, (0, 3, 1, 2)), [-1, conv_layer27_flatten_dim])
        connected_layer28 = self.create_connected_layer(conv_layer27_flatten, 512, True, 28)
        connected_layer29 = self.create_connected_layer(connected_layer28, 4096, True, 29)
        # dropout layer is only used during training
        self.dropout_prob = tf.placeholder(tf.float32)
        dropout_layer30 = self.create_dropout_layer(connected_layer29, self.dropout_prob)
        connected_layer31 = self.create_connected_layer(dropout_layer30, 1470, False, 31)
        self.output_layer = connected_layer31
        self.output_layer = tf.reshape(self.output_layer,shape[7,7,30])
        self.loss = self.computeYoloLossTF( self.output_layer, self.gts)

        ## KEEP THIS STRUCTURE FOR THE PRETRAINED NET
        # self.class_probs = tf.slice(self.output_layer, [0,],[49*20,])
        # self.confidences = tf.slice(self.output_layer, [49*20,],[49*2,])
        # self.bboxes = tf.slice(self.output_layer,[49*20 + 49*2,],[49*2*4])
        # self.class_probs = tf.reshape( self.class_probs, shape=[7,7,20])
        # self.confidences = tf.reshape( self.confidences, shape=[7,7,2])
        # self.bboxes = tf.reshape(self.bboxes,shape=[7,7,2,4])
        # self.final_class_probs = tf.cross( self.class_probs, self.confidences )
        # # can do tf.where, or tf.greater to see if greater than thresh
        # # otherwise make it zero

    def create_conv_layer(self, input_layer, d0, d1, filters, stride, weight_index):
        channels = int(input_layer.get_shape()[3])
        weight_shape = [d0, d1, channels, filters]
        bias_shape = [filters]

        weight = tf.random_normal(weight_shape, stddev = 0.35, dtype = tf.float32)
        bias = tf.random_normal(bias_shape, stddev = 0.35, dtype = tf.float32)
        if self.pretrained_weights:
            weight = np.empty(weight_shape, dtype = np.float32)
            weight_trained_path = os.path.join(self.weight_path, 'conv_weight_layer' + str(weight_index + 1) + '.csv')
            print 'Loading weights from file: ' + weight_trained_path
            weight_trained = np.genfromtxt(weight_trained_path, delimiter = ',', dtype = np.float32)
            for i in range(weight_shape[0]):
                for j in range(weight_shape[1]):
                    for k in range(weight_shape[2]):
                        for l in range(weight_shape[3]):
                            weight[i, j, k, l] = weight_trained[(l * weight_shape[0] * weight_shape[1] * weight_shape[2]) + (k * weight_shape[0] * weight_shape[1]) + (i * weight_shape[0]) + j]

            bias = np.empty(bias_shape, dtype = 'float32')
            bias_trained_path = os.path.join(self.weight_path, 'conv_bias_layer' + str(weight_index + 1) + '.csv')
            print 'Loading biases from file: ' + bias_trained_path
            bias_trained = np.genfromtxt(bias_trained_path, delimiter = ',', dtype = np.float32)
            for i in range(bias_shape[0]):
                bias[i] = bias_trained[i]

        weight = tf.Variable(weight)
        bias = tf.Variable(bias)
        input_layer = tf.Print(input_layer, [input_layer, weight, bias], "convolution")

        # mimic explicit padding used by darknet...a bit tricky
        # https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/src/parser.c#L145
        # note that padding integer in yolo-small.cfg actually refers to a boolean value (NOT an acutal padding size)
        d0_pad = int(d0/2)
        d1_pad = int(d1/2)
        input_layer_padded = tf.pad(input_layer, paddings = [[0, 0], [d0_pad, d0_pad], [d1_pad, d1_pad], [0, 0]])
        # we need VALID padding here to match the sizing calculation for output of convolutional used by darknet
        convolution = tf.nn.conv2d(input = input_layer_padded, filter = weight, strides = [1, stride, stride, 1], padding='VALID')
        convolution_bias = tf.add(convolution, bias)
        return self.activation(convolution_bias)

    def create_connected_layer(self, input_layer, d0, leaky, weight_index):
        weight_shape = [int(input_layer.get_shape()[1]), d0]
        bias_shape = [d0]

        weight = tf.random_normal(weight_shape, stddev = 0.35, dtype = tf.float32)
        bias = tf.random_normal(bias_shape, stddev = 0.35, dtype = tf.float32)
        if self.pretrained_weights:
            weight = np.empty(weight_shape, dtype = np.float32)
            weight_trained_path = os.path.join(self.weight_path, 'connect_weight_layer' + str(weight_index + 1) + '.csv')
            print 'Loading weights from file: ' + weight_trained_path
            weight_trained = np.genfromtxt(weight_trained_path, delimiter = ',', dtype = np.float32)
            for i in range(weight_shape[0]):
                for j in range(weight_shape[1]):
                    weight[i, j] = weight_trained[j * weight_shape[0] + i]

            bias = np.empty(bias_shape, dtype = 'float32')
            bias_trained_path = os.path.join(self.weight_path, 'connect_bias_layer' + str(weight_index + 1) + '.csv')
            print 'Loading biases from file: ' + bias_trained_path
            bias_trained = np.genfromtxt(bias_trained_path, delimiter = ',', dtype = np.float32)
            for i in range(bias_shape[0]):
                bias[i] = bias_trained[i]

        weight = tf.Variable(weight)
        bias = tf.Variable(bias)
        input_layer = tf.Print(input_layer, [input_layer, weight, bias], 'connected')

        return self.activation(tf.add(tf.matmul(input_layer, weight), bias), leaky)

    def create_maxpool_layer(self, input_layer, d0, d1, stride):
        input_layer = tf.Print(input_layer, [input_layer], 'pool')
        return tf.nn.max_pool(input_layer, ksize = [1, d0, d1, 1], strides = [1, stride, stride, 1], padding = 'SAME')

    def create_dropout_layer(self, input_layer, prob):
        input_layer = tf.Print(input_layer, [input_layer], 'dropout')
        return tf.nn.dropout(input_layer, prob)

    def activation(self, input_layer, leaky = True):
        if leaky:
            # trick to create leaky activation function
            # phi(x) = x if x > 0, 0.1x otherwise
            return tf.maximum(input_layer, tf.scalar_mul(0.1, input_layer))
        else:
            return input_layer



    def computeYoloLossTF(self, pred, gt ):
        """
        Need to find a way to compute this as a batch and not individually
        """

        return konstantinesCodeButConverted(pred, gt)

        # USE assignSlice( tensor, index, sliceValue) 

    