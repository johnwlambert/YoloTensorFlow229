# John Lambert, Matt Vilim, Konstantine Buhler
# Dec 14, 2016

import tensorflow as tf
import numpy as np
from functools import reduce

########################################################
NUM_CLASSES = 20
NUM_GRID = 7
########################################################
START_IDX_PROBS = 0
END_IDX_PROBS = (NUM_CLASSES) * (NUM_GRID**2)
START_IDX_CONFIDENCES = END_IDX_PROBS
END_IDX_CONFIDENCES = START_IDX_CONFIDENCES + (NUM_GRID**2)*2
START_IDX_BBOXES = END_IDX_CONFIDENCES
END_IDX_BBOXES = START_IDX_BBOXES + (NUM_GRID**2)*2*4
########################################################

#from YoloTensorFlowFunctions import *
from MatrixCompute_YoloLossTF import *

class YOLO_TrainingNetwork:
    def __init__(self, use_pretrained_weights):
        self.num_classes = 20
        self.S = 7
        self.B = 2
        self.use_open_cv = False
        self.add_placeholders()
        self.lr = 1e-4
        self.pretrained_weights = use_pretrained_weights
        self.create_network()

    def add_placeholders(self):
        """
        There are 3 input nodes to the computational graph: 
            images, ground truth, and dropout prob
        """
        self.input_layer = tf.placeholder(tf.float32, shape = [None, 448, 448, 3], name='Inputs')
        # GTs have shape 73=NUM_CLASSES+4+49
        self.gts = tf.placeholder(tf.float32, shape = [None, NUM_CLASSES + 4 + (7*7) ], name='GTs')
        # dropout prob is only set <1 during training
        self.dropout_prob = tf.placeholder(tf.float32)

    def create_network(self):
        # network structure is based on darknet yolo-small.cfg
        # https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/cfg/yolov1/yolo-small.cfg
        conv_layer0 = self.create_conv_layer(self.input_layer, 7, 7, 64, 2, 0, 'ConvLayer0')
        maxpool_layer1 = self.create_maxpool_layer(conv_layer0, 2, 2, 2 )
        conv_layer2 = self.create_conv_layer(maxpool_layer1, 3, 3, 192, 1, 2, 'ConvLayer2')
        maxpool_layer3 = self.create_maxpool_layer(conv_layer2, 2, 2, 2 )
        conv_layer4 = self.create_conv_layer(maxpool_layer3, 1, 1, 128, 1, 4, 'ConvLayer4')
        conv_layer5 = self.create_conv_layer(conv_layer4, 3, 3, 256, 1, 5, 'ConvLayer5')
        conv_layer6 = self.create_conv_layer(conv_layer5, 1, 1, 256, 1, 6, 'ConvLayer6')
        conv_layer7 = self.create_conv_layer(conv_layer6, 3, 3, 512, 1, 7, 'ConvLayer7')
        maxpool_layer8 = self.create_maxpool_layer(conv_layer7, 2, 2, 2)
        conv_layer9 = self.create_conv_layer(maxpool_layer8, 1, 1, 256, 1, 9, 'ConvLayer9')
        conv_layer10 = self.create_conv_layer(conv_layer9, 3, 3, 512, 1, 10, 'ConvLayer10')
        conv_layer11 = self.create_conv_layer(conv_layer10, 1, 1, 256, 1, 11, 'ConvLayer11')
        conv_layer12 = self.create_conv_layer(conv_layer11, 3, 3, 512, 1, 12, 'ConvLayer12')
        conv_layer13 = self.create_conv_layer(conv_layer12, 1, 1, 256, 1, 13, 'ConvLayer13')
        conv_layer14 = self.create_conv_layer(conv_layer13, 3, 3, 512, 1, 14, 'ConvLayer14')
        conv_layer15 = self.create_conv_layer(conv_layer14, 1, 1, 256, 1, 15, 'ConvLayer15')
        conv_layer16 = self.create_conv_layer(conv_layer15, 3, 3, 512, 1, 16, 'ConvLayer16')
        conv_layer17 = self.create_conv_layer(conv_layer16, 1, 1, 512, 1, 17, 'ConvLayer17')
        conv_layer18 = self.create_conv_layer(conv_layer17, 3, 3, 1024, 1, 18, 'ConvLayer18')
        maxpool_layer19 = self.create_maxpool_layer(conv_layer18, 2, 2, 2)
        conv_layer20 = self.create_conv_layer(maxpool_layer19, 1, 1, 512, 1, 20, 'ConvLayer20')
        conv_layer21 = self.create_conv_layer(conv_layer20, 3, 3, 1024, 1, 21, 'ConvLayer21')
        conv_layer22 = self.create_conv_layer(conv_layer21, 1, 1, 512, 1, 22, 'ConvLayer22')
        conv_layer23 = self.create_conv_layer(conv_layer22, 3, 3, 1024, 1, 23, 'ConvLayer23')
        conv_layer24 = self.create_conv_layer(conv_layer23, 3, 3, 1024, 1, 24, 'ConvLayer24')
        conv_layer25 = self.create_conv_layer(conv_layer24, 3, 3, 1024, 2, 25, 'ConvLayer25')
        conv_layer26 = self.create_conv_layer(conv_layer25, 3, 3, 1024, 1, 26, 'ConvLayer26')
        conv_layer27 = self.create_conv_layer(conv_layer26, 3, 3, 1024, 1, 27, 'ConvLayer27')
        # flatten layer for connection to fully connected layer
        conv_layer27_flatten_dim = int(reduce(lambda a, b: a * b, conv_layer27.get_shape()[1:]))
        conv_layer27_flatten = tf.reshape(tf.transpose(conv_layer27, (0, 3, 1, 2)), [-1, conv_layer27_flatten_dim])
        connected_layer28 = self.create_connected_layer(conv_layer27_flatten, 512, True, 28, 'ConnectedLayer28')
        connected_layer29 = self.create_connected_layer(connected_layer28, 4096, True, 29, 'ConnectedLayer29')

        dropout_layer30 = self.create_dropout_layer(connected_layer29, self.dropout_prob)
        connected_layer31 = self.create_connected_layer(dropout_layer30, 1470, False, 31, 'ConnectedLayer31')
        self.output_layer = connected_layer31

        self.class_probs = self.output_layer[:, START_IDX_PROBS: END_IDX_PROBS ]
        self.confidences = self.output_layer[:, START_IDX_CONFIDENCES: END_IDX_CONFIDENCES ]
        self.bboxes = self.output_layer[:, START_IDX_BBOXES : END_IDX_BBOXES]

        print 'Class Probs: ', self.class_probs.get_shape().as_list()
        print 'Confidences: ', self.confidences.get_shape().as_list()
        print 'Bboxes: ', self.bboxes.get_shape().as_list()

        self.class_probs = tf.reshape( self.class_probs,shape=[1,NUM_GRID,NUM_GRID,NUM_CLASSES])
        self.confidences = tf.reshape( self.confidences,shape=[1,NUM_GRID,NUM_GRID,2])
        self.bboxes = tf.reshape(self.bboxes, shape=[1,NUM_GRID,NUM_GRID,2*4])
        #bboxes = np.reshape( predictions[end_confidences:], [self.S, self.S, self.B, 4] )
        #pred_boxes_arr = pred_labels[:, :, :, NUM_CLASSES : NUM_CLASSES + NUM_BOX * 4]

        self.loss = computeYoloLossTF( self.class_probs, self.confidences, self.bboxes, self.gts)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # tf.contrib.metrics.streaming_sparse_average_precision_at_k(predictions, labels, k, weights=None, metrics_collections=None, updates_collections=None, name=None)

        # self.final_class_probs = tf.cross( self.class_probs, self.confidences )
        # # can do tf.where, or tf.greater to see if greater than thresh
        # # otherwise make it zero

    def create_conv_layer(self, input_layer, d0, d1, filters, stride, weight_index,name):
        channels = int(input_layer.get_shape()[3])
        weight_shape = [d0, d1, channels, filters]
        bias_shape = [filters]
        with tf.variable_scope(name + '_conv_weights'):
            weight = tf.get_variable( 'w_%s' % (name), weight_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.get_variable( 'b_%s' % (name), bias_shape, initializer=tf.constant_initializer(0.0))

        #weight = tf.random_normal(weight_shape, stddev = 0.35, dtype = tf.float32)
        #bias = tf.random_normal(bias_shape, stddev = 0.35, dtype = tf.float32)
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

        # weight = tf.Variable(weight)
        # bias = tf.Variable(bias)
        #input_layer = tf.Print(input_layer, [input_layer, weight, bias], "convolution")

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

    def create_connected_layer(self, input_layer, d0, leaky, weight_index, name):
        """
        INPUTS:
        -   input_layer: Tensor
        -   d0: 
        -   leaky
        -   weight_index: 
        -   name: string
        OUTPUTS:
        -   
        """
        weight_shape = [int(input_layer.get_shape()[1]), d0]
        bias_shape = [d0]

        with tf.variable_scope(name+'_fully_connected_weights'):
            weight = tf.get_variable('w_%s' % (name) , weight_shape, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('b_%s' % (name) , bias_shape, initializer=tf.constant_initializer(0.0))
        #weight = tf.random_normal(weight_shape, stddev = 0.35, dtype = tf.float32)
        #bias = tf.random_normal(bias_shape, stddev = 0.35, dtype = tf.float32)
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

        # weight = tf.Variable(weight)
        # bias = tf.Variable(bias)
        #input_layer = tf.Print(input_layer, [input_layer, weight, bias], 'connected')

        return self.activation(tf.add(tf.matmul(input_layer, weight), bias), leaky)

    def create_maxpool_layer(self, input_layer, d0, d1, stride):
        #input_layer = tf.Print(input_layer, [input_layer], 'pool')
        return tf.nn.max_pool(input_layer, ksize = [1, d0, d1, 1], strides = [1, stride, stride, 1], padding = 'SAME')

    def create_dropout_layer(self, input_layer, prob):
        """
        INPUTS:
        -   input_layer: incoming Tensor
        -   prob: float, drop out neurons uniformly at random with this probability
        OUTPUTS:
        -   output_layer: output Tensor after neurons are dropped out with prob = p
        """
        #input_layer = tf.Print(input_layer, [input_layer], 'dropout')
        return tf.nn.dropout(input_layer, prob)

    def activation(self, input_layer, leaky = True):
        """
        INPUTS:
        -   input_layer: incoming Tensor 
        -   leaky (optional): specifies that we use Leaky ReLU instead of ReLU
        OUTPUTS:
        -   input_layer: output of activation function
        """
        if leaky:
            # trick to create leaky activation function
            # phi(x) = x if x > 0, 0.1x otherwise
            return tf.maximum(input_layer, tf.scalar_mul(0.1, input_layer))
        else:
            return input_layer

    