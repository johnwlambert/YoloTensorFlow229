import argparse
import os
import csv
import cv2
import numpy as np
import tensorflow as tf

from plot_utils import *

class YOLO:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.num_classes = 20
        self.S = 7
        self.B = 2

    def create_network(self, weight_path):
        # network structure is based on darknet yolo-small.cfg
        # https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/cfg/yolov1/yolo-small.cfg
        self.input_layer = tf.placeholder(tf.float32, shape = [None, 448, 448, 3])
        conv_layer0 = self.create_conv_layer(self.input_layer, weight_path, 7, 7, 64, 2, 0)
        maxpool_layer1 = self.create_maxpool_layer(conv_layer0, 2, 2, 2)
        conv_layer2 = self.create_conv_layer(maxpool_layer1, weight_path, 3, 3, 192, 1, 2)
        maxpool_layer3 = self.create_maxpool_layer(conv_layer2, 2, 2, 2)
        conv_layer4 = self.create_conv_layer(maxpool_layer3, weight_path, 1, 1, 128, 1, 4)
        conv_layer5 = self.create_conv_layer(conv_layer4, weight_path, 3, 3, 256, 1, 5)
        conv_layer6 = self.create_conv_layer(conv_layer5, weight_path, 1, 1, 256, 1, 6)
        conv_layer7 = self.create_conv_layer(conv_layer6, weight_path, 3, 3, 512, 1, 7)
        maxpool_layer8 = self.create_maxpool_layer(conv_layer7, 2, 2, 2)
        conv_layer9 = self.create_conv_layer(maxpool_layer8, weight_path, 1, 1, 256, 1, 9)
        conv_layer10 = self.create_conv_layer(conv_layer9, weight_path, 3, 3, 512, 1, 10)
        conv_layer11 = self.create_conv_layer(conv_layer10, weight_path, 1, 1, 256, 1, 11)
        conv_layer12 = self.create_conv_layer(conv_layer11, weight_path, 3, 3, 512, 1, 12)
        conv_layer13 = self.create_conv_layer(conv_layer12, weight_path, 1, 1, 256, 1, 13)
        conv_layer14 = self.create_conv_layer(conv_layer13, weight_path, 3, 3, 512, 1, 14)
        conv_layer15 = self.create_conv_layer(conv_layer14, weight_path, 1, 1, 256, 1, 15)
        conv_layer16 = self.create_conv_layer(conv_layer15, weight_path, 3, 3, 512, 1, 16)
        conv_layer17 = self.create_conv_layer(conv_layer16, weight_path, 1, 1, 512, 1, 17)
        conv_layer18 = self.create_conv_layer(conv_layer17, weight_path, 3, 3, 1024, 1, 18)
        maxpool_layer19 = self.create_maxpool_layer(conv_layer18, 2, 2, 2)
        conv_layer20 = self.create_conv_layer(maxpool_layer19, weight_path, 1, 1, 512, 1, 20)
        conv_layer21 = self.create_conv_layer(conv_layer20, weight_path, 3, 3, 1024, 1, 21)
        conv_layer22 = self.create_conv_layer(conv_layer21, weight_path, 1, 1, 512, 1, 22)
        conv_layer23 = self.create_conv_layer(conv_layer22, weight_path, 3, 3, 1024, 1, 23)
        conv_layer24 = self.create_conv_layer(conv_layer23, weight_path, 3, 3, 1024, 1, 24)
        conv_layer25 = self.create_conv_layer(conv_layer24, weight_path, 3, 3, 1024, 2, 25)
        conv_layer26 = self.create_conv_layer(conv_layer25, weight_path, 3, 3, 1024, 1, 26)
        conv_layer27 = self.create_conv_layer(conv_layer26, weight_path, 3, 3, 1024, 1, 27)
        # flatten layer for connection to fully connected layer
        conv_layer27_flatten_dim = int(reduce(lambda a, b: a * b, conv_layer27.get_shape()[1:]))
        conv_layer27_flatten = tf.reshape(tf.transpose(conv_layer27, (0, 3, 1, 2)), [-1, conv_layer27_flatten_dim])
        connected_layer28 = self.create_connected_layer(conv_layer27_flatten, weight_path, 512, True, 28)
        connected_layer29 = self.create_connected_layer(connected_layer28, weight_path, 4096, True, 29)
        # dropout layer is only used during training
        self.dropout_prob = tf.placeholder(tf.float32)
        dropout_layer30 = self.create_dropout_layer(connected_layer29, self.dropout_prob)
        connected_layer31 = self.create_connected_layer(dropout_layer30, weight_path, 1470, False, 31)
        self.output_layer = connected_layer31

    def create_conv_layer(self, input_layer, weight_path, d0, d1, filters, stride, weight_index):
        channels = int(input_layer.get_shape()[3])
        weight_shape = [d0, d1, channels, filters]
        bias_shape = [filters]

        weight = tf.random_normal(weight_shape, stddev = 0.35, dtype = tf.float32)
        bias = tf.random_normal(bias_shape, stddev = 0.35, dtype = tf.float32)
        if weight_path:
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

    def create_connected_layer(self, input_layer, weight_path, d0, leaky, weight_index):
        weight_shape = [int(input_layer.get_shape()[1]), d0]
        bias_shape = [d0]

        weight = tf.random_normal(weight_shape, stddev = 0.35, dtype = tf.float32)
        bias = tf.random_normal(bias_shape, stddev = 0.35, dtype = tf.float32)
        if weight_path:
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

    def run_inference(self, image_path):
        self.create_network(None)

        session = tf.Session()
        #writer = tf.train.SummaryWriter( './data' , session.graph )
        session.run(tf.initialize_all_variables())
        tf.train.Saver().restore(session, self.checkpoint_path)

        img_resize, img = self.process_image(image_path)
        # add batch dimension
        input = np.expand_dims(img_resize, axis=0)
        prediction = session.run(self.output_layer, feed_dict = {self.input_layer: input, self.dropout_prob: 1})
        self.process_prediction(img, prediction)

    def save_checkpoint(self, weight_path):
        self.create_network(weight_path)
        session = tf.Session()
        session.run(tf.initialize_all_variables())
        tf.train.Saver().save(session, self.checkpoint_path)

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        img_resize = cv2.resize(img, (448, 448))
        # for some reason darknet switches red and blue channels...
        # https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/src/image.c#L391
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        # darknet scales color values from 0 to 1
        # https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/src/image.c#L469
        img_resize = (img_resize / 255.0) * 2.0 - 1.0
        return img_resize, img

    def process_video(self, video_path):
        vid = cv2.VideoCapture(video_path)

    def process_prediction(self, img, predictions):
        predictions = np.squeeze( predictions ) # remove 1 from first dim, so not (1,1470)
        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        # A bit unclear what the 'l.n' is that they use here
        # https://github.com/pjreddie/darknet/blob/master/src/detection_layer.c#L236
        end_probs = self.num_classes*(self.S**2)
        end_confidences = end_probs + self.B*(self.S**2)
        probs = np.reshape( predictions[0:end_probs], [self.S,self.S,self.num_classes] )
        confidences = np.reshape( predictions[end_probs:end_confidences], [self.S,self.S,self.B])
        bboxes = np.reshape( predictions[end_confidences:], [self.S, self.S, self.B, 4] )

        img_out = img.copy()
        img_out = plot_detections_on_im(img_out, probs, confidences, bboxes, classes)
        cv2.imwrite('data/out.png', img_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_path', help='path to weights/biases for each layer')
    parser.add_argument('-t', '--image_path', help='path to test image')
    parser.add_argument('-v', '--video_path', help='path to test video')
    parser.add_argument('-c', '--checkpoint_path', help='path to create checkpoint', required=True)
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
    yolo = YOLO(checkpoint_path)
    if args.weight_path:
        weight_path = os.path.abspath(os.path.expanduser(args.weight_path))
        yolo.save_checkpoint(weight_path)
    if args.image_path:
        image_path = os.path.abspath(os.path.expanduser(args.image_path))
        yolo.run_inference(image_path)
    if args.video_path:
        video_path = os.path.abspath(os.path.expanduser(args.video_path))
        print video_path

if __name__ == "__main__":
    main()
