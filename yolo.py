import argparse
import os
import csv
import cv2
import numpy as np
import tensorflow as tf

from plot_utils import *

class YOLO:
    def __init__(self, weight_path, checkpoint_path):
        self.debug = False
        self.weight_path = weight_path
        self.checkpoint_path = checkpoint_path
        self.num_classes = 20
        self.S = 7
        self.B = 2

        self.create_network()

    def create_network(self):
        # network structure is based on darknet yolo-small.cfg
        # https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/cfg/yolov1/yolo-small.cfg
        self.input_layer = tf.placeholder(tf.float32, shape = [None, 448, 448, 3])
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

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        # save checkpoint file if it doesn't exist
        if not os.path.exists(self.checkpoint_path):
            tf.train.Saver().save(self.session, self.checkpoint_path)
        else:
            tf.train.Saver().restore(self.session, self.checkpoint_path)

    def create_conv_layer(self, input_layer, d0, d1, filters, stride, weight_index):
        channels = int(input_layer.get_shape()[3])
        weight_shape = [d0, d1, channels, filters]
        bias_shape = [filters]

        weight = tf.random_normal(weight_shape, stddev = 0.35, dtype = tf.float32)
        bias = tf.random_normal(bias_shape, stddev = 0.35, dtype = tf.float32)
        if self.weight_path:
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
        if self.debug:
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
        if self.weight_path:
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
        if self.debug:
            input_layer = tf.Print(input_layer, [input_layer, weight, bias], 'connected')

        return self.activation(tf.add(tf.matmul(input_layer, weight), bias), leaky)

    def create_maxpool_layer(self, input_layer, d0, d1, stride):
        if self.debug:
            input_layer = tf.Print(input_layer, [input_layer], 'pool')
        return tf.nn.max_pool(input_layer, ksize = [1, d0, d1, 1], strides = [1, stride, stride, 1], padding = 'SAME')

    def create_dropout_layer(self, input_layer, prob):
        if self.debug:
            input_layer = tf.Print(input_layer, [input_layer], 'dropout')
        return tf.nn.dropout(input_layer, prob)

    def activation(self, input_layer, leaky = True):
        if leaky:
            # trick to create leaky activation function
            # phi(x) = x if x > 0, 0.1x otherwise
            return tf.maximum(input_layer, tf.scalar_mul(0.1, input_layer))
        else:
            return input_layer

    def process_image(self, img):
        img_resize = cv2.resize(img, (448, 448))
        # for some reason darknet switches red and blue channels...
        # https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/src/image.c#L391
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        # darknet scales color values from 0 to 1
        # https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/src/image.c#L469
        img_resize = (img_resize / 255.0) * 2.0 - 1.0

        input = np.expand_dims(img_resize, axis=0)
        predictions = self.session.run(self.output_layer, feed_dict = {self.input_layer: input, self.dropout_prob: 1})

        predictions = np.squeeze(predictions) # remove 1 from first dim, so not (1,1470)
        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        # A bit unclear what the 'l.n' is that they use here
        # https://github.com/pjreddie/darknet/blob/master/src/detection_layer.c#L236
        end_probs = self.num_classes*(self.S**2)
        end_confidences = end_probs + self.B*(self.S**2)
        probs = np.reshape( predictions[0:end_probs], [self.S,self.S,self.num_classes] )
        confidences = np.reshape( predictions[end_probs:end_confidences], [self.S,self.S,self.B])
        bboxes = np.reshape( predictions[end_confidences:], [self.S, self.S, self.B, 4] )

        img_out = img.copy()
        return plot_detections_on_im(img_out, probs, confidences, bboxes, classes)

    def process_video(self, video_path, start_frame, end_frame):
        frame_detections = []
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('data/output.avi', cv2.cv.CV_FOURCC(*'XVID'), 20.0, (w, h))

        i = 0
        frame_window = []
        while cap.isOpened() and i < end_frame:
            ret, frame = cap.read()

            if i >= start_frame:
                print "processing frame " + str(i)
                frame_out, bounding_boxes = self.process_image(frame)
                frame_window.append(bounding_boxes)
                # number of con
                window_size = 3
                if len(frame_window) > 3:
                    frame_window.pop(0)
                #if len(frame_window) == 3:
                #    self.group_cropped(frame_window, len(bounding_boxes))
                #frame_detections.append(bounding_boxes)
                #for box in bounding_boxes:
                #    print box.category
                #cv2.imwrite('data/' + str(i) + '.png', frame_out)
                out.write(frame_out)
            i = i + 1
        cap.release()
        out.release()

    def group_cropped(self, frame_window, group_count):
        # concatenate all objects across all frames within the frame window
        x = reduce(list.__add__, frame_window)
        size = frame_window[0][0].img.shape
        centroids = []
        for i in xrange(0, group_count):
            #centroids.append(np.random.randint(255, size=size))
            centroids.append(frame_window[0][i].img)
        c_old = np.zeros(len(x))
        while True:
            c = np.zeros(len(x))
            for i, x_i in enumerate(x):
                c_min = float('inf')
                for j, centroid in enumerate(centroids):
                    c_temp = np.square(np.linalg.norm(x_i.img - centroid, 2))
                    if c_temp < c_min:
                        c[i] = j
                        c_min = c_temp
            # update centroids
            for j, centroid in enumerate(centroids):
                numerator = np.zeros(size)
                denominator = 0.0
                for i, c_i in enumerate(c):
                    if (c_i == j):
                        numerator = numerator + x[i].img
                        denominator = denominator + 1.0
                centroids[j] = numerator / denominator
            # break after convergence
            if (c == c_old).all():
                break
            else:
                c_old = c
        print c_old

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_path', help='path to weights/biases for each layer')
    parser.add_argument('-t', '--image_path', help='path to test image')
    parser.add_argument('-c', '--checkpoint_path', help='path to create checkpoint', required=True)
    parser.add_argument('-v', '--video_path', help='path to test video')
    parser.add_argument('-s', '--start', help='video start frame')
    parser.add_argument('-e', '--end', help='video end frame')
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
    weight_path = None
    if args.weight_path:
        weight_path = os.path.abspath(os.path.expanduser(args.weight_path))
    yolo = YOLO(weight_path, checkpoint_path)
    if args.image_path:
        image_path = os.path.abspath(os.path.expanduser(args.image_path))
        img, bounding_boxes = yolo.process_image(cv2.imread(image_path))
        cv2.imwrite('data/out.png', img)
    if args.video_path:
        video_path = os.path.abspath(os.path.expanduser(args.video_path))
        yolo.process_video(video_path, int(args.start), int(args.end))

if __name__ == "__main__":
    main()
