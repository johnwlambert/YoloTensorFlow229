import argparse
import os
from xml.dom import minidom

# download dataset here: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

class bounding_box:
    def __init__(self, x_min, y_min, w, h, category):
        self.x_min = x_min
        self.y_min = y_min
        self.w = w
        self.h = h
        self.category = category

class annotated_image:
    def __init__(self, image_path):
        self.image_path = image_path
        # list of class bounding boxes
        self.bounding_boxes = []

# pass ../VOCdevkit/ path here
# function will return a list of annotated_image objects
def preprocess_data(voc_data_path):
    # list of all annotated_images in dataset
    annotated_images = []
    annotations_dir = os.path.join(voc_data_path, 'VOC2007', 'Annotations')
    images_dir = os.path.join(voc_data_path, 'VOC2007', 'JPEGImages')

    for filename in os.listdir(annotations_dir):
        image_number = os.path.splitext(os.path.basename(filename))[0]
        image_path = os.path.join(voc_data_path, 'VOC2007', 'JPEGImages', image_number + '.jpg')
        image = annotated_image(image_path)

        xml = minidom.parse(os.path.join(annotations_dir, filename))
        xml_objects = xml.getElementsByTagName('object')
        for xml_object in xml_objects:
            category =  str(xml_object.getElementsByTagName('name')[0].firstChild.nodeValue)
            xml_bounding_box = xml_object.getElementsByTagName('bndbox')[0]
            x_min = int(xml_bounding_box.getElementsByTagName('xmin')[0].firstChild.nodeValue)
            y_min = int(xml_bounding_box.getElementsByTagName('ymin')[0].firstChild.nodeValue)
            x_max = int(xml_bounding_box.getElementsByTagName('xmax')[0].firstChild.nodeValue)
            y_max = int(xml_bounding_box.getElementsByTagName('ymax')[0].firstChild.nodeValue)
            w  = x_max - x_min
            h = y_max - y_min
            box = bounding_box(x_min, y_min, w, h, category)
            image.bounding_boxes.append(box)

        annotated_images.append(image)
    return annotated_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', help='path to VOC data')
    args = parser.parse_args()
    preprocess_data(args.data_path)
