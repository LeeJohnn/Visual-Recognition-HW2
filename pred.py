# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from yolo.utils.box import visualize_boxes
from yolo.config import ConfigParser
from tqdm import trange

argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/svhn.json",
    help='config file')

argparser.add_argument(
    '-i',
    '--image',
    default="tests/samples/sample.jpeg",
    help='path to image file')

if __name__ == '__main__':
    args = argparser.parse_args()
    image_path   = args.image
    
    # 1. create yolo model & load weights
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)
    
    datas = []

    # 2. Load image
    for i in trange(13068):
        image = cv2.imread(image_path + str(i+1) + '.png')
        image = image[:,:,::-1]
        
        # 3. Run detection
        boxes, labels, probs = detector.detect(image, 0.5)
        
        # 4. draw detected boxes
        #visualize_boxes(image, boxes, labels, probs, config_parser.get_labels())

        # 5. plot    
        # plt.imshow(image)
        # plt.show()
        #plt.imsave("outImg.jpg", image)
        # labels = (labels + 1) % 10

        for x in range(len(labels)):
            labels[x] = (labels[x] + 1) % 10
        # if type(boxes) == 'list':
        #     print(boxes)
        # print(type(boxes))
        bb = list(boxes)

        output_boxes = []
        for x in range(len(boxes)):
            b = [bb[x][1], bb[x][0], bb[x][3], bb[x][2]]
            # tmp1 = bb[x][0]
            # bb[x][0] = bb[x][1]
            # bb[x][1] = tmp1
            # tmp2 = bb[x][2]
            # bb[x][2] = bb[x][3]
            # bb[x][3] = tmp2

            output_boxes.append(b)

        # probs = np.array(probs, np.float32)
        # probs = list(probs)
        # labels = np.array(labels, np.float32)
        # labels = list(labels)
        probs = [float(x) for x in probs]
        labels = [float(x) for x in labels]
        data = {'bbox': output_boxes, 'score': probs, 'label': labels}
        # print(boxes, probs, labels)
        #print(i)
        datas.append(data)
    # print(datas)
    
    with open('megumin.json', 'w') as fout:
      json.dump(datas, fout)

    # # save to json file
    # ret = json.dumps(datas, cls=NumpyEncoder)
    # with open('out.json', 'w') as fp:
    #     fp.write(ret)