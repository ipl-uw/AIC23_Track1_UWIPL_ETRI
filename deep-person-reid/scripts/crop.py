import numpy as np
from numpy import genfromtxt
import cv2
import glob 
import copy
from PIL import Image
import os
import collections

path_in = glob.glob('/work/u1436961/hsiangwei/dataset/train/*')
viz_list = []
for path in path_in:
    viz_list.append(os.path.basename(path))

print(path_in)
print(viz_list)


id_idx = 1

for sid, seq in enumerate(viz_list,1):
    print('processing seq {}/{}'.format(sid, len(viz_list)))
    path_in = '/work/u1436961/hsiangwei/dataset/train/' + seq + '/gt/gt.txt'
    labels = genfromtxt(path_in, delimiter=',', dtype=None)
    imgs = sorted(glob.glob('/work/u1436961/hsiangwei/dataset/train/' + seq + '/img1/*'))
    i = 0
    im_idx = -1
    im = None
    for lid, label in enumerate(labels):
        if lid % 20 == 13:
            if im_idx != label[0] - 1:
                im = cv2.imread(imgs[label[0] - 1])
            im_out = copy.deepcopy(im)
            x1 = label[2]
            y1 = label[3]
            x2 = x1+label[4]
            y2 = y1+label[5]
            crop = im_out[y1:y2,x1:x2]

            id_idx = sid*20 + label[1]
            fname = '{}_{}_{}'.format(id_idx, seq.replace('_', ''), label[0])
            #cv2.imwrite('/home/u1436961/cycyang/deep-person-reid/reid-data/dukemtmc-reid/bounding_box_train/{}.jpg'.format(fname), crop)
            if id_idx % 20 == 8:
                if label[0] % 100 > 10:
                    cv2.imwrite('/home/u1436961/cycyang/deep-person-reid/reid-data/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/{}.jpg'.format(fname), crop)
                else:
                    cv2.imwrite('/home/u1436961/cycyang/deep-person-reid/reid-data/dukemtmc-reid/DukeMTMC-reID/query/{}.jpg'.format(fname), crop)


