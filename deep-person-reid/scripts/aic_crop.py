'''
Crop data for ReID training.
'''
import numpy as np
import cv2
import os
import collections
import random

splits = ['train','validation']
data_path = '/home/u1436961/aicity/data/'
out_path = '/home/u1436961/cycyang/deep-person-reid/reid-data/dukemtmc-reid/DukeMTMC-reID/'

for split in splits:
    scene_path = os.path.join(data_path,split)
    scenes = os.listdir(scene_path)
    for scene in scenes:
        
        cams = os.listdir(os.path.join(data_path,split,scene))
        cams = [cam for cam in cams if not cam.endswith('png')]
        for cam in cams:
            
            print('processing {} / {} / {}'.format(split,scene,cam))
            
            video = os.path.join(data_path,split,scene,cam,'video.mp4')
            
            annotations = os.path.join(data_path,split,scene,cam,'label.txt')
            annotations = np.genfromtxt(annotations, delimiter=',', dtype=None)
            anno_mem = collections.defaultdict(list)
            for frame,t_id,x,y,dx,dy,_,_,_,_ in annotations:
                anno_mem[t_id].append([frame,int(x),int(y),int(dx),int(dy)])
            
            crop_list_train = collections.defaultdict(list)
            crop_list_test = collections.defaultdict(list)
            crop_list_query = collections.defaultdict(list)
            
            for t_id in anno_mem:
                if len(anno_mem[t_id])>100:
                    pick_train = []
                    pick_test = []
                    pick_query = []
                    for _ in range(100):
                        idx = random.randint(0,len(anno_mem[t_id])-1)
                        pick_train.append(anno_mem[t_id][idx])
                    for _ in range(30):
                        idx = random.randint(0,len(anno_mem[t_id])-1)
                        pick_test.append(anno_mem[t_id][idx])
                    for _ in range(5):
                        idx = random.randint(0,len(anno_mem[t_id])-1)
                        pick_query.append(anno_mem[t_id][idx])
                
                for frame_id,x,y,dx,dy in pick_train:
                    crop_list_train[frame_id].append([t_id,x,y,dx,dy])
                for frame_id,x,y,dx,dy in pick_test:
                    crop_list_test[frame_id].append([t_id,x,y,dx,dy])
                for frame_id,x,y,dx,dy in pick_query:
                    crop_list_query[frame_id].append([t_id,x,y,dx,dy])
            
            cap = cv2.VideoCapture(video)
            
            frame_id = 0
            ret_val = True
            while ret_val:
                
                ret_val, frame = cap.read()
                
                if frame_id in crop_list_train:
                    for t_id,x,y,dx,dy in crop_list_train[frame_id]:
                        crop_img = frame[y:y+dy, x:x+dx]
                        
                        filename = '{}_{}_{}.jpg'.format(scene[1:]+str(t_id).zfill(3),cam,frame_id)
                        
                        filename = os.path.join(out_path,'bounding_box_train',filename)
                        
                        cv2.imwrite(filename,crop_img)
                        
                if frame_id in crop_list_test:
                    for t_id,x,y,dx,dy in crop_list_test[frame_id]:
                        crop_img = frame[y:y+dy, x:x+dx]
                        
                        filename = '{}_{}_{}.jpg'.format(scene[1:]+str(t_id).zfill(3),cam,frame_id)
                        
                        filename = os.path.join(out_path,'bounding_box_test',filename)
                        
                        cv2.imwrite(filename,crop_img)
                        
                if frame_id in crop_list_query:
                    for t_id,x,y,dx,dy in crop_list_query[frame_id]:
                        crop_img = frame[y:y+dy, x:x+dx]
                        
                        filename = '{}_{}_{}.jpg'.format(scene[1:]+str(t_id).zfill(3),cam,frame_id)
                        
                        filename = os.path.join(out_path,'query',filename)
                        
                        cv2.imwrite(filename,crop_img)
                        
                frame_id += 1