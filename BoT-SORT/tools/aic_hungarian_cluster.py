'''
2023 aicity challenge

Using auto anchor generation.

input:single camera tracking results, tracklets from BoT-SORT and embedding
output:produce MCMT result at hungarian

'''
import pickle
import os
import collections
import argparse
import numpy as np
from loguru import logger
from collections import Counter
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append('.')

def make_parser():
    parser = argparse.ArgumentParser("clustering for synthetic data")
    parser.add_argument("root_path", default="/home/hsiangwei/Desktop/AICITY2023/data", type=str)
    return parser

def nms_fast(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    x2 = boxes[:, 4]
    y2 = boxes[:, 5]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked
    return boxes[pick].astype("float"), pick

def get_people(scene, dataset, threshold):
    scenes = ['S003','S009','S014','S018','S021','S022']
    assert scene in scenes
    distance_thers = [13,19.5,16,13,16,16]
    seq_idx = scenes.index(scene)
    detections = np.genfromtxt(root_path+'/test_det/{}.txt'.format(scene), delimiter=',', dtype=str)
    embeddings = np.load(root_path+'/test_emb/{}.npy'.format(scene),allow_pickle = True)

    '''
    nms
    '''
    embeddings = embeddings.tolist()
    embeddings = np.array(embeddings)

    all_dets = None
    all_embs = None

    for frame in threshold[seq_idx][0]:

        inds = detections[:,1] == str(frame-1)
        frame_detections = detections[inds]
        frame_embeddings = embeddings[inds]

        cams = np.unique(detections[:,0])

        for cam in cams:
            inds = frame_detections[:,0]==cam
            cam_det = frame_detections[inds][:,1:].astype("float")
            cam_embedding = frame_embeddings[inds]
            cam_det,pick = nms_fast(cam_det,None,nms_thres)
            cam_embedding = cam_embedding[pick]
            if len(cam_det) == 0:continue
            inds = cam_det[:,6]>threshold[seq_idx][1]
            cam_det = cam_det[inds]
            cam_embedding = cam_embedding[inds]
            if all_dets is None:
                all_dets = cam_det
                all_embs = cam_embedding
            else:
                all_dets = np.vstack((all_dets,cam_det))
                all_embs = np.vstack((all_embs,cam_embedding))

    clustering = AgglomerativeClustering(distance_threshold=distance_thers[seq_idx],n_clusters=None).fit(all_embs)
    
    return max(clustering.labels_)+1

def get_anchor(scene, dataset, threshold, nms_thres):
    
    '''
    
    input: scene
    output: dictionary (keys: anchor's global id, values: a list of embeddings for that anchor)
    
    '''
    
    if dataset == 'test':

        scenes = ['S003','S009','S014','S018','S021','S022']
        
    else:
        raise ValueError('{} not supported dataset!'.format(dataset))
    

    if scene in scenes:
        seq_idx = scenes.index(scene)
    else:
        raise ValueError('scene not in {} set!'.format(dataset))

    scene = scenes[seq_idx]
    k = get_people(scene,dataset,threshold)

    detections = np.genfromtxt(root_path+'/test_det/{}.txt'.format(scene), delimiter=',', dtype=str)
    embeddings = np.load(root_path+'/test_emb/{}.npy'.format(scene),allow_pickle = True)

    '''
    nms
    '''
    embeddings = embeddings.tolist()
    embeddings = np.array(embeddings)

    all_dets = None
    all_embs = None

    for frame in threshold[seq_idx][0]:

        inds = detections[:,1] == str(frame-1)
        frame_detections = detections[inds]
        frame_embeddings = embeddings[inds]

        cams = np.unique(detections[:,0])

        for cam in cams:
            inds = frame_detections[:,0]==cam
            cam_det = frame_detections[inds][:,1:].astype("float")
            cam_embedding = frame_embeddings[inds]
            cam_det,pick = nms_fast(cam_det,None,nms_thres)
            cam_embedding = cam_embedding[pick]
            if len(cam_det) == 0:continue
            inds = cam_det[:,6]>threshold[seq_idx][1]
            cam_det = cam_det[inds]
            cam_embedding = cam_embedding[inds]
            if all_dets is None:
                all_dets = cam_det
                all_embs = cam_embedding
            else:
                all_dets = np.vstack((all_dets,cam_det))
                all_embs = np.vstack((all_embs,cam_embedding))

            #cam_record += [cam]*len(cam_det)

    clustering = AgglomerativeClustering(n_clusters=k).fit(all_embs)

    # print(clustering.labels_)

    anchors = collections.defaultdict(list)

    for global_id in range(k):
        for n in range(len(all_embs)):
            if global_id == clustering.labels_[n]:
                anchors[global_id].append(all_embs[n])
    
    return anchors

def get_box_dist(feat,anchors):
    '''
    input : feature, anchors                                                anc1  anc2  anc3 ...
    output : a list with distance between feature and anchor, e.g.    feat [dist1,dist2,dist3,...]
    '''
    box_dist = []

    for idx in anchors:
        dists = []
        for anchor in anchors[idx]:
            anchor /= np.linalg.norm(anchor)
            dists += [distance.cosine(feat,anchor)]
        
        #dist = min(dists) # or average..?
        dist = sum(dists)/len(dists)
        
        box_dist.append(dist)    
            
    return box_dist
        

if __name__ == "__main__":
    
    args = make_parser().parse_args()
    root_path = args.root_path

    os.makedirs(os.path.join(root_path,'hungarian_cluster'),exist_ok=True)

    n = 15
    nms_thres = 1
    #nms_thres = 0.7
    #nms_thres = 0.3
    
    #dataset = 'validation'
    dataset = 'test'
    
    scene_path =  os.path.join(root_path,dataset)
    
    demo_synthetic_only = True
    
    if demo_synthetic_only:
        if dataset == 'test':
            scenes = ['S003','S009','S014','S018','S021','S022']
        else:
            scenes = ['S005','S008','S013','S017']

    # test
    threshold = [([1,5000,10000,15000],0.9), #OK
             ([1,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000],0.92), #OK
             ([1,2500,5000,7500,10000,15000],0.96), #OK
             ([1,5000,10000,15000],0.9), #OK
             ([1,2500,5000,7500,10000,12500,15000],0.95), #OK
             ([1,2500,5000,7500,10000],0.9)] #OK
    
    logger.info('clustering {} set'.format(dataset))
    logger.info('scenes list: {}'.format(scenes))
    logger.info('n = {}'.format(n))
    logger.info('nms_thres = {}'.format(nms_thres))
    logger.info('demo_synthetic_only = {}'.format(demo_synthetic_only))
    logger.info('threshold = {}'.format(threshold))
    
    for scene in scenes:
        
        anchors = get_anchor(scene,dataset,threshold,nms_thres)
        people = get_people(scene,dataset,threshold)
        logger.info('number of anchors {}'.format(len(anchors)))
        logger.info('number of people {}'.format(people))
        
        for anchor in anchors:
            logger.info('anchor {} : number of features {}'.format(anchor,len(anchors[anchor])))
        
        cams = os.listdir(os.path.join(scene_path,scene))
        cams = sorted([cam for cam in cams if not cam.endswith('png')])
        
        for cam in cams:
       
            logger.info('processing scene {} cam {}'.format(scene,cam))
            
            with open(root_path+'/tracklet/{}_{}.pkl'.format(scene,cam),'rb') as f:
                tracklets = pickle.load(f)
    
            mapper = collections.defaultdict(list)
            global_id_mapper = collections.defaultdict(list)
            
            for i,trk_id in enumerate(tracklets):
                if i%100==0:
                    logger.info('Progress: {}/{}'.format(i,len(tracklets)))
                trk = tracklets[trk_id]
                for feat in trk.features:
                    box_dist = get_box_dist(feat,anchors)
                    mapper[trk_id].append(box_dist)
                mapper[trk_id].append(box_dist) # extra length to prevent bug -- len(features) < len(tracklets) 
            sct = np.genfromtxt(root_path+'/SCT/{}_{}.txt'.format(scene,cam), delimiter=',', dtype=None)

            counter = collections.defaultdict(int)
            
            new_sct = root_path+'/hungarian_cluster/{}_{}.txt'.format(scene,cam)

            results = []
                
            cur_frame = -1
            cost_matrix = None

            for frame_id,trk_id,x,y,w,h,score,_,_,_ in sct:

                if len(tracklets[trk_id].features) == 0: # too short trajectories, not sure if this can caused bug!
                    continue
                
                if frame_id != cur_frame and cost_matrix is None: # first frame
                    cost_matrix = []
                    frame_trk_ids = []
                    cur_frame = frame_id
                
                elif frame_id != cur_frame and cost_matrix is not None: # next frame => conduct hungarian, clear cost matrix
                    cost_matrix = np.array(cost_matrix)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for row,col in zip(row_ind,col_ind):
                        global_id_mapper[frame_trk_ids[row]].append(col)
                    cost_matrix = []
                    frame_trk_ids = []
                    cur_frame = frame_id
                cost_matrix.append(mapper[trk_id][counter[trk_id]])   # row is cost for this feature to all anchors
                frame_trk_ids.append(trk_id)

                counter[trk_id] += 1

            # for trk_id in global_id_mapper:
            #     print(trk_id,Counter(global_id_mapper[trk_id]).most_common())

            new_global_id_mapper = collections.defaultdict(list)

            for trk_id in global_id_mapper:
                ids = global_id_mapper[trk_id] # id list
                new_ids = []
                cur_ids = []
                for id in ids:
                    cur_ids.append(id)
                    if len(cur_ids) == n:
                        new_ids += [Counter(cur_ids).most_common(1)[0][0]]*n
                        cur_ids = []
                        
                if len(cur_ids)>0:
                    if len(new_ids)>0:
                        new_ids += [new_ids[-1]]*(len(cur_ids)+1)
                    else:
                        new_ids += [Counter(cur_ids).most_common(1)[0][0]]*(len(cur_ids)+1)
                new_global_id_mapper[trk_id] = new_ids
                new_global_id_mapper[trk_id] += [new_ids[-1]]*n # some extra length to prevent bug
                

            # Write Result

            conflict = 0
            counter2 = collections.defaultdict(int)

            for frame_id,trk_id,x,y,w,h,score,_,_,_ in sct:
                
                if len(new_global_id_mapper[trk_id]) == 0:
                    continue
                
                if counter2[trk_id]>=len(new_global_id_mapper[trk_id]):
                    global_id = new_global_id_mapper[trk_id][-1]+1
                else:
                    global_id = new_global_id_mapper[trk_id][counter2[trk_id]]+1

                if frame_id != cur_frame:
                    global_ids = set()
                    cur_frame = frame_id
                if global_id in global_ids:
                    conflict+=1
                else:
                    global_ids.add(global_id)
                
                results.append(
                        f"{frame_id},{global_id},{x},{y},{w},{h},{score},-1,-1,-1\n"
                    )
                
                counter2[trk_id] += 1
            
            logger.info('conflict:{}'.format(conflict))
            
            with open(new_sct, 'w') as f:
                f.writelines(results)