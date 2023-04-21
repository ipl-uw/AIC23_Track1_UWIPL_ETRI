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
    parser = argparse.ArgumentParser("clustering for S001")
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
    distance_thers = [80]
    idx = 0
    detections = np.genfromtxt(os.path.join(root_path,'test_det','S001.txt'), delimiter=',', dtype=str)
    feature1 = np.load(os.path.join(root_path,'test_emb','S001_osnet_x1_0.npy'),allow_pickle=True)
    feature2 = np.load(os.path.join(root_path,'test_emb','S001_osnet_ibn_x1_0.npy'),allow_pickle=True)
    feature3 = np.load(os.path.join(root_path,'test_emb','S001_osnet_ain_x1_0.npy'),allow_pickle=True)
    feature4 = np.load(os.path.join(root_path,'test_emb','S001_osnet_x1_0_market.npy'),allow_pickle=True)
    feature5 = np.load(os.path.join(root_path,'test_emb','S001_osnet_x1_0_msmt17.npy'),allow_pickle=True)
    embeddings = np.array([None]*len(feature1))

    for i in range(len(feature1)):
        embeddings[i] = np.concatenate((feature1[i],feature2[i],feature3[i],feature4[i],feature5[i]),axis=0)

    '''
    nms
    '''
    embeddings = embeddings.tolist()
    embeddings = np.array(embeddings)

    all_dets = None
    all_embs = None

    for frame in threshold[idx][0]:

        inds = detections[:,1] == str(frame-1)
        frame_detections = detections[inds]
        frame_embeddings = embeddings[inds]

        cams = np.unique(detections[:,0])

        for cam in cams:
            inds = frame_detections[:,0]==cam
            cam_det = frame_detections[inds][:,1:].astype("float")
            cam_embedding = frame_embeddings[inds]
            
            if len(cam_det) == 0:continue
            
            inds = np.array([False]*len(cam_det))
            
            for i,(_,_,a1,b1,c1,d1,_) in enumerate(cam_det):
                if int((c1-a1)*(d1-b1))>10000:
                    inds[i] = True
            
            cam_det = cam_det[inds]
            cam_embedding = cam_embedding[inds]
            
            cam_det,pick = nms_fast(cam_det,None,nms_thres)
            cam_embedding = cam_embedding[pick]
            
            if len(cam_det) == 0:continue
            
            inds = cam_det[:,6]>threshold[idx][1]
            cam_det = cam_det[inds]
            cam_embedding = cam_embedding[inds]
            if all_dets is None:
                all_dets = cam_det
                all_embs = cam_embedding
            else:
                all_dets = np.vstack((all_dets,cam_det))
                all_embs = np.vstack((all_embs,cam_embedding))

    clustering = AgglomerativeClustering(distance_threshold=distance_thers[idx],n_clusters=None).fit(all_embs)
    
    return max(clustering.labels_)+1

def get_anchor(scene, dataset, threshold, nms_thres):
    
    '''
    
    input: scene
    output: dictionary (keys: anchor's global id, values: a list of embeddings for that anchor)
    
    '''
    
    if dataset == 'test':

        scenes = ['S001']

    else:
        raise ValueError('{} not supported dataset!'.format(dataset))
    

    if scene in scenes:
        seq_idx = scenes.index(scene)
    else:
        raise ValueError('scene not in {} set!'.format(dataset))

    scene = scenes[seq_idx]
    k = get_people(scene, dataset, threshold)

    detections = np.genfromtxt(os.path.join(root_path,'test_det','S001.txt'), delimiter=',', dtype=str)
    feature1 = np.load(os.path.join(root_path,'test_emb','S001_osnet_x1_0.npy'),allow_pickle=True)
    feature2 = np.load(os.path.join(root_path,'test_emb','S001_osnet_ibn_x1_0.npy'),allow_pickle=True)
    feature3 = np.load(os.path.join(root_path,'test_emb','S001_osnet_ain_x1_0.npy'),allow_pickle=True)
    feature4 = np.load(os.path.join(root_path,'test_emb','S001_osnet_x1_0_market.npy'),allow_pickle=True)
    feature5 = np.load(os.path.join(root_path,'test_emb','S001_osnet_x1_0_msmt17.npy'),allow_pickle=True)
    embeddings = np.array([None]*len(feature1))

    for i in range(len(feature1)):
        embeddings[i] = np.concatenate((feature1[i],feature2[i],feature3[i],feature4[i],feature5[i]),axis=0)

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
            
            if len(cam_det) == 0:continue
            
            inds = np.array([False]*len(cam_det))
            
            for i,(_,_,a1,b1,c1,d1,_) in enumerate(cam_det):
                if int((c1-a1)*(d1-b1))>10000:
                    inds[i] = True
            
            cam_det = cam_det[inds]
            cam_embedding = cam_embedding[inds]
            
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

    clustering = AgglomerativeClustering(n_clusters=k).fit(all_embs)

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
    
    n_list = [200,300,150,300,300,500,150]
    nms_thres = 1
    args = make_parser().parse_args()
    root_path = args.root_path

    os.makedirs(os.path.join(root_path,'hungarian_cluster_S001'),exist_ok=True)

    hungarian_thres = None
    
    dataset = 'test'
    
    scene_path =  os.path.join(root_path,dataset)
    
    scenes = ['S001']
    
    # test
    threshold = [([0,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000],0.95)]

    k = get_people('S001', dataset, threshold)
    
    logger.info('clustering {} set'.format(dataset))
    logger.info('scenes list: {}'.format(scenes))
    logger.info('n = {}'.format(n_list))
    logger.info('nms_thres = {}'.format(nms_thres))
    logger.info('anchor threshold = {}'.format(threshold))
    logger.info('hungarian threshold = {}'.format(hungarian_thres))
    logger.info('number of people = {}'.format(k))
    
    sct = np.genfromtxt(os.path.join(root_path,'SCT/S001_c001.txt'), delimiter=',', dtype=None)
    cur_frame = -1
    max_det = 0
    cur_det = 0
    ids = set()
    for frame,trk_id,x,y,w,h,_,_,_,_ in sct:
        if cur_det > k:
            extra_id_c001 = max(ids)
            extra_id_enter_frame = frame
            break
        if frame!=cur_frame:
            cur_det = 1
            cur_frame = frame
            ids = set()
            ids.add(trk_id)
        else:
            cur_det+=1
            ids.add(trk_id)

    logger.info('extra ID at c001 {} and enter frame {}'.format(extra_id_c001,extra_id_enter_frame))
            
    sct = np.genfromtxt(os.path.join(root_path,'SCT/S001_c002.txt'), delimiter=',', dtype=None)
    cur_frame = -1
    max_det = 0
    cur_det = 0
    ids = set()
    for frame,trk_id,x,y,w,h,_,_,_,_ in sct:

        if frame<extra_id_enter_frame:continue

        if x == 0: # enters from the bottom part of video
            extra_id_c002 = trk_id
            break

        if frame!=cur_frame:
            cur_det = 1
            cur_frame = frame
            ids = set()
            ids.add(trk_id)
        else:
            cur_det+=1
            ids.add(trk_id)
    
    logger.info('extra ID at c002 {} and enter frame {}'.format(extra_id_c002,frame))
    
    for scene in scenes:
        
        anchors = get_anchor(scene,dataset,threshold,nms_thres)
        logger.info('number of anchors {}'.format(len(anchors)))
        
        for anchor in anchors:
            logger.info('anchor {} : number of features {}'.format(anchor,len(anchors[anchor])))
        
        cams = os.listdir(os.path.join(scene_path,scene))
        cams = sorted([cam for cam in cams if not cam.endswith('png')])
        
        for c_idx,cam in enumerate(cams):
    
            n = n_list[c_idx]
    
            logger.info('processing scene {} cam {}'.format(scene,cam))
            
            with open(root_path+'/tracklet/{}_{}.pkl'.format(scene,cam),'rb') as f:
                tracklets = pickle.load(f)
    
            mapper = collections.defaultdict(list)
            global_id_mapper = collections.defaultdict(list)
            
            for i,trk_id in enumerate(tracklets):
                
                if trk_id == extra_id_c001 and cam == 'c001' or trk_id == extra_id_c002 and cam == 'c002':continue
                
                if i%100==0:
                    logger.info('Progress: {}/{}'.format(i,len(tracklets)))
                trk = tracklets[trk_id]
                for feat in trk.features:
                    box_dist = get_box_dist(feat,anchors)
                    mapper[trk_id].append(box_dist)
                mapper[trk_id].append(box_dist) # extra length to prevent bug -- len(features) < len(tracklets) 
            sct = np.genfromtxt(root_path+'/SCT/{}_{}.txt'.format(scene,cam), delimiter=',', dtype=None)

            counter = collections.defaultdict(int)
            
            new_sct = os.path.join(root_path,'hungarian_cluster_S001/{}_{}.txt'.format(scene,cam))

            results = []
                
            cur_frame = -1
            cost_matrix = None

            for frame_id,trk_id,x,y,w,h,score,_,_,_ in sct:

                if trk_id == extra_id_c001 and cam == 'c001' or trk_id == extra_id_c002 and cam == 'c002':continue
                    
                if len(tracklets[trk_id].features) <10 : # too short trajectories
                    continue
                
                if frame_id != cur_frame and cost_matrix is None: # first frame
                    cost_matrix = []
                    frame_trk_ids = []
                    cur_frame = frame_id
                
                elif frame_id != cur_frame and cost_matrix is not None: # next frame => conduct hungarian, clear cost matrix
                    cost_matrix = np.array(cost_matrix)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for row,col in zip(row_ind,col_ind):
                        if hungarian_thres:
                            if cost_matrix[row][col]<hungarian_thres:
                                global_id_mapper[frame_trk_ids[row]].append(col)
                            else:
                                global_id_mapper[frame_trk_ids[row]].append(-1)
                        else:
                            global_id_mapper[frame_trk_ids[row]].append(col)
                    cost_matrix = []
                    frame_trk_ids = []
                    cur_frame = frame_id
                cost_matrix.append(mapper[trk_id][counter[trk_id]])   # row is cost for this feature to all anchors
                frame_trk_ids.append(trk_id)

                counter[trk_id] += 1

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

                if trk_id == extra_id_c001 and cam == 'c001' or trk_id == extra_id_c002 and cam == 'c002':
                    results.append(
                        f"{frame_id},{8},{x},{y},{w},{h},{score},-1,-1,-1\n"
                    )
                    continue
                
                if len(new_global_id_mapper[trk_id]) == 0:
                    continue
                    
                # initiate stable tracklet check
                if len(new_global_id_mapper[trk_id])>2*n and new_global_id_mapper[trk_id][0] == new_global_id_mapper[trk_id][-1]:
                    global_id = new_global_id_mapper[trk_id][0]+1
                    
                else:
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
                
                if global_id!=0: # too large distance, global ID become -1+1=0
                    results.append(
                            f"{frame_id},{global_id},{x},{y},{w},{h},{score},-1,-1,-1\n"
                        )
                
                counter2[trk_id] += 1
            
            logger.info('conflict:{}'.format(conflict))
            
            with open(new_sct, 'w') as f:
                f.writelines(results)