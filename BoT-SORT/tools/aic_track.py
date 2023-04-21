import argparse
import os
import sys
#import cv2
import numpy as np
import pickle
#import torch

sys.path.append('.')

from loguru import logger

from tracker.tracking_utils.timer import Timer
from tracker.bot_sort import BoTSORT

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Tracks For Evaluation!")

    parser.add_argument("root_path", default="/home/hsiangwei/Desktop/AICITY2023/data/", type=str)
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument('--nms_thres', type=float, default=0.7, help='nms threshold')

    # CMC
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

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
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def image_track(cam_detections, cam_embedding, sct_output_path, trk_output_path, args):

    # Tracker
    tracker = BoTSORT(args, frame_rate=args.fps)
    results = []
    
    num_frames = int(cam_detections[-1][0]) # 0~18009

    for frame_id in range(num_frames+1):

        inds = cam_detections[:,0] == frame_id
        detections = cam_detections[inds][:,2:] # x1,y1,x2,y2,score
        embedding = cam_embedding[inds]

        # NMS
        detections,pick = nms_fast(detections, None, args.nms_thres)
        embedding = embedding[pick]
        
        if detections is not None:

            trackerTimer.tic()
            online_targets = tracker.update(detections, embedding)
            trackerTimer.toc()

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                vertical = False
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()

        else:
            timer.toc()
        if frame_id % 1000 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    with open(sct_output_path, 'w') as f:
        f.writelines(results)
    logger.info(f"save SCT results to {sct_output_path}")
    
    with open(trk_output_path, 'wb') as f:
        pickle.dump(tracker.tracklets, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"save Trk results to {trk_output_path}")
    
def main():
    
    args = make_parser().parse_args()
    args.fps = 30
    data_path = args.root_path
    dataset = 'test'
    scene_path =  os.path.join(data_path,dataset)

    os.makedirs(os.path.join(data_path,'tracklet'), exist_ok=True)
    os.makedirs(os.path.join(data_path,'SCT'), exist_ok=True)
    
    demo_synthetic_only = False
    
    trk_low_thres = {'c001':0.1,
                     'c002':0.95,
                     'c003':0.1,
                     'c004':0.1,
                     'c005':0.95,
                     'c006':0.9,
                     'c007':0.8}
    
    if demo_synthetic_only:
        scenes = [scene for scene in os.listdir(scene_path) if scene != 'S001']
    else:
        scenes = os.listdir(scene_path)
    
    scenes.sort()
        
    for scene in scenes:

        cams = os.listdir(os.path.join(scene_path,scene))
        cams = [cam for cam in cams if not cam.endswith('png')]
        cams.sort()
        
        detections = np.genfromtxt(os.path.join(data_path,'{}_det'.format(dataset),scene+'.txt'), delimiter=',', dtype=str)
        
        if scene == 'S001':
            args.aspect_ratio_thresh = 10
            args.nms_thres = 1
            
            feature1 = np.load(os.path.join(data_path,'test_emb','S001_osnet_x1_0.npy'),allow_pickle=True)
            feature2 = np.load(os.path.join(data_path,'test_emb','S001_osnet_ibn_x1_0.npy'),allow_pickle=True)
            feature3 = np.load(os.path.join(data_path,'test_emb','S001_osnet_ain_x1_0.npy'),allow_pickle=True)
            feature4 = np.load(os.path.join(data_path,'test_emb','S001_osnet_x1_0_market.npy'),allow_pickle=True)
            feature5 = np.load(os.path.join(data_path,'test_emb','S001_osnet_x1_0_msmt17.npy'),allow_pickle=True)
            embedding = np.array([None]*len(feature1))
            
            for i in range(len(feature1)):
                embedding[i] = np.concatenate((feature1[i],feature2[i],feature3[i],feature4[i],feature5[i]),axis=0)
        
        else:
            args.nms_thres = 0.7
            embedding = np.load(os.path.join(data_path,'{}_emb'.format(dataset),scene+'.npy'), allow_pickle=True)
        
        for cam in cams:
            
            if scene == 'S001':
                args.track_low_thresh = trk_low_thres[cam]
            
            logger.info('Processing scene {} and camera {}'.format(scene,cam))
            
            # extract camera-based detection and embedding
            inds = detections[:,0] == cam
            
            # cam detections format: frame_id,cls,x1,y1,x2,y2,score
            cam_detections = detections[inds][:,1:].astype(float) 
            cam_embedding = embedding[inds]
            
            if cam == 'c005':
                inds = np.array([True]*len(cam_detections))
                for idx,(frame,cls,x1,y1,x2,y2,score) in enumerate(cam_detections):
                    if (x1+x2)/2<640 and (y1+y2)/2<360:
                        inds[idx] = False
                cam_detections = cam_detections[inds]
                cam_embedding = cam_embedding[inds]
                
            sct_output_path = os.path.join(data_path,'SCT','{}_{}.txt'.format(scene,cam))
            tracklet_output_path = os.path.join(data_path,'tracklet','{}_{}.pkl'.format(scene,cam))
            
            # SCT tracking
            image_track(cam_detections, cam_embedding, sct_output_path, tracklet_output_path, args)
            
if __name__ == "__main__":
    main()