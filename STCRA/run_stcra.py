import argparse
import collections
import glob
import os
import warnings

import numpy as np

import cam_utils
from file_utils import load_tracking, write_tracking

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# ==================== UTILS FUNCTIONS ====================
def get_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
        box1 (list): A list [x, y, w, h] representing the coordinates of the first box.
        box2 (list): A list [x, y, w, h] representing the coordinates of the second box.

    Returns:
        float: The IoU of the two boxes.
    """
    box1, box2 = box1[:4], box2[:4]
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    # Calculate the area of intersection rectangle
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of the two boxes
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # Calculate the area of union
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area

    return iou


def smooth_avg_tracks(avg_tracks, window_size=60):
    # smooth the location of each track based on a sliding window over fid
    # the structure of smooth_avg_tracks = {tid: {fid: [xworld, yworld]}}
    smoothed_avg_tracks = collections.defaultdict(dict)
    for tid, track in avg_tracks.items():
        for fid, location in track.items():
            # get the location of the track in the sliding window
            window = []
            for i in range(-window_size // 2, window_size // 2 + 1):
                if fid + i in track.keys():
                    window.append(track[fid + i])
            # smooth the location of the track
            if len(window) > 0:
                smoothed_avg_tracks[tid][fid] = np.mean(window, axis=0)
    return smoothed_avg_tracks


def id_reassignment(tracks_list, dis_thr=200, conf_thr=0.7, try_second_nearest=True, verbose=False):
    # 1. first merge the location from each camera based on same tid and same frame into a dict
    # if two of the same tid are in the same frame, first reassign them to tid closest to the average location
    # the structure of merge_tracks = {tid: {fid: [(cam_name, [xworld, yworld]), ...]}}
    merge_tracks = collections.defaultdict(dict)
    reassign_duplicate_ids = []
    error_iou_8 = 0
    for cam_name, tracks in tracks_list.items():
        for tid, track in tracks.items():
            for fid, location in track.items():
                # if multiple location for same tid in current fid
                if len(location) > 1:
                    if get_iou(location[0], location[1]) >= 0.75:
                        # when iou is larger than 0.8, we only keep the one with higher confidence score
                        if location[0][6] >= location[1][6]:
                            tracks_list[cam_name][tid][fid] = [location[0]]
                        else:
                            tracks_list[cam_name][tid][fid] = [location[1]]
                    else:
                        reassign_duplicate_ids.append((cam_name, tid, fid, location[0]))
                        reassign_duplicate_ids.append((cam_name, tid, fid, location[1]))
                        tracks_list[cam_name][tid][fid] = []
                elif len(location):
                    # x, y, w, h = location[0][:4]
                    x2d, y2d = location[0][4], location[0][5]
                    xworld, yworld = cam_utils.image2world([x2d, y2d], cam_utils.HOMO["S001"][cam_name])
                    if fid in merge_tracks[tid].keys():
                        merge_tracks[tid][fid].append((cam_name, [xworld, yworld], location[0]))
                    else:
                        merge_tracks[tid][fid] = [(cam_name, [xworld, yworld], location[0])]
    if len(reassign_duplicate_ids):
        print(f'The number of duplicate ids to reassign is {len(reassign_duplicate_ids)}')

    # 2. find the outlier for each tid and fid and calculate the average wc location
    # the structure of avg_tracks = {tid: {fid: [xworld, yworld]}}
    cam_weight = { # the number are roughly assigned by the total ground-plane each camera view is covering
        "c001": 4.0, "c002": 1.5, "c003": 1.5, "c004": 1.0,
        "c005": 1.0, "c006": 1.5, "c007": 2.0
    }
    avg_tracks = collections.defaultdict(dict)
    reassign_outliers = []
    case1_cnt = 0
    case2_cnt = 0
    for tid, track in merge_tracks.items():
        for fid, loc_list in track.items():
            # remove the location that have a distance larger than 300 with all other wc location with same id
            if len(loc_list) > 2:
                cam = [_loc[0] for _loc in loc_list]
                coord = np.asarray([_loc[1] for _loc in loc_list])
                loc = np.asarray([_loc[2] for _loc in loc_list])
                dist = [np.linalg.norm(coord - coord[i], axis=1) for i in range(len(coord))]
                delete_index = []
                for i in range(len(coord)):
                    # case 1: find the location that have a distance larger than $dis_thr with all other location 
                    # excluding themselves
                    if np.all(dist[i][0:i] > dis_thr) and np.all(dist[i][i + 1:] > dis_thr):
                        # print("delete", tid, t, i, cam[i])
                        delete_index.append(i)
                        reassign_outliers.append((cam[i], tid, fid, loc[i]))
                        case1_cnt += 1
                    # elif np.sum(dist[i] > dis_thr) >= len(coord) - 2 and len(coord) >= 5:
                    #     delete_index.append(i)
                    #     reassign_outliers.append((cam[i], tid, fid, loc[i]))
                    #     case2_cnt += 1
                # this condition is to prevent the case that all the location are outlier
                if len(delete_index) != len(coord):
                    # remove the outlier before calculating the average wc location
                    loc_list = np.delete(loc_list, delete_index, axis=0)
            # check corner cases
            if np.sum(np.array([cam_weight[_loc[0]] for _loc in loc_list])) < 1:
                print(np.sum(np.array([cam_weight[_loc[0]] for _loc in loc_list])))
                print(cam_name, tid, fid, loc_list)
            # weighted average coordinate base on the camera weighting
            avg_loc = np.sum(np.array([np.multiply(_loc[1], cam_weight[_loc[0]]) for _loc in loc_list]),
                             axis=0) / np.sum(np.array([cam_weight[_loc[0]] for _loc in loc_list]))
            avg_tracks[tid][fid] = avg_loc
    print(
        f'The number of outlier ids to reassign is {len(reassign_outliers)} from {case1_cnt} case1 + {case2_cnt} case2')

    # 3. reassign the duplicate to the nearest track
    cnt_all = 0
    cnt_same = 0
    corrected_list = []
    for cam_name, tid, fid, location in reassign_duplicate_ids:
        # x, y, w, h, = loc
        x2d, y2d = location[4], location[5]
        xworld, yworld = cam_utils.image2world([x2d, y2d], cam_utils.HOMO["S001"][cam_name])
        # data structure of avg_tracks -> tid: {t: [xworld, yworld]}
        # find the closest location in avg_tracks at each t
        # get the list for every tid at the current t
        dist_list = []
        id_list = []
        # not captured by any other camera except the current camera
        if fid not in avg_tracks[tid].keys():
            tracks_list[cam_name][tid][fid] = [location]
            cnt_all += 1
            cnt_same += 1
        else:
            for c_tid, track in avg_tracks.items():
                if fid in track.keys():
                    dist = np.linalg.norm(np.array([xworld, yworld]) - np.array(track[fid]))
                    dist_list.append(dist)
                    id_list.append(c_tid)
            min_index = np.argmin(dist_list)
            corrected_id_name = id_list[min_index]
            try:
                confidence = 1 - dist_list[min_index] / dist_list[id_list.index(tid)]
            except ValueError:
                print(cam_name, tid, fid, location, dist_list, id_list, min_index)
            # check if the corrected id is already assigned to some detection under same camera
            if fid not in tracks_list[cam_name][corrected_id_name].keys() or not \
                    tracks_list[cam_name][corrected_id_name][fid]:
                corrected_list.append((tid, fid, cam_name, corrected_id_name, confidence, dist_list[min_index],
                                       dist_list[id_list.index(tid)]))
                if confidence >= conf_thr or corrected_id_name == tid:
                    tracks_list[cam_name][corrected_id_name][fid] = [location]
                    if corrected_id_name == tid:
                        cnt_same += 1
                    cnt_all += 1
            # try second-nearest id
            elif try_second_nearest and corrected_id_name != tid:
                del dist_list[min_index]
                del id_list[min_index]
                min_index = np.argmin(dist_list)
                corrected_id_name = id_list[min_index]
                confidence = 1 - dist_list[min_index] / dist_list[id_list.index(tid)]

                if fid not in tracks_list[cam_name][corrected_id_name].keys() or not \
                        tracks_list[cam_name][corrected_id_name][fid]:
                    corrected_list.append((tid, fid, cam_name, corrected_id_name, confidence, dist_list[min_index],
                                           dist_list[id_list.index(tid)]))
                    if confidence >= conf_thr:
                        tracks_list[cam_name][corrected_id_name][fid] = [location]
                        if corrected_id_name == tid:
                            cnt_same += 1
                        cnt_all += 1
    print(f'The number of duplicate ids to reassign is {cnt_all}, {cnt_same} of them are the same as the original id')

    # 4. reassign the outlier to the nearest track
    for cam_name, tid, fid, location in reassign_outliers:
        # x, y, w, h, = loc
        x2d, y2d = location[4], location[5]
        xworld, yworld = cam_utils.image2world([x2d, y2d], cam_utils.HOMO["S001"][cam_name])
        # data structure of avg_tracks -> tid: {t: [xworld, yworld]}
        # find the closest location in avg_tracks at each t
        # get the list for every tid at the current t
        dist_list = []
        id_list = []
        for c_tid, track in avg_tracks.items():
            if fid in track.keys():
                dist = np.linalg.norm(np.array([xworld, yworld]) - np.array(track[fid]))
                dist_list.append(dist)
                id_list.append(c_tid)
        min_index = np.argmin(dist_list)
        corrected_id_name = id_list[min_index]
        confidence = 1 - dist_list[min_index] / dist_list[id_list.index(tid)]
        # check if the corrected id is already assigned to some detection under same camera
        if fid not in tracks_list[cam_name][corrected_id_name].keys() or not tracks_list[cam_name][corrected_id_name][
            fid]:
            corrected_list.append((tid, fid, cam_name, corrected_id_name, confidence, dist_list[min_index],
                                   dist_list[id_list.index(tid)]))
            if confidence >= conf_thr:
                tracks_list[cam_name][corrected_id_name][fid] = [location]
                del tracks_list[cam_name][tid][fid]
        # try second-nearest id
        elif try_second_nearest and corrected_id_name != tid:
            del dist_list[min_index]
            del id_list[min_index]
            min_index = np.argmin(dist_list)
            corrected_id_name = id_list[min_index]
            confidence = 1 - dist_list[min_index] / dist_list[id_list.index(tid)]

            if fid not in tracks_list[cam_name][corrected_id_name].keys() or not \
                    tracks_list[cam_name][corrected_id_name][fid]:
                corrected_list.append((tid, fid, cam_name, corrected_id_name, confidence, dist_list[min_index],
                                       dist_list[id_list.index(tid)]))
                if confidence >= conf_thr:
                    tracks_list[cam_name][corrected_id_name][fid] = [location]
                    del tracks_list[cam_name][tid][fid]

    # 5. sort corrected_list based on cam_name, fid, tid, corrected_id_name
    corrected_list = sorted(corrected_list, key=lambda _x: (_x[2], _x[0], _x[3], _x[1]))
    # correct the tid in tracks_list
    if verbose:
        for tid, fid, cam_name, corrected_id_name, conf, new_dist, old_dist in corrected_list:
            print("{} frame={} | {}->{} | conf: {} ({}->{})".format(cam_name, fid, tid,
                                                                    corrected_id_name,
                                                                    conf,
                                                                    old_dist,
                                                                new_dist))

    return tracks_list, corrected_list, avg_tracks



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder_name', type=str, default='tracking_file/0324_without_interpolation_feet')
    parser.add_argument('--output_folder_name', type=str, default='tracking_file/0324_without_interpolation_feet_corrected')
    return parser.parse_args()


def main():

    args = parse_args()

    tracks_file_list = sorted(glob.glob(os.path.join(args.input_folder_name, '*.txt')))

    # tracks_file_list = [
    #     "tracking_file/0324_without_interpolation_feet/S001_c001.txt",
    #     "tracking_file/0324_without_interpolation_feet/S001_c002.txt",
    #     "tracking_file/0324_without_interpolation_feet/S001_c003.txt",
    #     "tracking_file/0324_without_interpolation_feet/S001_c004.txt",
    #     "tracking_file/0324_without_interpolation_feet/S001_c005.txt",
    #     "tracking_file/0324_without_interpolation_feet/S001_c006.txt",
    #     "tracking_file/0324_without_interpolation_feet/S001_c007.txt"
    # ]

    tracks_list = {
        _track_file.split('_')[-1][:-4]: load_tracking(_track_file, enable_feet=True) for _track_file in
        tracks_file_list
    }

    new_tracks_list, corrected_list, avg_tracks = id_reassignment(tracks_list, dis_thr=250, conf_thr=0.65)
    new_tracks_list, corrected_list, avg_tracks = id_reassignment(tracks_list, dis_thr=200, conf_thr=0.7)
    new_tracks_list, corrected_list, avg_tracks = id_reassignment(tracks_list, dis_thr=150, conf_thr=0.75)

    write_to_file = True
    if write_to_file:
        output_folder_name = "reproduce"
        os.makedirs("{}".format(output_folder_name), exist_ok=True)
        for cam_name in tracks_list.keys():
            for tid, track in new_tracks_list[cam_name].items():
                output_filename = "{}/S001_{}.txt".format(output_folder_name, cam_name)
                write_tracking(output_filename, new_tracks_list[cam_name])

    return tracks_list


if __name__ == '__main__':
    main()
