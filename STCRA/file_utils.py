import collections


# ==================== FILE I/O FUNCTIONS ====================
def xywh2location(x, y, w, h, w_ratio=0.5, h_ratio=0.95):
    """
    Convert the bounding box to the estimated feet location on ground-plane.

    Parameters:
        x (float): The x coordinate of the top-left corner of the bounding box.
        y (float): The y coordinate of the top-left corner of the bounding box.
        w (float): The width of the bounding box.
        h (float): The height of the bounding box.
        w_ratio (float): The ratio of the width of the bounding box to the width of the feet.
        h_ratio (float): The ratio of the height of the bounding box to the height of the feet.

    Returns:
        tuple: The estimated feet location on ground-plane (in image coordinate).
    """
    return x + w * w_ratio, y + h * h_ratio


def load_tracking(filename, debug_frame=0, enable_feet=False):
    tracks = collections.defaultdict(dict)
    cnt_duplicate = 0
    cnt_bbox = 0
    duplicate_id = collections.defaultdict(int)
    with open(filename, 'r') as file:
        # Read each line of the file
        for line in file:
            line = line.strip()
            values = line.split(',')

            assert len(values) == 16 if enable_feet else 10, f"Wrong format of tracking in {filename} with line: {line}"

            frame_id, track_id = int(values[0]), int(values[1])
            x, y, w, h = float(values[2]), float(values[3]), float(values[4]), float(values[5])
            score = float(values[6])

            # compute (fx, fy) if the confidence of feet detection is high enough
            # otherwise, use the pre-assumed points of each bounding box
            if enable_feet and float(values[12]) > 0.5 and float(values[15]) > 0.5:
                fx = (float(values[10]) + float(values[13])) / 2.0
                fy = (float(values[11]) + float(values[14])) / 2.0
            else:
                fx, fy = xywh2location(x, y, w, h)

            if debug_frame and frame_id > debug_frame:
                break

            if frame_id in tracks[track_id].keys():
                cnt_duplicate += 1
                tracks[track_id][frame_id].append([x, y, w, h, fx, fy, score])
                duplicate_id[track_id] += 1
            else:
                tracks[track_id][frame_id] = [[x, y, w, h, fx, fy, score]]
            cnt_bbox += 1
    print(f"duplicate detection in {filename}: \t{cnt_duplicate}/{cnt_bbox}({cnt_duplicate / cnt_bbox:.2%})")
    # print sorted duplicate_id
    print(f"\t duplicate id stats {sorted(duplicate_id.items(), key=lambda _x: _x[0], reverse=False)}")
    return tracks


def write_tracking(filename, track):
    lines = []
    for track_id, track in track.items():
        for frame_id, detection in track.items():
            if detection:
                assert len(
                    detection) == 1, f"More than one detection in frame {frame_id} of track {track_id} {detection}"
                assert len(
                    detection[0]) >= 4, f"Wrong format of detection in frame {frame_id} of track {track_id} {detection}"
                x, y, w, h = detection[0][:4]
                lines.append((frame_id, track_id, x, y, w, h, 1.0, -1, -1, -1))
    lines = sorted(lines, key=lambda x: (x[0], x[1]))
    with open(filename, 'w') as file:
        for line in lines:
            file.write(','.join([str(x) for x in line]) + '\n')
