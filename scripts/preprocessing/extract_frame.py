import glob
import os

import ffmpeg

fps = 30

data_root = "data/"

DATA_METADATA = {
    "train": {
        "market": {
            2: [8, 9, 10, 11, 12, 13],
            4: [20, 21, 22, 23, 24],
            6: [30, 31, 32, 33, 34, 35]
        },
        "office": {
            10: [53, 54, 55, 56, 57, 58],
            11: [59, 60, 61, 62, 63, 64]
        },
        "hospital": {
            11: [59, 60, 61, 62, 63, 64],
            12: [65, 66, 67, 68, 69, 70]
        },
        "storage": {
            15: [82, 83, 84, 85, 86, 87],
            16: [88, 89, 90, 91, 92, 93]
        },    # test on S018
        "warehouse": {
            19: [106, 107, 108, 109, 110, 111]
        }
    },
    "validation": {
        "market": {
            5: [25, 26, 27, 28, 29]
        },
        "office": {
            8: [41, 42, 43, 44, 45, 46]
        },
        "hospital": {
            13: [71, 72, 73, 74, 75]
        },
        "storage": {
            17: [94, 95, 96, 97, 98, 99]
        },
        "warehouse": {
            20: [112, 113, 114, 115, 116, 117]
        }
    }
}

train_seq_id = []
for scene_id in DATA_METADATA["train"].keys():
    for seq_id in DATA_METADATA["train"][scene_id].keys():
        train_seq_id.append((scene_id, seq_id))
val_seq_id = []
for scene_id in DATA_METADATA["validation"].keys():
    for seq_id in DATA_METADATA["validation"][scene_id].keys():
        val_seq_id.append((scene_id, seq_id))

train_folder_list = [data_root + "train/" + "S{:03d}/".format(sid) for sid in train_seq_id]
val_folder_list = [data_root + "validation/" + "S{:03d}/".format(sid) for sid in val_seq_id]

for folder_path in train_folder_list + val_folder_list:
    camera_folder_list = glob.glob(folder_path + '*/')
    for camera_path in camera_folder_list:
        frame_folder_path = camera_path + 'frame/'
        video_path = camera_path + 'video.mp4'

        print("processing {}".format(frame_folder_path))

        os.makedirs(frame_folder_path, exist_ok=True)

        ffmpeg.input(video_path).filter('fps', fps='30').output(frame_folder_path + '/%05d.jpg', start_number=0, **{'qscale:v': 2}).overwrite_output().run(quiet=True)
