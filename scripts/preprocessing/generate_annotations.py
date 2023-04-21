import collections
import json
import os

DATA_METADATA = {
    "train": {
        "market": {
            2: [8, 9, 10, 11, 12, 13],
            4: [20, 21, 22, 23, 24],
            6: [30, 31, 32, 33, 34, 35]
        },
        "office": {
            7: [36, 37, 38, 39, 40],
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
        },  # test on S018
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

output_file = ""

cat_info = {'id': 1, 'name': 'person'}


def load_detection(filename, coco_format_json, image_idx, bbox_idx, img_folder, sample_rate, sample_offset):
    tracks = collections.defaultdict(dict)
    prev_frame_id = -1

    with open(filename, 'r') as file:
        annotations = collections.defaultdict(list)
        for line in file:
            line = line.strip()
            values = line.split(',')
            frame_id, track_id = int(values[0]), int(values[1])
            x, y, w, h = int(values[2]), int(values[3]), int(values[4]), int(values[5])
            annotations[frame_id].append([x, y, w, h])

        for frame_id in annotations.keys():
            if (frame_id + sample_offset) % sample_rate == 0:
                image_idx += 1
                img_name = "{:05d}.jpg".format(frame_id)
                coco_format_json["images"].append(
                    {
                        'file_name': os.path.join(img_folder, img_name),
                        'id': image_idx,
                        'width': 1920,
                        'height': 1080
                    }
                )

                for x, y, w, h in annotations[frame_id]:
                    bbox_idx += 1
                    coco_format_json["annotations"].append(
                        {
                            'image_id': image_idx,
                            'id': bbox_idx,
                            'category_id': 1,
                            'bbox': [x, y, w, h],
                            'area': w * h,
                            'segmentation': [],
                            'iscrowd': 0
                        }
                    )

    return image_idx, bbox_idx, coco_format_json


def prepare_annotations(scene_tag="all", train_and_val=True, sample_rate=20, sample_offset=0):
    assert scene_tag in ["all", "market", "office", "hospital", "storage", "warehouse"]  # "shop" only in testing

    coco_format_json = {
        "images": [],
        "categories": [cat_info],
        "annotations": []
    }

    image_idx = 0
    bbox_idx = 0

    train_seq_id = []
    if scene_tag == "all":
        for scene_id in DATA_METADATA["train"].keys():
            for seq_id in DATA_METADATA["train"][scene_id].keys():
                train_seq_id.append((scene_id, seq_id))
    else:
        for seq_id in DATA_METADATA["train"][scene_tag].keys():
            train_seq_id.append((scene_tag, seq_id))

    for scene_id, seq_id in train_seq_id:
        for cam_id in DATA_METADATA["train"][scene_id][seq_id]:
            seq_tag = "S{:03d}".format(seq_id)
            cam_tag = "c{:03d}".format(cam_id)

            gt_file = "data/train/{}/{}/label.txt".format(seq_tag, cam_tag)
            img_folder = "data/train/{}/{}/frame/".format(seq_tag, cam_tag)

            image_idx, bbox_idx, coco_format_json = load_detection(gt_file, coco_format_json, image_idx, bbox_idx,
                                                                   img_folder, sample_rate, sample_offset)

            print(seq_tag, cam_tag, image_idx, bbox_idx)

    if train_and_val:
        val_seq_id = []
        if scene_tag == "all":
            for scene_id in DATA_METADATA["validation"].keys():
                for seq_id in DATA_METADATA["validation"][scene_id].keys():
                    val_seq_id.append((scene_id, seq_id))
        else:
            for seq_id in DATA_METADATA["validation"][scene_tag].keys():
                val_seq_id.append((scene_tag, seq_id))

        for scene_id, seq_id in val_seq_id:
            for cam_id in DATA_METADATA["validation"][scene_id][seq_id]:
                seq_tag = "S{:03d}".format(seq_id)
                cam_tag = "c{:03d}".format(cam_id)

                gt_file = "data/validation/{}/{}/label.txt".format(seq_tag, cam_tag)
                img_folder = "data/validation/{}/{}/frame/".format(seq_tag, cam_tag)

                image_idx, bbox_idx, coco_format_json = load_detection(gt_file, coco_format_json, image_idx, bbox_idx,
                                                                       img_folder, sample_rate, sample_offset)

                print(seq_tag, cam_tag, image_idx, bbox_idx)

    if train_and_val:
        output_file_name = "train_{}_val_{}_sr_{}_{}_img_{}.json".format(scene_tag, scene_tag, sample_rate,
                                                                         sample_offset, image_idx)
    else:
        output_file_name = "train_{}_sr_{}_{}_img_{}.json".format(scene_tag, sample_rate, sample_offset, image_idx)
    with open(os.path.join("data/annotations", output_file_name), 'w') as f:
        json.dump(coco_format_json, f)
    print("saved to {}".format(os.path.join("data/annotations", output_file_name)))


if __name__ == "__main__":
    prepare_annotations("all", True, 20, 10)
    prepare_annotations("market", True, 20, 0)
    prepare_annotations("office", True, 20, 0)
    prepare_annotations("hospital", True, 20, 0)
    prepare_annotations("storage", True, 20, 0)
