# Installation 
Reference to
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.

# Run pose estimation for foot points

Put the tracking results generated from previous steps in `tracking_result` folder

```
python demo/top_down_video_demo_with_track_file.py <tracking_file.txt> \ 
       configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
       https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
       --video-path <video_file.mp4> \
       --out-file <out_keypoint.json>

python tools/convert.py
```
