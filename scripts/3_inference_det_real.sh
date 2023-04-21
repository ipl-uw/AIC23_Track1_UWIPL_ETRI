conda activate botsort_env
cd BoT-SORT

python3 tools/aic_get_detection_S001.py -f yolox/exps/example/mot/yolox_x_mix_det.py -c bytetrack_x_mot17.pth.tar

conda deactivate