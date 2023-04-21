conda activate mmyolo
cd mmyolo

python3 tools/aic_get_detection_syn.py configs/custom_dataset/yolov7_base.py work_dirs/yolov7_market/epoch_10_S003_market.pth --scene S003 --out-dir ../data/
python3 tools/aic_get_detection_syn.py configs/custom_dataset/yolov7_base.py work_dirs/yolov7_office/epoch_10_S009_office.pth --scene S009 --out-dir ../data/
python3 tools/aic_get_detection_syn.py configs/custom_dataset/yolov7_base.py work_dirs/yolov7_hospital/epoch_10_S014_hospital.pth --scene S014 --out-dir ../data/
python3 tools/aic_get_detection_syn.py configs/custom_dataset/yolov7_base.py work_dirs/yolov7_storage/epoch_100_S018_storage.pth --scene S018 --out-dir ../data/

python3 tools/aic_get_detection_syn.py configs/custom_dataset/yolov7_base.py work_dirs/yolov7_base/epoch_100_base.pth --scene S021 --out-dir ../data/
python3 tools/aic_get_detection_syn.py configs/custom_dataset/yolov7_base.py work_dirs/yolov7_base/epoch_100_base.pth --scene S022 --out-dir ../data/

conda deactivate