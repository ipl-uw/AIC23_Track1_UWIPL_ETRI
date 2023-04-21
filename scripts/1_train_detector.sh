cd mmyolo
conda activate mmyolo

python3 tools/train.py configs/custom_dataset/yolov7_base.py

python3 tools/train.py configs/custom_dataset/yolov7_hospital.py
python3 tools/train.py configs/custom_dataset/yolov7_market.py
python3 tools/train.py configs/custom_dataset/yolov7_office.py
python3 tools/train.py configs/custom_dataset/yolov7_storage.py

conda deactivate