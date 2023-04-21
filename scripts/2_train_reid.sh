cd deep-person-reid
conda activate torchreid

python3 run.py

conda deactivate

cp  log/aicity_imagenet_pretrained/model/model.pth.tar-60 checkpoints/synthetic_reid_model_60_epoch.pth