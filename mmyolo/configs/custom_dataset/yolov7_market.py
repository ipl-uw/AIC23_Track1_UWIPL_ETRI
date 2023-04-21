_base_ = '../yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py'

max_epochs = 10
data_root = '../../data/'  # change to custom data root path
load_from = "/path/to/weights/base_epoch_55.pth"

train_batch_size_per_gpu = 8
train_num_workers = 4

save_epoch_intervals = 5

base_lr = 0.0005

# anchors = [
#     [(68, 69), (154, 91), (143, 162)],  # P3/8
#     [(242, 160), (189, 287), (391, 207)],  # P4/16
#     [(353, 337), (539, 341), (443, 432)]  # P5/32
# ]

class_name = ('person', )
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60)]

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=0,
    val_interval=save_epoch_intervals
)

model = dict(
    backbone=dict(arch='X'),
    neck=dict(
        in_channels=[640, 1280, 1280],
        out_channels=[160, 320, 640],
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.4,
            block_ratio=0.4,
            num_blocks=3,
            num_convs_in_block=2),
        use_repconv_outs=False),
    bbox_head=dict(head_module=dict(in_channels=[320, 640, 1280])))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/train_market_val_market_sr_20_0_img_19965.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/train_all_val_all_sr_600_0_img_2569.json',
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/train_all_val_all_sr_600_0_img_2569.json')
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=10,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=250))

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
