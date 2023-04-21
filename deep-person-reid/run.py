import torchreid

datamanager = torchreid.data.ImageDataManager(
    root="reid-data",
    sources="dukemtmcreid",
    targets="dukemtmcreid",
    height=256,
    width=128,
    batch_size_train=64,
    batch_size_test=100,
    transforms=["random_flip"]
)

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="softmax",
    pretrained=True
)

weight_path = "checkpoints/osnet_x1_0_imagenet.pth"

torchreid.utils.load_pretrained_weights(model, weight_path)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir="log/aicity_imagenet_pretrained",
    max_epoch=60,
    eval_freq=10,
    print_freq=20,
    test_only=False
)
