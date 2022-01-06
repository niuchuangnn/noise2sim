target_type = "noise-sim"
model_name = "pcct_unet2_{}_leg".format(target_type)
model_weight = 'results/pcct_unet2_noise-sim_leg/checkpoint.pth.tar'
workers = 4
epochs = 1000
start_epoch = 0
batch_size = 32
crop_size = 256
num_channel = 1
print_freq = 10
test_freq = 20
resume = None
world_size = 1
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = None
multiprocessing_distributed = True


data_train = dict(
    type="pcct",
    data_file='./datasets/leg/LDCT.mat',
    crop_size=crop_size,
    neighbor=5,
    slice_crop=None,
    hu_range=[0, 1.5],
    ks=9,
    th=0.2,
    target_type=target_type,
    random_flip=True,
    ims_per_batch=batch_size,
    shuffle=True,
    train=True,
)

data_test = dict(
    type="pcct",
    data_file='./datasets/leg/LDCT.mat',
    data_file_clean='./datasets/leg/NDCT-reference.mat',
    crop_size=None,
    center_crop=[10, 310, 90, 390],
    neighbor=5,
    hu_range=[0, 1.5],
    slice_crop=[60, 100],
    ks=9,
    th=None,
    target_type=target_type,
    random_flip=False,
    ims_per_batch=1,
    shuffle=False,
    train=False,
)

model = dict(
    type="common_denoiser",
    base_net=dict(
        type="unet2",
        n_channels=5,
        n_classes=5,
        activation_type="relu",
        bilinear=False,
        residual=True,
        use_bn=True
    ),

    denoiser_head=dict(
        loss_type="l2",
        loss_weight={"l2": 10},
    ),

    weight=None,
)

solver = dict(
    type="adam",
    base_lr=0.0005,
    bias_lr_factor=1,
    betas=(0.1, 0.99),
    weight_decay=0,
    weight_decay_bias=0,
    lr_type="ramp",
    max_iter=epochs,
    ramp_up_fraction=0.1,
    ramp_down_fraction=0.3,
)

results = dict(
    output_dir="./results/{}".format(model_name),
)