target_type = "random_noise-mapping"
model_name = "bsd400_unet2_{}_ps3-ns8-net1-gpu1".format(target_type)
model_weight = None
workers = 4
epochs = 5000
start_epoch = 0
batch_size = 32
crop_size = 128
num_channel = 1
num_sim = 8
num_select = 8
print_freq = 10
test_freq = 10
resume = None
world_size = 1
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = None
multiprocessing_distributed = True


data_train = dict(
    type="lmdb",
    lmdb_file="./datasets/bsd400_gaussian25_ps3_ns8_net1_lmdb",
    meta_info_file="./datasets/bsd400_gaussian25_ps3_ns8_net1_lmdb_meta_info.pkl",
    crop_size=crop_size,
    target_type=target_type,
    random_flip=True,
    prune_dataset=None,
    patch_size=5,
    num_sim=num_sim,
    num_select=num_select,
    std=25.0,
    load_data_all=False,
    incorporate_noise=True,
    dtype="float32",
    ims_per_batch=batch_size,
    shuffle=True,
    aspect_ratio_grouping=False,
    train=True,
)

data_test = dict(
    type="bsd_npy",
    data_file='./datasets/bsd68_gaussian25.npy',
    target_file='./datasets/bsd68_groundtruth.npy',
    # norm=[-112.073, 367.314],
    norm=[0.0, 255.0],
    std=25.0,
    shuffle=False,
    ims_per_batch=1,
    aspect_ratio_grouping=False,
    train=False,
)

model = dict(
    type="common_denoiser",
    base_net=dict(
        type="unet2",
        n_channels=1,
        n_classes=1,
        activation_type="relu",
        bilinear=False,
        residual=True,
        use_bn=True
    ),

    denoiser_head=dict(
        head_type="supervise",
        loss_type="l2",
        loss_weight={"l2": 1},
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