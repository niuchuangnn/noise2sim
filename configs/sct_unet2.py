target_type = "noise-sim"
model_name = "sct_unet2_{}".format(target_type)
model_weight = 'results/sct_unet2_noise-sim/checkpoint_0199.pth.tar'
workers = 4
epochs = 200
start_epoch = 0
batch_size = 32
crop_size = 256
num_channel = 1
print_freq = 100
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
    type="sct",
    data_file='./datasets/rec_spirSIRT6-10ch.mat',
    crop_size=crop_size,
    neighbor=5,
    range=[-300, 250],
    ks=9,
    th=40,
    target_type=target_type,
    random_flip=True,
    ims_per_batch=batch_size,
    shuffle=True,
    train=True,
)

data_test = dict(
    type="sct",
    data_file='./datasets/rec_spirSIRT6-10ch.mat',
    crop_size=394,
    range=[-300, 250],
    ks=9,
    th=40,
    target_type='noise-clean',
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
        head_type="supervise",
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