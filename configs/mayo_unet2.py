target_type = "noise-sim"
model_name = "mayo_unet2_{}".format(target_type)
model_weight = 'results/mayo_unet2_noise-sim/checkpoint_0079.pth.tar'
workers = 4
epochs = 100
start_epoch = 0
batch_size = 16
crop_size = 512
num_channel = 1
num_sim = 8
num_select = 8
print_freq = 100
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
    type="mayo",
    data_file='./datasets/Mayo/mayo_train.txt',
    crop_size=crop_size,
    neighbor=2,
    range=[-160, 240],
    ks=7,
    th=40,
    target_type=target_type,
    random_flip=True,
    ims_per_batch=batch_size,
    shuffle=True,
    train=True,
)

data_test = dict(
    type="mayo",
    data_file='./datasets/Mayo/mayo_test.txt',
    target_type='noise-clean',
    range=[-160, 240],
    random_flip=False,
    crop_size=None,
    shuffle=False,
    ims_per_batch=1,
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