target_type = "noise-sim"
model_type = "unet2"
data_type = "ldct"
train_file = "./datasets/Mayo/mayo_train.txt"
test_file = "./datasets/Mayo/mayo_test.txt"

epochs = 300
start_epoch = 0
batch_size = 32
crop_size = 512
neighbor = 2
ks = 7
th = 30
flag = "l1"


workers = 4
print_freq = 100
test_freq = 10
world_size = 1
rank = 0
dist_url = 'tcp://localhost:10004'
dist_backend = "nccl"
seed = None
gpu = None
multiprocessing_distributed = True


model_name = "{}_{}_{}_mayo_{}".format(data_type, model_type, target_type, flag)
model_weight = 'results/{}/checkpoint.pth.tar'.format(model_name)
if start_epoch > 0:
    resume = "results/{}/checkpoint_{:04d}.pth.tar".format(model_name, start_epoch)
else:
    resume = None

data_train = dict(
    type=data_type,
    data_file=train_file,
    crop_size=crop_size,
    neighbor=neighbor,
    hu_range=[-160, 240],
    ks=ks,
    th=th,
    target_type=target_type,
    random_flip=True,
    ims_per_batch=batch_size,
    shuffle=True,
    train=True,
)

data_test = dict(
    type=data_type,
    data_file=test_file,
    target_type='noise-clean',
    hu_range=[-160, 240],
    random_flip=False,
    crop_size=None,
    shuffle=False,
    ims_per_batch=1,
    train=False,
)

model = dict(
    type="common_denoiser",
    base_net=dict(
        type=model_type,
        n_channels=1,
        n_classes=1,
        activation_type="relu",
        bilinear=False,
        residual=True,
        use_bn=True
    ),

    denoiser_head=dict(
        loss_type="l1",
        loss_weight={"l1": 10},
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