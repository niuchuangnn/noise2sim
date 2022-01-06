import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from ..config.config import Config
from ..modeling.architectures import build_architecture
from ..solver import make_lr_scheduler, make_optimizer
from ..data.bulid_data import build_dataset
from ..utils.miscellaneous import mkdir, save_config
import numpy as np


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def clip_to_uint8(arr):
    if isinstance(arr, np.ndarray):
        return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    x = torch.clamp(arr * 255.0 + 0.5, 0, 255).to(torch.uint8)
    return x


def calculate_psnr(a, b, axis=None):
    a, b = [clip_to_uint8(x) for x in [a, b]]
    if isinstance(a, np.ndarray):
        a, b = [x.astype(np.float32) for x in [a, b]]
        x = np.mean((a - b)**2, axis=axis)
        return np.log10((255 * 255) / x) * 10.0
    a, b = [x.to(torch.float32) for x in [a, b]]
    x = torch.mean((a - b)**2)
    return torch.log((255 * 255) / x) * (10.0 / math.log(10))


def main(config_file):
    cfg = Config.fromfile(config_file)
    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)

    output_config_path = os.path.join(output_dir, 'config.py')
    save_config(cfg, output_config_path)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.world_size = ngpus_per_node * cfg.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg.copy()))
    else:
        # Simply call main_worker function
        main_worker(cfg.gpu, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    cfg.gpu = gpu

    # suppress printing if not master
    if cfg.multiprocessing_distributed and cfg.gpu != 0:
        def print_pass(*cfg):
            pass
        builtins.print = print_pass

    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)
    # create model
    model = build_architecture(cfg.model)
    print(model)

    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.workers = int((cfg.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    optimizer = make_optimizer(cfg, model)
    scheduler = None

    # optionally resume from a checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            if cfg.gpu is None:
                checkpoint = torch.load(cfg.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(cfg.gpu)
                checkpoint = torch.load(cfg.resume, map_location=loc)
            cfg.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True

    if "lr_type" in cfg.solver:
        scheduler = make_lr_scheduler(cfg, optimizer)

    # Data loading code
    train_dataset = build_dataset(cfg.data_train)

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if cfg.data_test is not None:
        dataset_val = build_dataset(cfg.data_test)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)
    else:
        val_loader = None

    psnr_best = 0
    best_epoch = 0
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        if scheduler is not None:
            scheduler.step()

        # train for one epoch
        train(train_loader, model, optimizer, epoch, cfg)

        if not cfg.multiprocessing_distributed or (cfg.multiprocessing_distributed
                and cfg.rank % ngpus_per_node == 0 and (epoch+1) % cfg.test_freq == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(cfg.results.output_dir, epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_last.pth.tar'.format(cfg.results.output_dir))
            if (epoch+1) == cfg.epochs:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_final.pth.tar'.format(cfg.results.output_dir))

            if val_loader is not None:
                model.eval()
                psnrs = []
                for _, (images, images_clean, std, idx) in enumerate(val_loader):
                    inputs = images.to(cfg.gpu)
                    with torch.no_grad():
                        outputs = model(inputs)

                    images_clean = images_clean.to(cfg.gpu)
                    psnr = calculate_psnr(outputs, images_clean)
                    psnrs.append(psnr.cpu().numpy())

                psnr_mean = np.array(psnrs).mean()
                if psnr_best < psnr_mean:
                    psnr_best = psnr_mean
                    best_epoch = epoch
                print(psnr_mean)
                print("Best PSNR: {}, epoch: {}".format(psnr_best, best_epoch))
                model.train()


def train(train_loader, model, optimizer, epoch, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lr = AverageMeter('lr', ':.6f')
    lr.update(optimizer.param_groups[0]["lr"])

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, lr],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images_noise, images_target, images_ignore, idx) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if cfg.gpu is not None:
            images_noise = images_noise.cuda(cfg.gpu, non_blocking=True)
            images_target = images_target.cuda(cfg.gpu, non_blocking=True)

        if len(images_noise.shape) == 5:
            images_noise = images_noise.reshape([-1, images_noise.shape[2], images_noise.shape[3], images_noise.shape[4]])
            images_target = images_target.reshape(images_noise.shape)

        if images_ignore.max() > 0:
            images_ignore = images_ignore.cuda(cfg.gpu, non_blocking=True)
            mask = 1 - images_ignore
            if images_noise.shape[0] != mask.shape[0]:
                mask = mask[0:1, :, :, :]
        else:
            mask = None

        # compute output
        loss_dict, model_info = model(images_noise, target=images_target, mask=mask)
        loss = loss_dict["loss_total"]
        losses.update(loss.item(), images_target[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
