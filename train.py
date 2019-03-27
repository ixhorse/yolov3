import argparse
import time

import torch.distributed as dist
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.dataset_voc import VOCDetection

import torch
from pprint import pprint
import pdb

def train(
        cfg,
        img_size=416,
        resume=False,
        epochs=270,
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
        num_workers=4
):
    np.random.seed(666)
    device = torch_utils.select_device()

    if not multi_scale:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Initialize model
    model = Darknet(cfg, img_size).to(device)

    # Optimizer
    lr0 = 0.001  # initial learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=.9, weight_decay=0.0005)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']
        del checkpoint  # current, saved

    else:
        # Initialize model with backbone (optional)
        if cfg.endswith('yolov3-tiny.cfg'):
            cutoff = load_darknet_weights(model, 'weights/yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, 'weights/darknet53.conv.74')

    # Transfer learning (train only YOLO layers)
    # for i, (name, p) in enumerate(model.named_parameters()):
    #     p.requires_grad = True if (p.shape[0] == 255) else False

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1, last_epoch=start_epoch - 1)

    # Dataset
    train_dataset = VOCDetection(root=os.path.join('~', 'data', 'VOCdevkit'), img_size=img_size, mode='train')

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend=opt.backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
        model = torch.nn.parallel.DistributedDataParallel(model)
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = None

    # Dataloader
    dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=train_dataset.collate_fn,
                            sampler=sampler)

    # Start training
    nB = len(dataloader)
    t = time.time()
    # model_info(model)
    n_burnin = 1000  # burn-in batches
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss = defaultdict(float)  # mean loss

        for i, (imgs, targets, _, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            nT = len(targets)
            if nT == 0:  # if no targets continue
                continue

            # Plot images with bounding boxes
            plot_images = False
            if plot_images:
                from matplotlib import pyplot as plt
                fig = plt.figure(figsize=(10, 10))
                for ip in range(batch_size):
                    labels = xywh2xyxy(targets[targets[:, 0] == ip, 2:6]).numpy() * img_size
                    plt.subplot(4, 4, ip + 1).imshow(imgs[ip].numpy().transpose(1, 2, 0))
                    plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '.-')
                    plt.axis('off')
                fig.tight_layout()
                fig.savefig('batch_%g.jpg' % i, dpi=fig.dpi)

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = lr0 * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            optimizer.zero_grad()
            # Run model
            pred = model(imgs)
            # Build targets
            target_list = build_targets(model, targets)
            # Compute loss
            loss, loss_dict = compute_loss(pred, target_list)
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nB:
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            for key, val in loss_dict.items():
                mloss[key] = (mloss[key] * i + val) / (i + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nB - 1),
                mloss['xy'], mloss['wh'], mloss['conf'], mloss['cls'],
                mloss['total'], nT, time.time() - t)
            t = time.time()
            if i % 30 == 0:
                print(s)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)

        # Update best loss
        if mloss['total'] < best_loss:
            best_loss = mloss['total']

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                    'best_loss': best_loss,
                    'model': model.module.state_dict() if type(
                        model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                    'optimizer': optimizer.state_dict()}
        if epoch % 5 == 0:
            torch.save(checkpoint, 'weights/epoch_%03d.pt' % epoch)

        if epoch > 9 and epoch % 10 == 0:
            with torch.no_grad():
                APs, mAP = test.test(cfg, weights=None, batch_size=32, img_size=img_size, model=model)
                pprint(APs)
                print(mAP)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=270, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    train(
        opt.cfg,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        multi_scale=opt.multi_scale,
        num_workers=opt.num_workers
    )
