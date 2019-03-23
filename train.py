import argparse
import time

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.dataset_voc import VOCDetection

import torch
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
):
    np.random.seed(666)
    device = torch_utils.select_device()

    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Initialize model
    model = Darknet(cfg, img_size)

    # Get dataloader
    train_dataset = VOCDetection(root=os.path.join('~', 'data', 'VOCdevkit'), img_size=img_size, mode='train')
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
        num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    lr0 = 0.001  # initial learning rate
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     p.requires_grad = True if (p.shape[0] == 255) else False
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

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

        # Set optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=.9)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device).train()

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 80], gamma=0.1)

    # Start training
    t0 = time.time()
    # model_info(model)
    n_burnin = min(round(len(dataloader) / 5 + 1), 1000)  # burn-in batches
    for epoch in range(epochs):
        model.train()
        epoch += start_epoch

        print(('\n%8s%12s' + '%10s' * 6) % (
            'Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'time'))

        # Update scheduler (automatic)
        scheduler.step()

        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone and (epoch < 1):
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)  # running loss
        
        for i, (imgs, targets, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            # convert to [imgidx, cls, x, y, w, h] 
            new_targets = []   
            for idx, target in enumerate(targets):
                target = target[target[:, 0] == 1]
                target[:, 0] = idx
                new_targets.append(target)
            targets = torch.cat(new_targets, 0).to(device)

            # SGD burn-in
            # if (epoch == 0) & (i <= n_burnin):
            #     lr = lr0 * (i / n_burnin) ** 4
            #     for g in optimizer.param_groups:
            #         g['lr'] = lr 
                       
            optimizer.zero_grad()
            # Run model
            pred = model(imgs.to(device))

            # Build targets
            target_list = build_targets(model, targets, pred)

            # Compute loss
            loss, loss_dict = compute_loss(pred, target_list)

            loss.backward()
            optimizer.step()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in loss_dict.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            if(i % 30 == 0):
                print(('%8s%12s' + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, len(dataloader) - 1),
                    rloss['xy'], rloss['wh'], rloss['conf'], rloss['cls'], rloss['total'], time.time() - t0))
            t0 = time.time()

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataloader.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataloader.img_size)

        # Update best loss
        if rloss['total'] < best_loss:
            best_loss = rloss['total']

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.module.state_dict() if type(model) is nn.DataParallel else model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        if epoch % 5 == 0:
            torch.save(checkpoint, 'weights/epoch_%03d.pt' % epoch)
        
        if epoch > 19 and epoch % 10 == 0:
            with torch.no_grad():
                P, R, mAP = test.test(cfg, weights='weights/epoch_%03d.pt'%epoch, batch_size=32, img_size=img_size)
                print(P, R, mAP)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=270, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', type=str, default=None, help='resume training flag')
    parser.add_argument('--var', type=float, default=0, help='test variable')
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
    )
