import argparse
import time

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.dataset_voc import VOCDetection
import pdb

def train(
        cfg,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        multi_scale=False,
        freeze_backbone=False,
        var=0,
):
    np.random.seed(666)
    device = torch_utils.select_device()

    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Initialize model
    model = Darknet(cfg, img_size)

    # Get dataloader
    train_dataset = VOCDetection(root=os.path.join('~', 'data', 'VOCdevkit'),
        batch_size=batch_size, img_size=img_size, multi_scale=multi_scale, mode='train')
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
        num_workers=8, pin_memory=False, shuffle=True, drop_last=True)

    lr0 = 0.001
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        model.to(device).train()

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
        if cfg.endswith('yolov3.cfg'):
            load_darknet_weights(model, 'weights/darknet53.conv.74')
            cutoff = 75
        elif cfg.endswith('yolov3-tiny.cfg'):
            load_darknet_weights(model, 'weights/yolov3-tiny.conv.15')
            cutoff = 15

        model.to(device).train()
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    # Start training
    t0 = time.time()
    # model_info(model)
    n_burnin = 1000  # number of burn-in batches
    for epoch in range(start_epoch, epochs):
        print(('%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))
        scheduler.step()

        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone and (epoch < 2):
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)  # running loss

        for i, (imgs, targets, numboxes, _) in enumerate(dataloader):
            if sum(numboxes) < 1:  # if no targets continue
                continue

             # SGD burn-in
            # if (epoch == 0) & (i <= n_burnin):
            #     lr = lr0 * (i / n_burnin) ** 4
            #     for g in optimizer.param_groups:
            #         g['lr'] = lr

            imgs = imgs.float().to(device)
            targets = [targets[i, :nL, :].float() for i,nL in enumerate(numboxes)]
            optimizer.zero_grad()

            # Compute loss
            loss = model(imgs, targets, var=var)
            loss.backward()
            optimizer.step()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)
            if(i % 20 == 0):
                print(('%8s%12s' + '%10.3g' * 7) % ('%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, len(dataloader) - 1),
                    rloss['xy'], rloss['wh'], rloss['conf'], rloss['cls'], rloss['loss'], model.losses['nT'], time.time() - t0))
            t0 = time.time()

        # Update best loss
        if rloss['loss'] < best_loss:
            best_loss = rloss['loss']

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        if epoch % 5 == 0:
            torch.save(checkpoint, 'weights/epoch_%03d.pt' % epoch)
        
        if epoch > 19 and epoch % 10 == 0:
            with torch.no_grad():
                mAP, R, P = test.test(cfg, weights='weights/epoch_%03d.pt'%epoch, batch_size=32, img_size=img_size)
                print(mAP, R, P)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
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
        accumulated_batches=opt.accumulated_batches,
        multi_scale=opt.multi_scale,
        var=opt.var,
    )
