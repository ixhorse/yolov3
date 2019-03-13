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
    device = torch_utils.select_device()

    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Initialize model
    model = Darknet(cfg, img_size)

    # Get dataloader
    train_dataset = VOCDetection(root=os.path.join('~', 'data', 'VOCdevkit'), 
        batch_size=batch_size, img_size=img_size, multi_scale=multi_scale, augment=True)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
        num_workers=6, pin_memory=False, shuffle=True, drop_last=True)

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

        start_epoch = checkpoint['epoch'] + 1
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    # Start training
    t0 = time.time()
    # model_info(model)
    n_burnin = min(round(train_dataset.nB / 5 + 1), 1000)  # number of burn-in batches
    for epoch in range(epochs):
        epoch += start_epoch
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
        loss_per_target = rloss['loss'] / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        if(epoch % 5 == 0):
            torch.save(checkpoint, 'weights/epoch_%03d.pt' % epoch)

        # Save best checkpoint
        if best_loss == loss_per_target:
            torch.save(checkpoint, 'weights/best.pt')

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
