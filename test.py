import argparse
import json
import time
from pathlib import Path

from models import *
from utils.datasets import *
from utils.dataset_voc import VOCDetection
from utils.utils import *
import pdb

def test(
        cfg,
        weights,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
        save_json=False,
        model=None
):
    device = torch_utils.select_device()

    if model is None:
        # Initialize model
        model = Darknet(cfg, img_size)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

    model.to(device).eval()

    # Get dataloader
    vocset = VOCDetection(root=os.path.join('~', 'data', 'VOCdevkit'), splits=((2007, 'test'),),
                        batch_size=batch_size, img_size=img_size, mode='test')
    dataloader = torch.utils.data.DataLoader(vocset, batch_size=batch_size, num_workers=16)

    nC = vocset.num_class #num class
    classes = vocset.classes

    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    mP, mR, mAPs, TP, jdict = [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    coco91class = coco80_to_coco91_class()
    for batch_i, (imgs, targets, numboxes, shapes) in enumerate(dataloader):
        targets = targets.to(device)
        t = time.time()
        # pdb.set_trace()
        targets = [targets[i, :nL, :].float() for i,nL in enumerate(numboxes)]
        output = model(imgs.to(device))
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)

        # Compute average precision for each sample
        for si, detections in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            seen += 1

            if detections is None:
                # If there are labels but no detections mark as zero AP
                if len(labels) != 0:
                    mP.append(0), mR.append(0), mAPs.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections[(-detections[:, 4]).argsort()]

            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                box = detections[:, :4].clone()  # xyxy
                scale_coords(img_size, box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

                # add to json dictionary
                for di, d in enumerate(detections):
                    jdict.append({
                        'image_id': int(Path(paths[si]).stem.split('_')[-1]),
                        'category_id': coco91class[int(d[6])],
                        'bbox': [float3(x) for x in box[di]],
                        'score': float3(d[4] * d[5])
                    })

            # If no labels add number of detections as incorrect
            correct = []
            if len(labels) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mP.append(0), mR.append(0), mAPs.append(0)
                continue
            else:
                # Extract target boxes as (x1, y1, x2, y2)
                target_box = xywh2xyxy(labels[:, 1:5]) * img_size
                target_cls = labels[:, 0]

                detected = []
                for *pred_box, conf, cls_conf, cls_pred in detections:
                    # Best iou, index between pred and targets
                    iou, bi = bbox_iou(pred_box, target_box).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and cls_pred == target_cls[bi] and bi not in detected:
                        correct.append(1)
                        detected.append(bi)
                    else:
                        correct.append(0)

            # Compute Average Precision (AP) per class
            AP, AP_class, R, P = ap_per_class(tp=np.array(correct),
                                              conf=detections[:, 4].cpu().numpy(),
                                              pred_cls=detections[:, 6].cpu().numpy(),
                                              target_cls=target_cls.cpu().numpy())

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mP.append(P.mean())
            mR.append(R.mean())
            mAPs.append(AP.mean())

            # Means of all images
            mean_P = np.mean(mP)
            mean_R = np.mean(mR)
            mean_mAP = np.mean(mAPs)

        # Print image mAP and running mean mAP
    #     print(('%11s%11s' + '%11.3g' * 4 + 's') %
    #           (seen, vocset.nF, mean_P, mean_R, mean_mAP, time.time() - t))
    #
    # # Print mAP per class
    # print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP') + '\n\nmAP Per Class:')
    #
    # for i, c in enumerate(classes):
    #     print('%15s: %-.4f' % (c, AP_accum[i] / (AP_accum_count[i] + 1E-16)))

    # Return mAP
    return mean_P, mean_R, mean_mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-voc.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    with torch.no_grad():
        mAP = test(
            opt.cfg,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.iou_thres,
            opt.conf_thres,
            opt.nms_thres,
            opt.save_json)
