import os
import torch
from utils.dataset_voc import VOCDetection
import pdb

voc = VOCDetection(root=os.path.join('~', 'data', 'VOCdevkit'), batch_size=1,
                   img_size=416)

dataloader = torch.utils.data.DataLoader(voc, batch_size=16,
    num_workers=8, pin_memory=False)

for imgs, targets, numboxes, _ in dataloader:
    print(sum(numboxes))
