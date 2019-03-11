#!/bin/bash

CUDA_VISIBLE_DEVICE=0 python train.py \
	--epochs=100 \
	--batch-size=16 \
	--cfg="cfg/yolov3.cfg" \
	--img-size=416 \

