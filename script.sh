#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py \
	--epochs=101 \
	--batch-size=16 \
	--cfg="cfg/yolov3-voc.cfg" \
	--img-size=416 \

