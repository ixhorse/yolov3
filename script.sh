#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python train.py \
	--epochs=201 \
	--batch-size=32 \
	--cfg="cfg/yolov3-voc.cfg" \
	--img-size=416 \
	--num-workers=8

