#!/bin/bash

config_path="./code/config/deeper_forensics.yaml"
checkpoint_path="data/pretrained_model/00000145-checkpoint.pth.tar"
pretrained_checkpoint="data/pretrained_model/vox-cpk.pth.tar"
best_checkpoint="data/pretrained_model/best-checkpoint.pth.tar"
echo "Running output script"

CUDA_VISIBLE_DEVICE=1 python ./code/output.py --config $config_path --checkpoint $best_checkpoint 
