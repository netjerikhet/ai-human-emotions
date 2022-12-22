### This script trains the model on a given dataset ###

config_path="./code/config/deeper_forensics.yaml"  ## Path of model config file (Please check the config file to see the setup)
checkpoint_path="./data/pretrained_model/vox-cpk.pth.tar" ## Path of a pretrained checkpoint file to continue training (Not needed if training without weights)

echo "Running training script" 


## Check run_train.py for information regarding the arguements passed
CUDA_VISIBLE_DEVICE=1 python ./code/run_train.py --config $config_path --checkpoint $checkpoint_path  