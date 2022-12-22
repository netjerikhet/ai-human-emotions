config_path="./code/config/deeper_forensics.yaml"
checkpoint_path="./data/pretrained_model/vox-cpk.pth.tar"

echo "Running training script"

CUDA_VISIBLE_DEVICE=1 python code/run_train_per_emotion.py --config $config_path --checkpoint $checkpoint_path