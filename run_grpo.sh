unset LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE="offline"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd src
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /mnt/public/yhz/open-rs-main/recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 open_r1/grpo.py \
    --config /mnt/public/yhz/open-rs-main/recipes/calculate_time_1.5b.yaml