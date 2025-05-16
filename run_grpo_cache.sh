unset LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd src
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /mnt/public/yhz/open-rs-main/recipes/accelerate_configs/zero2.yaml \
    --num_processes=6 open_r1/grpo_cache.py \
    --config /mnt/public/yhz/open-rs-main/recipes/grpo_cache_epsilon_0.5.yaml