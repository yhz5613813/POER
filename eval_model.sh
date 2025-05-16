export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE=/home/yhz/.cache/huggingface/datasets
NUM_GPUS=4
MODEL=/mnt/public/yhz/open-rs-main/src/data/OpenRS-GRPO-Cache-qwen1.5b-bs336-distill
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.5,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK="minerva"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks src/open_r1/evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"