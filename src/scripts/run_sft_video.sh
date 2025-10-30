cd src/r1-v
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export WANDB_MODE="offline"

# You should refine the model_path and exp_name here.
MODEL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
EXP_NAME="sft"
OUT_DIR="/path/to/ckpts/${EXP_NAME}"

DATA_ROOT=$(python -c "from configs.data_root import DATA_ROOT; print(DATA_ROOT)")
# mkdir -p ./train_logs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/sft_multi_task.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "${DATA_ROOT}/json_data/STGR-SFT.json" \
    --deepspeed "local_scripts/zero2.json" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $EXP_NAME \
    --save_steps 500 \
    --max_grad_norm 5 \
    --save_only_model true
