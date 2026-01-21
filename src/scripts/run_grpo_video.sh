SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/config/run_config.sh"

cd "${REPO_ROOT}/src/r1-v"

MODEL_PATH="${GRPO_MODEL_PATH}"
EXP_NAME="${GRPO_EXP_NAME}"
OUT_DIR="${GRPO_OUT_DIR}"

DATA_ROOT=$(python -c "from configs.data_root import DATA_ROOT; print(DATA_ROOT)")
# mkdir -p ./train_logs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12321" \
    src/open_r1/grpo.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "${DATA_ROOT}/json_data/STGR-RL.json" \
    --deepspeed "local_scripts/zero3.json"\
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name $EXP_NAME \
    --save_steps 500 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 4
