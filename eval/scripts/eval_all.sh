#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/config/run_config.sh"

MODEL_PATH="${EVAL_MODEL_PATH}"
EXP_NAME="${EVAL_EXP_NAME}"

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"

# for v-star
mkdir -p ./logs/vstar_logs
MODEL_KWARGS="./config/vstar.yaml"
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 python ./test/test_vstar_multi_images.py \
    --video_folder "/hkfs/work/workspace/scratch/uzivy-open-o3-data/eval/v-star/videos/" \
    --anno_file "/hkfs/work/workspace/scratch/uzivy-open-o3-data/eval/v-star/V_STaR_test.json" \
    --result_file "./logs/vstar_logs/${EXP_NAME}_vstar.json" \
    --model_path $MODEL_PATH \
    --model_kwargs $MODEL_KWARGS \
    --think_mode > "./logs/vstar_logs/test_${EXP_NAME}_vstar.log" 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./test/eval_vstar.py \
    --result_file  "./logs/vstar_logs/${EXP_NAME}_vstar.json" \
    --model_path $LLM_PATH > "./logs/vstar_logs/eval_${EXP_NAME}_vstar.log" 2>&1
    
# # for videomme
# mkdir -p ./logs/videomme_logs
# MODEL_KWARGS="./config/video_mme.yaml"
# NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./test/test_videomme.py \
#     --exp_name "${EXP_NAME}_mme" \
#     --data_dir "/mnt/bn/strategy-mllm-train/user/jiahao/datasets/Video-MME" \
#     --model_path $MODEL_PATH \
#     --model_kwargs $MODEL_KWARGS \
#     --N 1 \
#     --vote 'majority_voting' \
#     --think_mode > "./logs/videomme_logs/${EXP_NAME}_mme.log" 2>&1

# # for worldsense
# mkdir -p ./logs/world_logs
# MODEL_KWARGS="./config/world_sense.yaml"
# NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./test/test_worldsense.py \
#     --exp_name "${EXP_NAME}_wds" \
#     --data_dir "/mnt/bn/strategy-mllm-train/user/jiahao/datasets/WorldSense" \
#     --model_path $MODEL_PATH \
#     --model_kwargs $MODEL_KWARGS \
#     --N 1 \
#     --vote 'majority_voting' \
#     --think_mode > "./logs/world_logs/${EXP_NAME}_wds.log" 2>&1

# # for videommmu
# mkdir -p ./logs/videommmu_logs
# MODEL_KWARGS="./config/video_mmmu.yaml"
# NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./test/test_videommmu.py \
#     --exp_name "${EXP_NAME}_videommmu" \
#     --data_dir "/mnt/bn/strategy-mllm-train/user/jiahao/datasets/VideoMMMU/" \
#     --model_path $MODEL_PATH \
#     --model_kwargs $MODEL_KWARGS \
#     --N 1 \
#     --vote 'majority_voting' \
#     --think_mode > "./logs/videommmu_logs/${EXP_NAME}_videommmu.log" 2>&1

