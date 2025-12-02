#!/bin/bash

# 1. Hugging Face Cache 설정
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

export HF_HOME="/home/win2dvp21/cmu/NLPLSMA/hf_models"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="/home/win2dvp21/cmu/NLPLSMA/hf_models/datasets"
export HF_MODEL_CACHE="/home/win2dvp21/cmu/NLPLSMA/hf_models/models"

export HF_HUB_ENABLE_HF_TRANSFER=0

# 2. WandB 비활성화
export WANDB_MODE="disabled"
unset WANDB_TOKEN

# 3. 데이터셋 루트 경로
export IMAGES_ROOT="/home/win2dvp21/cmu/LSMA_proj/dataset/LLaVA-CoT-100k"

# 4. 실행
python main.py \
    --num_gpus 10 \
    --images_root $IMAGES_ROOT \
    --dataset_name "Xkev/LLaVA-CoT-100k" \
    --output_dir "outputs_vectors" \
    --output_dir2 "outputs_eval" \
    --num_vector_samples -1 \
    --num_eval_samples -1
