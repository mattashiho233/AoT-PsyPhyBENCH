#!/bin/bash
API_KEY="REPLACE_HERE"
MAX_CONCURRENT_REQUESTS=20
FPS=4

# MODEL_NAME="gpt-4.1-2025-04-14"
MODEL_NAME="gpt-4o-2024-11-20"

DATASET_PATH="./neuro_paper_data/"
DATASET_NAME="neuro_paper_data"
DATA_NUM=424

# some params are model specific like repetition penalty. double check. 
temperature=0.6
max_tokens=8192
top_p=0.95
frequency_penalty=0
presence_penalty=0

# Few-shot data settings
FEW_SHOT_DATA_NAME="few_shot_no_reasoning"
FEW_SHOT_NUM=2
FEW_SHOT_METHOD="first"

# "openai" or "vllm"
API_TYPE="openai"

SETTING_NAME="${MODEL_NAME}_fps${FPS}_${FEW_SHOT_NUM}_shots_${FEW_SHOT_METHOD}_from_${FEW_SHOT_DATA_NAME}"

OUTPUT_PATH="./outputs/${SETTING_NAME}"

SYSTEM_PROMPT="You are a helpful assistant. You will see videos provided from the user, played either forward or backward. Finish your answer with F or B only. F for forward and B for backward. "

# this will override the user prompt in the test data.
CUSTOM_INSTRUCTION="Detect whether the video plays forward or backward with confidence. "

python run_fb_task_mm_pre_process.py \
    --api_key $API_KEY \
    --api_type $API_TYPE \
    --max_concurrent_requests $MAX_CONCURRENT_REQUESTS \
    --model_name $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --dataset_name $DATASET_NAME \
    --data_num $DATA_NUM \
    --setting_name $SETTING_NAME \
    --output_path $OUTPUT_PATH \
    --system_prompt "$SYSTEM_PROMPT" \
    --custom_instruction "$CUSTOM_INSTRUCTION" \
    --fps $FPS \
    --few_shot_data_name $FEW_SHOT_DATA_NAME \
    --few_shot_num $FEW_SHOT_NUM \
    --few_shot_method $FEW_SHOT_METHOD \
    --temperature $temperature \
    --max_tokens $max_tokens \
    --top_p $top_p \
    --frequency_penalty $frequency_penalty \
    --presence_penalty $presence_penalty
