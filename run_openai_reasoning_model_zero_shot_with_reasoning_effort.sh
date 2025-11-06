#!/bin/bash
API_KEY="REPLACE_HERE"
MAX_CONCURRENT_REQUESTS=20
FPS=4

# MODEL_NAME="o3-2025-04-16"
# MODEL_NAME="o4-mini-2025-04-16"
MODEL_NAME="gpt-5-2025-08-07"

DATASET_PATH="./neuro_paper_data/"
DATASET_NAME="neuro_paper_data"
DATA_NUM=424
REASONING_EFFORT="medium"

# some params are model specific.
max_tokens=8192
# temperature=0.6 
# top_p=0.95
# frequency_penalty=0
# presence_penalty=0 


# "openai" or "vllm"
API_TYPE="openai"

SETTING_NAME="${MODEL_NAME}_fps${FPS}_zero_shot_reasoning_effort_${REASONING_EFFORT}"

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
    --reasoning_effort $REASONING_EFFORT \
    --max_output_tokens $max_tokens