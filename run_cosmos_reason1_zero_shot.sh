#!/bin/bash
API_KEY="token-abc123"
CUSTOM_SERVER="http://localhost:8000/v1"
MAX_CONCURRENT_REQUESTS=1
FPS=4
MODEL_NAME="nvidia/Cosmos-Reason1-7B"
DATASET_PATH="./neuro_paper_data/"
DATASET_NAME="neuro_paper_data"
DATA_NUM=424

temperature=0.6
max_tokens=8192
top_p=0.95
frequency_penalty=0
presence_penalty=0
repetition_penalty=1.05

# "openai" or "vllm"
API_TYPE="vllm"

SETTING_NAME="cosmos_reason1_fps_${FPS}"

OUTPUT_PATH="./outputs/${SETTING_NAME}"

SYSTEM_PROMPT="You are a helpful assistant. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>"

# this will override the user prompt in the test data.
CUSTOM_INSTRUCTION="Detect whether the video plays forward or backward with confidence. \nA: forward, B: backward \n"

python run_fb_task_mm_pre_process.py \
    --api_key $API_KEY \
    --custom_server $CUSTOM_SERVER \
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
    --api_type \
    --temperature $temperature \
    --max_tokens $max_tokens \
    --top_p $top_p \
    --frequency_penalty $frequency_penalty \
    --presence_penalty $presence_penalty \
    --repetition_penalty $repetition_penalty \

