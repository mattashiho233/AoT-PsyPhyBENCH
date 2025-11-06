# This code is for v1 of the openai package: pypi.org/project/openai
import json
import os
import random
import asyncio
import openai
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError, BadRequestError
from tqdm.asyncio import tqdm_asyncio
import fire
from qwen_vl_utils import process_vision_info
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import ipdb
from typing import Tuple, List, Dict, Any
import copy


random.seed(42)

# from: https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
# modified: added fps sampling logic
def read_video_frames(video_path, fps=None):
    video = cv2.VideoCapture(video_path)
    base64Frames = []

    # Get original video fps
    orig_fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # If fps is not specified, return all frames
    if fps is None or fps <= 0 or orig_fps == 0:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    else:
        # Sample frames according to the specified fps
        step = int(round(orig_fps / fps))
        if step < 1:
            step = 1
        frame_idx = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            if frame_idx % step == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            frame_idx += 1

    video.release()
    print("original fps:", orig_fps, "sampled fps:", fps, "total frames:", len(base64Frames))
    return base64Frames

# vllm preprocess logic from: https://github.com/vllm-project/vllm/pull/13533
def prepare_message_for_vllm(content_messages):
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                print("video_kwargs", video_kwargs, video_input.shape)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}

# Function to locally preprocess video for either openai or vllm API inference. 
# NOTE: it is NOT this function's job to trim the last assistant message which contains the ground truth label.
def process_video_frames(data, api_type, fps) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if api_type == "openai":
        for v_idx, video_file_path in enumerate(data["videos"]):
            # Find the corresponding user message for this video index
            user_msg_idx = v_idx*2 + 1 if "system" in [msg["role"] for msg in data["messages"]] else v_idx*2

            assert data["messages"][user_msg_idx]["role"] == "user"

            user_message = data["messages"][user_msg_idx]["content"]
            video_frames = read_video_frames(video_file_path, fps=fps)
            # Replace the user message content with OpenAI-compatible format
            data["messages"][user_msg_idx]["content"] = [
                {
                    "type": "text",
                    "text": user_message,
                },
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                    }
                    for frame in video_frames
                ]
            ]
        # TODO: check if not changing assistant message is OK
        return data["messages"], None

    elif api_type == "vllm":
        for v_idx, video_file_path in enumerate(data["videos"]):
            # Find the corresponding user message for this video index
            user_msg_idx = v_idx*2 + 1 if "system" in [msg["role"] for msg in data["messages"]] else v_idx*2

            assert data["messages"][user_msg_idx]["role"] == "user"

            user_message = data["messages"][user_msg_idx]["content"]
            # Replace the user message content with vLLM-compatible format
            data["messages"][user_msg_idx]["content"] = [
                {
                    "type": "text",
                    "text": user_message,
                },
                {
                    "type": "video",
                    "video": f"file://{video_file_path}",
                    "fps": fps,
                },
            ]

        messages_with_video, video_kwargs = prepare_message_for_vllm(data["messages"])
        return messages_with_video, video_kwargs
    else: 
        raise ValueError(f"Unknown API type: {api_type}. Supported types are 'openai' and 'vllm'.")


def to_responses_input(messages):
    """
    Convert Chat Completions-style `messages` to Responses API `input`.
    - system/content (string) -> {"type":"text","text": ...}
    - user/text -> {"type":"input_text","text": ...}
    - user/image_url -> {"type":"input_image","image_url": ...}
    """
    out = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        items = []

        if isinstance(content, str):
            # system or assistant strings
            items.append({"type": "input_text", "text": content})
        else:
            for c in content:
                if c.get("type") == "text":
                    items.append({"type": "input_text", "text": c["text"]})
                elif c.get("type") in ("image_url", "input_image"):
                    # accept either {"image_url": {"url": "..."} } or {"image_url": "..."}
                    img = c["image_url"]
                    url = img["url"] if isinstance(img, dict) else img
                    items.append({"type": "input_image", "image_url": url})
                else:
                    # fall back: treat anything unknown as text
                    if "text" in c:
                        items.append({"type": "input_text", "text": c["text"]})

        out.append({"role": role, "content": items})
    return out

@retry(wait=wait_fixed(30), stop=stop_after_attempt(2), retry=retry_if_exception_type((RateLimitError)))
async def async_call_to_api(client, model_name, data, fps, api_type, output_path, semaphore, progress, **generation_params):
    # video_file_path = data["videos"][-1]
    video_file_name = os.path.basename(data["videos"][-1])
    pass

    async with semaphore: 
        try:
            print("Getting response for example:" , video_file_name)
            gt = data["messages"][-1]["content"]
            data["messages"] = data["messages"][:-1]  # remove the last message which is the ground truth label
            # deep copy the messages for logging
            messages_for_logging = copy.deepcopy(data["messages"])

            if api_type == "openai": 
                messages_with_videos, _ = process_video_frames(data, api_type, fps)
            elif api_type == "vllm":
                messages_with_videos, video_kwargs = process_video_frames(data, api_type, fps)
            # ipdb.set_trace()

            # Print messages without video content (video frames/base64) because it's too large
            def strip_video_content(messages):
                def strip_part(part):
                    if part.get("type") == "image_url":
                        return {"type": "image_url", "image_url": {"url": "<stripped>"}}
                    if part.get("type") == "video_url":
                        return {"type": "video_url", "video_url": {"url": "<stripped>"}}
                    if part.get("type") == "video":
                        return {**part, "video": "<stripped>"}
                    return part
                stripped = []
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        stripped_content = [strip_part(p) for p in msg["content"]]
                        stripped.append({**msg, "content": stripped_content})
                    else:
                        stripped.append(msg)
                return stripped


            extra_body = {"mm_processor_kwargs": video_kwargs} if api_type == "vllm" else {}

            reasoning_tokens = -1
            # for openai reasoning models
            if "reasoning_effort" in generation_params and "gemini" not in model_name.lower():
                print("!!!!!! converting to Responses API because reasoning_effort is set !!!!!!")
                reasoning_effort = generation_params["reasoning_effort"]
                del generation_params["reasoning_effort"]
                responses_input = to_responses_input(messages_with_videos)

                print(to_responses_input(strip_video_content(messages_with_videos)))

                response = await client.responses.create(
                    model=model_name,
                    input=responses_input,
                    reasoning={"effort": reasoning_effort},
                    **generation_params,
                )
                print(response)
                generated = response.output_text
                print("GT:", gt)
                print("Model output:", generated)
                reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
            # for gemini models with reasoning_effort through openai-compatible API
            elif "reasoning_effort" in generation_params and "gemini" in model_name.lower():
                print("!!!!!! using gemini model with reasoning_effort set !!!!!!")
                reasoning_effort = generation_params["reasoning_effort"]
                del generation_params["reasoning_effort"]
                print(strip_video_content(messages_with_videos))
                extra_body["reasoning_effort"] = reasoning_effort
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages_with_videos,
                    extra_body=extra_body,
                    **generation_params
                )
                print(response)
                generated = response.choices[0].message.content
                print("GT:", gt)
                print("Model output:", generated)
                # gemini openai compatible API exposes token usage like this: 
                # usage=CompletionUsage(completion_tokens=111, prompt_tokens=3407, total_tokens=4147)
                completion_tokens = response.usage.completion_tokens
                prompt_tokens = response.usage.prompt_tokens
                total_tokens = response.usage.total_tokens
                
                reasoning_tokens = total_tokens - prompt_tokens - completion_tokens
            # no reasoning effort
            else:
                print(strip_video_content(messages_with_videos))
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages_with_videos,
                    extra_body=extra_body,
                    **generation_params
                )
                print(response)
                generated = response.choices[0].message.content
                print("GT:", gt)
                print("Model output:", generated)

            progress.update(1)

            # save output to a file under output_path, named as video_file_name
            output_file_path = os.path.join(output_path, f"{video_file_name}.jsonl")
            json_to_save = {
                "video_file_name": video_file_name,
                "gt": gt,
                "generated": generated,
                "messages": messages_for_logging,
                "reasoning_tokens": reasoning_tokens,
            }
            with open(output_file_path, "a") as f:
                f.write(json.dumps(json_to_save, ensure_ascii=False) + "\n")

            return json_to_save
        except RateLimitError as e:
            print("Rate Limit Hit...")
            print(f"Error details: {e}")
            raise
        except BadRequestError as e:
            print("Bad Request Error...")
            print(f"Error details: {e}")
            progress.update(1)
            # if "length" in str(e):
                # print("Request Error... Possibly by length.")
            raise

def prepare_dataset(dataset_path, dataset_name, data_num, few_shot_data_name=None, few_shot_num=None, few_shot_method=None, system_prompt=None, api_type=None, fps=None, custom_instruction=None):
    """
    Reads info from dataset json. 

    Actual dataset path should look like this:
    datasets/neuro_paper_data.json
    datasets/neuro_paper_data/00001.mp4

    in this case:
    dataset_path should be datasets/
    dataset_name should be neuro_paper_data
    Video paths are parsed from the json file. 

    Few-shot data should be under the same dataset_path
    """
    json_config = f"{dataset_name}.json"

    with open(os.path.join(dataset_path, json_config)) as f:
        json_data = json.load(f)

    # read few-shot data if provided
    if few_shot_data_name and few_shot_num:
        few_shot_json_config = f"{few_shot_data_name}.json"
        with open(os.path.join(dataset_path, few_shot_json_config)) as f:
            few_shot_json_data = json.load(f)

    dataset = []

    for idx, item in enumerate(json_data[:data_num]):
        assert "messages" in item
        assert len(item["messages"]) == 2
        assert "videos" in item
        assert len(item["videos"]) == 1

        data = item.copy()
        
        video_file_path = dataset_path + data["videos"][0]

        data["videos"][0] = video_file_path # now it stores absolute path

        if custom_instruction:
            # if custom instruction is provided, replace the first user message with it
            data["messages"][0]["content"] = custom_instruction
        else:
            data["messages"][0]["content"] = data["messages"][0]["content"].replace("<video>", "")  # <video> is used in llama_factory. Not needed in online vLLM
        
        # this is copied to replace few-shot user messages. 
        user_message = data["messages"][0]["content"]

        # if a system prompt is provided, prepend it to the messages
        if system_prompt:
            data["messages"] = [{"role": "system", "content": system_prompt}] + data["messages"]

        # boxing few-shots
        if few_shot_data_name and few_shot_num and few_shot_num > 0:
            few_shot_user_messages = []
            few_shot_assistant_messages = []
            few_shot_videos = []

            # sample few-shot data w.r.t. few_shot_num and few_shot_method
            # if total few-shot number in file is bigger than few_shot_num, sample few_shot_num messages based on few_shot_method
            if few_shot_num < len(few_shot_json_data):
                if few_shot_method == "random":
                    few_shot_samples = random.sample(few_shot_json_data, few_shot_num)
                elif few_shot_method == "first":
                    few_shot_samples = few_shot_json_data[:few_shot_num]
                else:
                    raise ValueError(f"Unknown few-shot method: {few_shot_method}. If you are not using all data in the few-shot data, you must use methods 'random' or 'first'.")
            elif few_shot_num == len(few_shot_json_data):
                if not few_shot_method:
                    print(f"Warning: few_shot_num is equal to the total number of few-shot samples ({len(few_shot_json_data)}) and few_shot_method not set. Using 'first' method (all of the data, in order) by default.")
                    few_shot_samples = few_shot_json_data
                elif few_shot_method == "first":
                    few_shot_samples = few_shot_json_data[:few_shot_num]
                elif few_shot_method == "random":
                    random.shuffle(few_shot_json_data)
                    few_shot_samples = few_shot_json_data
            else:
                raise ValueError(f"Few-shot number {few_shot_num} is bigger than the total number of few-shot samples {len(few_shot_json_data)}. Please adjust few_shot_num.")
            
            for i in range(few_shot_num):
                few_shot_data = few_shot_samples[i]
                assert "messages" in few_shot_data
                assert len(few_shot_data["messages"]) == 2
                assert "videos" in few_shot_data
                assert len(few_shot_data["videos"]) == 1

                # user message in few-shot data does not matter. using either custom_instruction or the first user message
                few_shot_user_messages.append(user_message)
                few_shot_assistant_messages.append(few_shot_data["messages"][1]["content"])  # assistant message is the second message
                # video file path is the first video in few-shot data
                few_shot_videos.append(os.path.join(dataset_path, few_shot_data["videos"][0]))

        # assemble few-shots
        if few_shot_num and few_shot_num > 0:
            assert len(few_shot_user_messages) == few_shot_num
            assert len(few_shot_assistant_messages) == few_shot_num
            assert len(few_shot_videos) == few_shot_num

            few_shot_messages = []
            for i in range(few_shot_num):
                few_shot_messages.append({
                    "role": "user",
                    "content": few_shot_user_messages[i]
                })
                few_shot_messages.append({
                    "role": "assistant",
                    "content": few_shot_assistant_messages[i]
                })
            # add few-shot messages and videos to the data
            # insert few_shot messages after the system prompt if it exists
            if system_prompt:
                data["messages"] = data["messages"][:1] + few_shot_messages + data["messages"][1:]
            else:
                data["messages"] = few_shot_messages + data["messages"]
            # add few-shot videos to the data
            data["videos"] = few_shot_videos + data["videos"]
        
        # ipdb.set_trace()
        dataset.append(data)

    return dataset

async def run(api_key,
              max_concurrent_requests,
              model_name,
              dataset_path,
              dataset_name,
              data_num,
              setting_name,
              output_path,
              fps,
              custom_server="",
              few_shot_data_name=None,
              few_shot_num=None,
              few_shot_method=None,
              api_type=False,
              system_prompt=None,
              custom_instruction=None,
              **generation_params):
    # prepare environment
    if not custom_server:
        print("Using default OpenAI server...")
        client = openai.AsyncOpenAI(api_key=api_key if api_key else os.environ["OPENAI_API_KEY"])
    else:
        client = openai.AsyncOpenAI(
            base_url=custom_server,
            api_key=api_key
        )

    # make dir if not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # prepare few-shots and inputs
    print("Preparing data...")
    # TODO: add few-shot support
    dataset = prepare_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        data_num=data_num,
        system_prompt=system_prompt,
        custom_instruction=custom_instruction,
        few_shot_data_name=few_shot_data_name,
        few_shot_num=few_shot_num,
        few_shot_method=few_shot_method,
        api_type=api_type,
        fps=fps,
    )

    semaphore = asyncio.Semaphore(max_concurrent_requests)
    progress = tqdm_asyncio(total=len(dataset))

    print("Generating async tasks...")
    # TODO: few shot
    tasks = [asyncio.create_task(async_call_to_api(client=client,
                                                    model_name=model_name,
                                                    data=item,
                                                    fps=fps,
                                                    api_type=api_type,
                                                    output_path=output_path,
                                                    semaphore=semaphore,
                                                    progress=progress,
                                                    **generation_params))
            for item in dataset]
    
    print("Firing tasks...")
    response_data = await asyncio.gather(*tasks, return_exceptions=True)
    
    # save examples with no errors to jsonl
    # if other error occures during tasks, this will be empty, but we have backup in real-time log file
    good_responses = []
    error_count = 0
    for response in response_data:
        if isinstance(response, Exception):
            error_count += 1
            print(response)
            continue
        good_responses.append(response)

    with open(os.path.join(output_path, f"{setting_name}.jsonl"), "w", encoding="utf-8") as f:
        for item in good_responses:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("total examples:", len(dataset))
    print("error count:", error_count)


def main(**kwargs):
    asyncio.run(run(**kwargs))

if __name__ == "__main__":
    fire.Fire(main)