import multiprocessing
import base64
import time
import os
import requests
from tqdm import tqdm
import json
import re
import math
import traceback
import sys
import argparse
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
from PIL import Image
import torch
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Model Configuration Parameters")

    parser.add_argument(
        "--video_folder",
        type=str,
        default="/path/to/V-STaR/videos/",
        help="Path to video folder",
    )
    parser.add_argument(
        "--anno_file",
        type=str,
        default="/psth/to/V-STaR/V_STaR_test.json",
        help="Path to annotation file",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="/path/to/your/save_path.json",
        help="Path to result file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/path/to/your/model",
        help="Path to model",
    )
    parser.add_argument(
        "--think_mode",
        action="store_true",
        help="use think mode or not",
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default=None,
        help="Path to YAML file containing model keyword arguments.",
    )

    args = parser.parse_args()

    try:
        import yaml
        with open(args.model_kwargs, "r") as f:
            args.model_kwargs = yaml.safe_load(f)
        if not isinstance(args.model_kwargs, dict):
            raise ValueError("YAML file must contain a dictionary")
    except ImportError:
        parser.error(
            "PyYAML is required for YAML parsing. Install with: pip install pyyaml"
        )
    except FileNotFoundError:
        parser.error(f"Model kwargs file not found: {args.model_kwargs}")
    except Exception as e:
        parser.error(f"Error parsing YAML file: {e}")
    return args


def get_cuda_visible_devices():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible_devices:
        return []
    gpu_list = [
        int(gpu_id.strip())
        for gpu_id in cuda_visible_devices.split(",")
        if gpu_id.strip()
    ]
    return gpu_list

def extract_frames_from_video(video_path, fps=1.0, max_frames=16):
    """
    Extract frames from video at specified fps with a maximum frame limit.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    frame_interval = int(video_fps / fps)

    expected_frames = int(duration * fps)
    if expected_frames > max_frames:
        # If too many frames, sample evenly across the video
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        # Sample at regular intervals
        frame_indices = list(range(0, total_frames, frame_interval))
        if len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]

    frames = []
    frame_times = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)
        time_sec = idx / video_fps
        frame_times.append(time_sec)

    cap.release()
    return frames, frame_times

def extract_timestamps(result):
    """extract timestamps from the answer"""
    match = re.search(r"<answer>(.*?)</answer>", result, re.DOTALL)
    if match:
        result = match.group(1).strip()

    time_stamps = re.findall(r'(\d+:\d+)', result)
    for ts in time_stamps:
        minutes, seconds = map(int, ts.split(':'))
        seconds = minutes*60 + seconds
        result = result.replace(ts, f'<t>{seconds}</t>s')

    match = re.findall(r"\b\d+(?:\.\d+)?\b", result)
    return [float(match[0]), float(match[1])] if len(match) == 2 else []


def fix_incomplete_json(json_str):
    """
    fix the incomplete brackets of the json
    """
    # Counting left and right brackets
    open_square = json_str.count("[")
    close_square = json_str.count("]")
    open_curly = json_str.count("{")
    close_curly = json_str.count("}")

    # Complete the square brackets
    if open_square > close_square:
        json_str += "]" * (open_square - close_square)
    elif close_square > open_square:
        json_str = "[" * (close_square - open_square) + json_str

    # Complete the curly brackets
    if open_curly > close_curly:
        json_str += "}" * (open_curly - close_curly)
    elif close_curly > open_curly:
        json_str = "{" * (close_curly - open_curly) + json_str

    return json_str

    
def create_frame_prompt(frame_times):
    """
    Create the frame prompt string similar to training format.
    """
    frame_prompt = ""
    for i, time_sec in enumerate(frame_times):
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)
        time_str = f"{minutes}:{seconds:02d}"
        frame_prompt += f"Frame {i + 1} at {round(time_sec, 1)}s: <|vision_start|><|image_pad|><|vision_end|>\n"
    return frame_prompt


def inference(video_path, prompt, model, fps=1.0):
    max_frames = model.video_max_frames
    # Extract frames from video
    frames, frame_times = extract_frames_from_video(
        video_path, fps=fps, max_frames=max_frames
    )

    # print("frames", len(frames))

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    # Create frame prompt
    frame_prompt = create_frame_prompt(frame_times)

    if "<|vision_start|><|video_pad|><|vision_end|>" in prompt:
        prompt = prompt.replace(
            "<|vision_start|><|video_pad|><|vision_end|>", frame_prompt
        )
    else:
        prompt = frame_prompt + prompt

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
            + [{"type": "image", "image": frame} for frame in frames],
        },
    ]

    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    def remove_vision_tags(text):
        start_marker = "the question about the video"
        end_marker = "<|im_start|>assistant"
        tag_to_remove = "<|vision_start|><|image_pad|><|vision_end|>"

        start_idx = text.find(start_marker)
        if start_idx == -1:
            return text
        end_idx = text.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            return text
        middle_part = text[start_idx + len(start_marker) : end_idx]
        cleaned_middle = middle_part.replace(tag_to_remove, "")

        result = text[: start_idx + len(start_marker)] + cleaned_middle + text[end_idx:]

        return result

    text = remove_vision_tags(text)

    # print("text", text)
    # print("Processing vision info...")
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages], return_video_kwargs=True
    )

    # print(f"Extracted {len(frames)} frames from video")

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    llm_inputs = [{
        "prompt": text,
        "multi_modal_data": mm_data,
    }]
    generated_text = model.inference_wo_process(llm_inputs)
    # print("Full interaction:")
    # print(text + generated_text)

    return generated_text, len(frames), (frames[0].size if frames else (0, 0))


def read_anno(anno_file):
    with open(anno_file, "r") as f:
        data = json.load(f)
    return data


def find_video(video_folder, vid):
    """
    Finds the vid.mp4 file in the video_folder and its subfolders.
    """
    target_filename = f"{vid}.mp4"
    for root, _, files in os.walk(video_folder):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None


def get_answer_vqa(data, video_path, model, think_mode=True):
    prompt = f"Answer the question about the video: {data['question']} \n (If the answer is a person, you don't need to identify the person.)"
    
    if think_mode:
        prompt += "You must first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. Do not use <box>, <obj> and <t> in the answer part."

    answer_vqa, num_frames, frame_size = inference(video_path, prompt, model)
    return answer_vqa


def get_answer_temporal(data, video_path, model, think_mode=True):
    video_length = round(data["frame_count"] / data["fps"], 2)
    temporal_question = data["temporal_question"]

    # Create a prompt that includes frame information placeholder
    prompt = f"This video is {video_length} seconds long. <|vision_start|><|video_pad|><|vision_end|>\nAnswer the question about the video: {temporal_question} \nDirectly output the start and end moment timestamps. You must follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'."

    if think_mode:
        prompt = f"This video is {video_length} seconds long. <|vision_start|><|video_pad|><|vision_end|>\nAnswer the question about the video: {temporal_question} \n. You must first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. The answer must follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'"

    answer_temporal, num_frames, frame_size = inference(video_path, prompt, model)
    return answer_temporal


def get_answer_temporal_2(data, video_path, bboxes, model, think_mode=True):
    video_length = round(data["frame_count"] / data["fps"], 2)
    temporal_question = data["temporal_question"]
    w, h = data["width"], data["height"]

    prompt = f"This video is {video_length} seconds long with a resolution of {w}x{h} (width x height). <|vision_start|><|video_pad|><|vision_end|>\nAnswer the question about the video: {temporal_question} \nThere are {len(bboxes)} bounding boxes of the key object related to the question in the video without knowing the time, which are:{bboxes}. Directly output the start and end moment timestamps. You must follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'."

    if think_mode:
        prompt = f"This video is {video_length} seconds long with a resolution of {w}x{h} (width x height). <|vision_start|><|video_pad|><|vision_end|>\nAnswer the question about the video: {temporal_question} \nThere are {len(bboxes)} bounding boxes of the key object related to the question in the video without knowing the time, which are:{bboxes}. You must first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. The answer must follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'"
    
    answer_temporal, num_frames, frame_size = inference(video_path, prompt, model)
    return answer_temporal


def get_answer_spatial(data, video_path, model, think_mode=True):
    video_length = round(data["frame_count"] / data["fps"], 2)
    st, et = math.ceil(data["timestamps"][0]), math.floor(data["timestamps"][1])
    time_range = list(range(st, et + 1))
    w, h = data["width"], data["height"]
    spatial_question = data["spatial_question"]

    prompt = f"""<|vision_start|><|video_pad|><|vision_end|>
Please answer the question about the video: {spatial_question} with a series of bounding boxes in [x1, y1, x2, y2] format.
For each whole second within the time range {time_range} provided (inclusive of the boundaries), output a series of bounding boxes of the object in JSON format. The keys should be the whole seconds (as strings), and the values should be the box in [x1, y1, x2, y2] format.
Example output: {{"{time_range[0]}": [x1, y1, x2, y2],...}}
"""
    if think_mode:
        prompt = f"""<|vision_start|><|video_pad|><|vision_end|>
Please answer the question about the video: {spatial_question} with a series of bounding boxes in [x1, y1, x2, y2] format.
You must first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
In the answer part, for each whole second within the time range {time_range} provided (inclusive of the boundaries), output a series of bounding boxes of the object in JSON format. The keys should be the whole seconds (as strings), and the values should be the box in [x1, y1, x2, y2] format.
Example output of the answer part: {{"{time_range[0]}": [x1, y1, x2, y2],...}}
"""

    answer_spatial, num_frames, frame_size = inference(video_path, prompt, model)
    return answer_spatial, frame_size[0], frame_size[1]  # width, height


def get_answer_spatial_2(data, video_path, bboxes, model, think_mode=True):
    video_length = round(data["frame_count"] / data["fps"], 2)
    st, et = math.ceil(data["timestamps"][0]), math.floor(data["timestamps"][1])
    time_range = list(range(st, et + 1))
    w, h = data["width"], data["height"]
    spatial_question = data["spatial_question_2"]

    prompt = f"""<|vision_start|><|video_pad|><|vision_end|>
Please answer the question about the video: {spatial_question} with a series of bounding boxes in [x1, y1, x2, y2] format.
For each whole second that may related to the question, output a series of bounding boxes of the object in JSON format. You only need to output {len(bboxes)} bbox(es). You need to determine which frame is related to the question, and you don't need to output the bbox for the frames not related to the question.
The keys should be the whole seconds (as strings), and the values should be the bounding box in [x0,y0,x1,y1] format.

Example output:
{{"0": [x0,y0,x1,y1], "1":..., ..., "{len(bboxes)}":...}} (if the frames at 0~{len(bboxes)} second are related to the questions)
"""
    if think_mode:
        prompt = f"""<|vision_start|><|video_pad|><|vision_end|>
Please answer the question about the video: {spatial_question} with a series of bounding boxes in [x1, y1, x2, y2] format.
You must first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
In the answer part, for each whole second that may related to the question, output a series of bounding boxes of the object in JSON format. You only need to output {len(bboxes)} bbox(es). You need to determine which frame is related to the question, and you don't need to output the bbox for the frames not related to the question.
The keys should be the whole seconds (as strings), and the values should be the bounding box in [x0,y0,x1,y1] format.

Example output of the answer part:
{{"0": [x0,y0,x1,y1], "1":..., ..., "{len(bboxes)}":...}} (if the frames at 0~{len(bboxes)} second are related to the questions)
"""

    answer_spatial, num_frames, frame_size = inference(video_path, prompt, model)
    return answer_spatial, frame_size[0], frame_size[1]  # width, height


def extract_bounding_boxes(answer_spatial, data, input_width, input_height):
    """
    Extract bounding boxes from the input answer_spatial and denormalize the coordinates using the width and height from the data.
    """
    match = re.search(r"<answer>(.*?)</answer>", answer_spatial, re.DOTALL)
    if match:
        answer_spatial = match.group(1).strip()
 
    w, h = data["width"], data["height"]

    def denormalize_bbox(bbox):
        """
        denormalize the coordinates of bbox
        """
        try:
            if len(bbox) == 1:
                bbox = bbox[0]
            if len(bbox) == 2:
                bbox = bbox[1]
            x_min = int(bbox[0] / input_width * w)
            y_min = int(bbox[1] / input_height * h)
            x_max = int(bbox[2] / input_width * w)
            y_max = int(bbox[3] / input_height * h)
            return [x_min, y_min, x_max, y_max]
        except Exception as e:
            print(f"Processing {bbox} occurs Error {e}")
            return bbox

    # match markdown json
    markdown_pattern = r"```json\s*\n(\[.*?\]|\{.*?\})\s*\n```"
    match = re.search(markdown_pattern, answer_spatial, re.DOTALL)
    if not match:
        # If there is no Markdown wrapper, then try to match the JSON format directly
        json_pattern = r"(\[[\s\S]*\]|\{[\s\S]*\})"
        match = re.search(json_pattern, answer_spatial, re.DOTALL)
    if match:
        # match bbox in JSON
        bounding_boxes_str = match.group(1).strip()
        # Replace single quotes with double quotes to conform to the JSON specification
        bounding_boxes_str = bounding_boxes_str.replace("'", '"')
        try:
            # Convert strings to dictionary or list format
            bounding_boxes = json.loads(bounding_boxes_str)
            # If it's a list and contains a dictionary inside, expand it to a single dictionary
            if isinstance(bounding_boxes, list) and all(
                isinstance(item, dict) for item in bounding_boxes
            ):
                combined_dict = {}
                for item in bounding_boxes:
                    combined_dict.update(item)
                bounding_boxes = combined_dict
                # Determine if the extracted JSON is a dictionary or a list.
            if isinstance(bounding_boxes, list):
                # bounding boxes in list
                return {str(box[0]): box[1] for box in bounding_boxes}
            elif isinstance(bounding_boxes, dict):
                # bounding boxes in dictionary
                return {key: value for key, value in bounding_boxes.items()}
        except Exception as e:
            # if failed, try to fix it.
            fixed_bounding_boxes_str = fix_incomplete_json(bounding_boxes_str)
            try:
                bounding_boxes = json.loads(fixed_bounding_boxes_str)
                if isinstance(bounding_boxes, list):
                    return [box for box in bounding_boxes]
                elif isinstance(bounding_boxes, dict):
                    return {key: value for key, value in bounding_boxes.items()}
            except Exception as e:
                print(
                    f"Failed after fixing: {e}\nExtracted JSON: {fixed_bounding_boxes_str}"
                )
                return None
    else:
        print("No match found for the bounding box JSON.")
        return None


def build_model(
    model_path,
    temperature,
    max_tokens,
    video_max_pixels,
    video_max_frames,
):
    from models.model_vllm import QwenVL_VLLM

    model = QwenVL_VLLM(
        model_path,
        rt_shape=True,
        temperature=temperature,
        max_tokens=max_tokens,
        video_max_pixels=video_max_pixels,
        video_max_frames=video_max_frames,
    )

    return model


def worker(
    gpu_id,
    data_chunk,
    results_list,
    error_queue,
    args
):
    """
    The worker function for each process. It processes a chunk of data on a specific GPU.
    """
    # Step 1: Assign process to a specific GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        model = build_model(args.model_path, **args.model_kwargs)
        video_folder = args.video_folder
        desc = f"GPU {gpu_id}"
        for data in tqdm(data_chunk, desc=desc, position=gpu_id):
            if not error_queue.empty():  # Check if another process has failed
                break
            try:
                vid = data["vid"]
                video_path = find_video(video_folder, vid)
                if not video_path:
                    print(
                        f"Warning: Video for vid '{vid}' not found on GPU {gpu_id}. Skipping."
                    )
                    continue

                boxes = [
                    [
                        box_data["xmin"],
                        box_data["ymin"],
                        box_data["xmax"],
                        box_data["ymax"],
                    ]
                    for box_data in data["bboxes"]
                ]

                answer_vqa = get_answer_vqa(data, video_path, model, args.think_mode)

                data["answer_vqa_raw_output"] = answer_vqa

                if args.think_mode:
                    match = re.search(r"<answer>(.*?)</answer>", answer_vqa, re.DOTALL)
                    if match:
                        answer_vqa = match.group(1).strip()

                # chain one
                answer_temporal = get_answer_temporal(
                    data, video_path, model, args.think_mode
                )
                answer_temporal_post = extract_timestamps(answer_temporal)

                answer_spatial, input_width, input_height = get_answer_spatial(
                    data, video_path, model, args.think_mode
                )
                answer_spatial_post = extract_bounding_boxes(
                    answer_spatial, data, input_width, input_height
                )

                # chain two
                answer_spatial_2, input_width, input_height = get_answer_spatial_2(
                    data, video_path, boxes, model, args.think_mode
                )
                answer_spatial_post_2 = extract_bounding_boxes(
                    answer_spatial_2, data, input_width, input_height
                )

                answer_temporal_2 = get_answer_temporal_2(
                    data, video_path, boxes, model, args.think_mode
                )
                answer_temporal_post_2 = extract_timestamps(answer_temporal_2)

                # update data (original_index is preserved)
                data["answer_vqa"] = answer_vqa
                data["answer_temporal_pre"] = answer_temporal
                data["answer_temporal"] = answer_temporal_post
                data["answer_spatial_pre"] = answer_spatial
                data["answer_spatial"] = answer_spatial_post

                data["answer_spatial_pre_2"] = answer_spatial_2
                data["answer_spatial_2"] = answer_spatial_post_2
                data["answer_temporal_pre_2"] = answer_temporal_2
                data["answer_temporal_2"] = answer_temporal_post_2

                data["input_shape"] = (input_width, input_height)

                results_list.append(data)

            except Exception as e:
                # Capture full traceback and put it in the error queue
                error_info = f"ERROR processing item with original_index {data.get('original_index', 'N/A')} on GPU {gpu_id}:\n{traceback.format_exc()}"
                error_queue.put(error_info)
                break

    except Exception as e:
        # Capture fatal initialization errors
        error_info = f"FATAL ERROR on GPU {gpu_id} during initialization:\n{traceback.format_exc()}"
        error_queue.put(error_info)



def main():
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Parse command line arguments
    print("Start Time:", datetime.now())
    args = parse_args()
    
    num_gpus = int(os.getenv("NUM_GPUS"))
    gpu_list = get_cuda_visible_devices()

    if num_gpus != len(gpu_list):
        raise ValueError("Number of GPUs does not match the number of CUDA visible devices.")

    # Extract parameters
    anno_file = args.anno_file
    result_file = args.result_file

    print("use think:", args.think_mode)


    # Load annotation data
    anno = read_anno(anno_file)
    print(f"Loaded {len(anno)} samples from {anno_file}")

    # 1. Add a unique, sequential index to each item for robust reordering later.
    for i, item in enumerate(anno):
        item["original_index"] = i

    import random
    random.shuffle(anno)

    num_items = len(anno)
    items_per_gpu = num_items // num_gpus

    # Split the data into chunks for each process
    data_chunks = []
    for i in range(num_gpus):
        start_idx = i * items_per_gpu
        end_idx = (i + 1) * items_per_gpu
        if i == num_gpus - 1:  # Give the last GPU all remaining items
            end_idx = num_items
        data_chunks.append(anno[start_idx:end_idx])

    # Use a Manager to share data between processes
    with multiprocessing.Manager() as manager:
        results_list = manager.list()  # For collecting results
        error_queue = manager.Queue()  # For collecting detailed error messages
        processes = []

        print(f"Starting {num_gpus} processes to process {num_items} items...")

        for i in range(num_gpus):
            if not data_chunks[i]:  # Don't start a process for an empty chunk
                continue
            p = multiprocessing.Process(
                target=worker,
                args=(
                    gpu_list[i],
                    data_chunks[i],
                    results_list,
                    error_queue,
                    args
                ),
            )
            processes.append(p)
            p.start()

        while any(p.is_alive() for p in processes):
            if not error_queue.empty():
                error_message = error_queue.get()
                print("\n" + "=" * 80)
                print(
                    "An error was detected in a worker process. Terminating all processes..."
                )
                print("ERROR DETAILS:")
                print(error_message)
                print("=" * 80 + "\n")

                for p in processes:
                    if p.is_alive():
                        p.terminate()
                break
            time.sleep(1) 

        for p in processes:
            p.join()

        if not error_queue.empty():
            if "error_message" not in locals(): 
                error_message = error_queue.get()
                print("\n" + "=" * 80)
                print(
                    "Program terminated due to an unrecoverable error in a child process."
                )
                print("ERROR DETAILS:")
                print(error_message)
                print("=" * 80 + "\n")
            sys.exit(1)

        print(
            "\nAll processes finished successfully. Aggregating and saving results..."
        )

        # Convert manager list to a regular list
        final_results_list = list(results_list)
        results_map = {item["original_index"]: item for item in final_results_list}
        ordered_anno = []
        for original_item in anno:
            ordered_anno.append(
                results_map.get(original_item["original_index"], original_item)
            )
        with open(result_file, "w") as f:
            json.dump(ordered_anno, f, indent=4)

        print(
            f"Processed {len(final_results_list)} / {num_items} items. Results successfully saved to {result_file}"
        )

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()