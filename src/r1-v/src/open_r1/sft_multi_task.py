import os
import time
import logging
os.environ["WANDB_MODE"] = "offline" 

import os
from configs.data_root import DATA_ROOT

ROOT = os.path.join(DATA_ROOT, "videos")
TREEVGR_ROOT = os.path.join(ROOT, "treevgr")
TVG_ROOT = os.path.join(ROOT, "tvg_r1")
STR_KF_ROOT = os.path.join(ROOT, "stgr/temporal_grounding/kfs")
STR_DATA = os.path.join(ROOT, "stgr/temporal_grounding/videos")
STR_PLM_KF_ROOT = os.path.join(ROOT, "stgr/plm/kfs")
STR_PLM_DATA = os.path.join(ROOT, "stgr/plm/videos")
GENERAL_VIDEO_ROOT = os.path.join(ROOT, "videor1")

### for debug ###
# os.environ["MASTER_PORT"] = "29501"
# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["LOCAL_RANK"] = "0"

import os
import json
import random
import requests
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from src.open_r1.vision_process import process_vision_info

from datasets import Dataset, DatasetDict, Features, Sequence, Value

import wandb
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import copy

# ============== CONFIGURATION ==============
logger = logging.getLogger(__name__)

# Performance options (disabled by default for reproducibility)
CUDNN_BENCHMARK = os.environ.get("CUDNN_BENCHMARK", "0").lower() in {"1", "true", "yes"}
ENABLE_TF32 = os.environ.get("ENABLE_TF32", "0").lower() in {"1", "true", "yes"}
USE_LINEAR_PATCH_EMBED = os.environ.get("USE_LINEAR_PATCH_EMBED", "0").lower() in {"1", "true", "yes"}


def _configure_logging() -> None:
    """Configure logging format for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def _configure_torch_backends() -> None:
    """Configure PyTorch backends based on environment variables."""
    import torch.backends.cuda
    import torch.backends.cudnn

    if CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cudnn.benchmark")

    if ENABLE_TF32:
        # PyTorch 2.9+ uses new precision API
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        else:
            # Fallback for PyTorch < 2.9
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 for matmul and cudnn")


class LinearPatchEmbed(torch.nn.Module):
    """
    Optimized patch embedding using Linear instead of Conv3d.

    When kernel_size == stride and input spatial dims == kernel dims,
    Conv3d is mathematically equivalent to Linear but Linear uses
    highly optimized cuBLAS instead of potentially slow cuDNN Conv3d paths.

    This provides 40-100x speedup for Qwen3-VL patch embedding.
    """

    def __init__(self, conv3d_module: torch.nn.Module):
        super().__init__()
        self.in_channels = conv3d_module.in_channels
        self.temporal_patch_size = conv3d_module.kernel_size[0]
        self.patch_size = conv3d_module.kernel_size[1]
        self.embed_dim = conv3d_module.out_channels

        in_features = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        has_bias = conv3d_module.bias is not None

        self.proj = torch.nn.Linear(in_features, self.embed_dim, bias=has_bias)

        # Copy weights from Conv3d (reshape from [out, in, t, h, w] to [out, in*t*h*w])
        with torch.no_grad():
            self.proj.weight.copy_(conv3d_module.weight.view(self.embed_dim, -1))
            if has_bias:
                self.proj.bias.copy_(conv3d_module.bias)

        self.proj = self.proj.to(device=conv3d_module.weight.device, dtype=conv3d_module.weight.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
        return hidden_states


def _replace_conv3d_with_linear_patch_embed(model: torch.nn.Module) -> bool:
    """
    Replace Conv3d patch embedding with LinearPatchEmbed in Qwen3-VL models.

    Returns True if replacement was made, False otherwise.
    """
    if not USE_LINEAR_PATCH_EMBED:
        return False

    # Navigate to the patch_embed module
    visual_model = None
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        visual_model = model.model.visual
    elif hasattr(model, "visual"):
        visual_model = model.visual

    if visual_model is None:
        logger.warning("Could not find visual model for LinearPatchEmbed replacement")
        return False

    if not hasattr(visual_model, "patch_embed"):
        logger.warning("Visual model has no patch_embed attribute")
        return False

    patch_embed = visual_model.patch_embed

    # Check if it's a Conv3d
    if not isinstance(patch_embed, torch.nn.Conv3d):
        logger.info(f"patch_embed is {type(patch_embed).__name__}, not Conv3d - skipping replacement")
        return False

    # Verify kernel_size == stride (required for equivalence)
    if patch_embed.kernel_size != patch_embed.stride:
        logger.warning(f"Conv3d kernel_size {patch_embed.kernel_size} != stride {patch_embed.stride}, cannot replace")
        return False

    # Create and install the Linear replacement
    linear_patch_embed = LinearPatchEmbed(patch_embed)
    visual_model.patch_embed = linear_patch_embed

    logger.info(
        f"Replaced Conv3d patch_embed with LinearPatchEmbed "
        f"(in_features={linear_patch_embed.proj.in_features}, "
        f"out_features={linear_patch_embed.proj.out_features})"
    )
    return True


_configure_logging()


def ensure_media_types(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if not item.get("type"):
                    if item.get("image") or item.get("image_url"):
                        item["type"] = "image"
                    elif item.get("video"):
                        item["type"] = "video"
    return messages

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""

    if example['task'] == 'visual QA':
        system_message = "A conversation between user and assistant. The user provides an image and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant MUST localize the object with bounding box coordinates between <box> and </box>. You MUST strictly follow the format."
        image_root = TREEVGR_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(image_root,example['image_path'])
                    },
                    {
                        "type": "text",
                        "text": question,
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        messages = ensure_media_types(messages)
        return {"messages": messages, "image_size": example["image_size"], "task": "visual QA", "source": example["source"], "key_frames":[]}
    
    elif example['task'] == 'temporal-spatial free-form QA':
        system_message = "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`."
        question = example['question']
        video_root = STR_DATA
        if example['source'] == "STR_plm_rdcap":
            video_root = STR_PLM_DATA
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]    
        messages = ensure_media_types(messages)
        return {"messages": messages, "key_frames": example["key_frames"], "task": "temporal-spatial free-form QA", "source": example["source"], "image_size":[]}
    
    elif example['task'] == 'temporal QA':
        system_message = "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant determines the precise time period that answers the question. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'"
        video_root = TVG_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": "Question: " + question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        messages = ensure_media_types(messages)
        return {"messages": messages, "task": "temporal QA", "source": example["source"], "key_frames":[], "image_size":[]}
    elif example["task"] == "General video QA MCQ":
        system_message = "A conversation between user and assistant. The user provides a video and asks a multiple-choice question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Only output the correct option in the <answer> </answer> section."
        video_root = GENERAL_VIDEO_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": "Question: " + question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        messages = ensure_media_types(messages)
        return {"messages": messages, "task": "General video QA MCQ", "source": example["source"], "key_frames":[], "image_size":[]}
    elif example["task"] == "General video QA Free-form":
        system_message = "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
        video_root = GENERAL_VIDEO_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": "Question: " + question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        messages = ensure_media_types(messages)
        return {"messages": messages, "task": "General video QA Free-form", "source": example["source"], "key_frames":[], "image_size":[]}
    
    raise ValueError(f"Unknown task: {example['task']}")


def convert_coord_format_espressso(bbox, image_size):
    # for videoespresso
    # image_size: (W, H)
    nx, ny, nw, nh = [coord / 1000.0 for coord in bbox]
    x_center = nx * image_size[0]
    y_center = ny * image_size[1]
    width = nw * image_size[0]
    height = nh * image_size[1]

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_size[0], x_max)
    y_max = min(image_size[1], y_max)

    return [x_min, y_min, x_max, y_max]

def convert_coord_format_gemini(coords, image_size):
    # for gemini annotated data
    norm_x_min, norm_y_min, norm_x_max, norm_y_max = coords
    width, height = image_size
    real_x_min = norm_x_min * width
    real_y_min = norm_y_min * height
    real_x_max = norm_x_max * width
    real_y_max = norm_y_max * height
    return [real_x_min, real_y_min, real_x_max, real_y_max]

import re
def resize_bounding_boxes_for_image(text: str, old_image_size: tuple, new_image_size: tuple) -> str:

    old_w, old_h = old_image_size
    new_w, new_h = new_image_size
    ratios = (new_w / old_w, new_h / old_h, new_w / old_w, new_h / old_h)

    def resizer(match: re.Match) -> str:
        coords = [int(c) for c in match.group(1).strip('[]').split(',')]
        new_coords = [int(round(c * r)) for c, r in zip(coords, ratios)]
        return f"<box>[{','.join(map(str, new_coords))}]</box>"

    return re.sub(r"<box>(\[.*?\])</box>", resizer, text)

def replace_boxes_for_videoespresso(text, image_size):
    import re
    pattern = re.compile(r'<box>\[([^]]+)\]</box>')
    
    def replacer(match):
        box_str = match.group(1)
        coords = list(map(float, box_str.split(',')))
        new_coords = convert_coord_format_espresso(coords, image_size)
        new_coords = str([round(coord) for coord in new_coords])
        new_coords = new_coords.replace(" ","")
        return '<box>' + new_coords + '</box>'
    
    return pattern.sub(replacer, text)


def replace_boxes_for_gemini_data(text, image_size):
    import re
    pattern = re.compile(r'<box>\[([^]]+)\]</box>')
    
    def replacer(match):
        box_str = match.group(1)
        coords = list(map(float, box_str.split(',')))
        new_coords = convert_coord_format_gemini(coords, image_size)
        new_coords = str([round(coord) for coord in new_coords])
        new_coords = new_coords.replace(" ","")
        return '<box>' + new_coords + '</box>'
    
    return pattern.sub(replacer, text)

def _get_media_paths(messages: List[Dict[str, Any]]) -> List[str]:
    paths = []
    for msg in messages:
        for item in msg.get("content", []):
            if item.get("type") == "image" and "image" in item:
                paths.append(item["image"])
            if item.get("type") == "video" and "video" in item:
                paths.append(item["video"])
    return paths

def _count_media_tokens(text: str) -> Dict[str, int]:
    return {
        "image": text.count("<image>"),
        "video": text.count("<video>"),
    }

def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(messages)
    for message in cleaned:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            for key in list(item.keys()):
                if item[key] is None:
                    del item[key]
    return cleaned

def _inject_media_tokens(messages: List[Dict[str, Any]], image_tokens: int, video_tokens: int) -> List[Dict[str, Any]]:
    injected = copy.deepcopy(messages)
    for message in injected:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        tokens = []
        if image_tokens > 0:
            tokens.append({"type": "text", "text": "<image>"})
        if video_tokens > 0:
            tokens.append({"type": "text", "text": "<video>"})
        if tokens:
            message["content"] = tokens + content
        break
    return injected

def _ensure_video_pad(text: str) -> str:
    pad = "<|vision_start|><|video_pad|><|vision_end|>"
    if pad in text:
        return text
    return f"{pad}\n{text}"

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    debug_pipeline = os.getenv("DEBUG_DATA_PIPELINE") == "1"
    if debug_pipeline and not hasattr(collate_fn, "_debug_counter"):
        collate_fn._debug_counter = 0
    texts = []
    batch_images = []
    example_times = []

    batch_start = time.perf_counter() if debug_pipeline else None
    for i, example in enumerate(examples):
        example_start = time.perf_counter() if debug_pipeline else None
        try:
            sanitized_messages = _sanitize_messages(example["messages"])
            text = processor.apply_chat_template(sanitized_messages, tokenize=False)
            image_inputs, video_inputs, video_kwargs = process_vision_info(sanitized_messages, return_video_kwargs=True)
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

        if example["task"] == "visual QA":
            old_image_size = example["image_size"]
            new_image_size = [image_inputs[0].size[0], image_inputs[0].size[1]]  # W * H
            text = resize_bounding_boxes_for_image(text, old_image_size, new_image_size)

            media_tokens = _count_media_tokens(text)
            if image_inputs and media_tokens["image"] == 0:
                injected = _inject_media_tokens(sanitized_messages, image_tokens=1, video_tokens=0)
                text = processor.apply_chat_template(injected, tokenize=False)

            batch_images.append([image_inputs[0]])

        elif example["task"] == "temporal-spatial free-form QA":
            width, height = video_inputs[0].size(3), video_inputs[0].size(2)
            image_size = (width, height)

            # Here, we need to add key frames.
            key_frame_root = STR_KF_ROOT
            if example["source"] == "STR_plm_rdcap":
                key_frame_root = STR_PLM_KF_ROOT

            key_frames = []

            for key_frame in example["key_frames"]:
                kf_path = os.path.join(key_frame_root, key_frame["path"])
                kf = Image.open(kf_path)
                kf = kf.convert("RGB")
                resized_kf = kf.resize(image_size)
                resized_kf = np.array(resized_kf)
                resized_kf = np.transpose(resized_kf, (2, 0, 1))
                resized_kf = torch.from_numpy(resized_kf)
                key_frames.append((key_frame["time"], resized_kf))

            frame_prompt = ""
            refined_image_inputs = []
            kf_idx = 0
            ori_idx = 0
            frame_idx = 1
            while ori_idx < len(video_inputs[0]):
                time_now = int(ori_idx / video_kwargs["fps"][0])
                if kf_idx < len(key_frames) and time_now >= key_frames[kf_idx][0]:
                    refined_image_inputs.append(key_frames[kf_idx][1])
                    time_now = key_frames[kf_idx][0]
                    frame_prompt += (
                        f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                    )
                    kf_idx += 1
                else:
                    refined_image_inputs.append(video_inputs[0][ori_idx])
                    time_now = round(ori_idx / video_kwargs["fps"][0], 1)
                    frame_prompt += (
                        f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                    )
                    ori_idx += 1
                frame_idx += 1

            refined_image_inputs = torch.stack(refined_image_inputs)
            text = _ensure_video_pad(text)
            text = text.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
            text = replace_boxes_for_gemini_data(text, image_size)

            batch_images.append(list(refined_image_inputs))

        elif example["task"] == "temporal QA" or "General video QA" in example["task"]:
            frame_prompt = ""
            ori_idx = 0
            while ori_idx < len(video_inputs[0]):
                time_now = round(ori_idx / video_kwargs["fps"][0], 1)
                frame_prompt += f"Frame {ori_idx + 1} at {time_now}: <|vision_start|><|image_pad|><|vision_end|>\n"
                ori_idx += 1
            frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
            text = _ensure_video_pad(text)
            text = text.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)

            batch_images.append(list(video_inputs[0]))
        else:
            raise ValueError(f"Unknown task: {example['task']}")

        texts.append(text)
        if debug_pipeline:
            example_times.append(time.perf_counter() - example_start)

    processor_start = time.perf_counter() if debug_pipeline else None
    try:
        inputs = processor(
            text=texts,
            images=batch_images,
            videos=None,
            return_tensors="pt",
            padding=True,
        )
    except Exception:
        batch_tasks = [ex.get("task") for ex in examples]
        print(f"[collate_fn] processor failed batch_tasks={batch_tasks}", file=sys.stderr)
        raise
    finally:
        if debug_pipeline:
            collate_fn._debug_counter += 1
            should_log = collate_fn._debug_counter <= 5 or collate_fn._debug_counter % 50 == 0
            if should_log:
                total_time = time.perf_counter() - batch_start
                processor_time = time.perf_counter() - processor_start
                batch_tasks = [ex.get("task") for ex in examples]
                batch_image_counts = [len(images) for images in batch_images]
                avg_example = sum(example_times) / max(len(example_times), 1)
                print(
                    "[collate_fn] timing "
                    f"batch={collate_fn._debug_counter} "
                    f"total_s={total_time:.3f} "
                    f"processor_s={processor_time:.3f} "
                    f"avg_example_s={avg_example:.3f} "
                    f"tasks={batch_tasks} "
                    f"images_per_sample={batch_image_counts}",
                    file=sys.stderr,
                )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs


class MySFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch) 
        # we can add more loss here
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
        # quantization_config=bnb_config,
    )
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen3-VL" in model_config.model_name_or_path:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    # Configure PyTorch backends (CUDNN benchmark, TF32)
    _configure_torch_backends()

    # Apply LinearPatchEmbed optimization for Qwen3-VL if enabled
    _replace_conv3d_with_linear_patch_embed(model)

    # Ensure flash attention is used
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "flash_attention_2"
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "flash_attention_2"

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    # Prepare dataset
    prepared_features = Features(
        {
            "messages": [
                {
                    "role": Value("string"),
                    "content": [
                        {
                            "type": Value("string"),
                            "text": Value("string"),
                            "image": Value("string"),
                            "video": Value("string"),
                        }
                    ],
                }
            ],
            "image_size": [Value("int64")],
            "key_frames": [
                {
                    "idx": Value("int64"),
                    "path": Value("string"),
                    "time": Value("float64"),
                }
            ],
            "task": Value("string"),
            "source": Value("string"),
        }
    )
    prepared_dataset = dataset["train"].map(
        prepare_dataset,
        num_proc=16,
        remove_columns=dataset["train"].column_names,
        desc="Preparing dataset",
        features=prepared_features,
    )

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer
    trainer = MySFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
    )

    # Train model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
