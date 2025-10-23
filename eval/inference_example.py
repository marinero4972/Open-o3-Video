import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

# Set model path
model_path = "..."

# Set video path and question
video_path_list = ["./example_video.mp4" for _ in range(2)]
question_list = ["What is the color of the bowling ball?", "What is the first scene about?"]

# Initialize the LLM
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    max_model_len=81920,
    gpu_memory_utilization=0.7,
    limit_mm_per_prompt={"video": 1, "image": 16},
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=2048,
    stop_token_ids=[],
)

# Load processor and tokenizer
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer
system_message = "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed.",

for idx in range(len(question_list)):
    question = question_list[idx]
    video_path = video_path_list[idx]
    # Construct multimodal message
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
                    "video": video_path,
                    "nframes": 16      # max frame number
                },
                {
                    "type": "text",
                    "text": question
                },
            ],
        }
    ]

    # Convert to prompt string
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process video input
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    frame_prompt = ""
    for i in range(len(video_inputs[0])):
        frame_prompt += f"Frame {i+1} at {round(i / video_kwargs['fps'][0],1)} second: <|vision_start|><|image_pad|><|vision_end|>\n"    
    prompt = prompt.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)


    llm_inputs = [{
        "prompt": prompt,
        "multi_modal_data": {"image": video_inputs[0]},
    }]

    # Run inference
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    output_text = outputs[0].outputs[0].text

    print(output_text)
