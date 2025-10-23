from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np


class QwenVL_VLLM:
    def __init__(self, llm_name, rt_shape=False, **llm_args):
        temperature = llm_args.pop("temperature", 0.0)
        max_tokens = llm_args.pop("max_tokens", 512)

        self.rt_shape = rt_shape

        self.video_max_pixels = llm_args.pop("video_max_pixels", 360 * 420)
        self.video_max_frames = llm_args.pop("video_max_frames", 16)
        print("Start initialize the model.")

        self.llm = LLM(
            model=llm_name,
            limit_mm_per_prompt={"image": 32, "video": 10},
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_num_seqs=5,
            gpu_memory_utilization=0.8,
            **llm_args,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            # top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=max_tokens,
            stop_token_ids=[],
        )
        self.sample_params = sampling_params
        self.processor = AutoProcessor.from_pretrained(llm_name, max_pixels=854 * 480)
        self.processor.tokenizer.padding_side = "left"
        print(f"Initialize model {llm_name} successfully with args {llm_args}")

    def get_batch_messages(self, video_paths, queries, query_image, duration=1.0):
        messages = []
        for video_path, query, image in zip(video_paths, queries, query_image):
            content = [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": self.video_max_pixels,
                    "max_frames": self.video_max_frames,
                },
                {"type": "text", "text": query},
            ]
            if image is not None:
                content.insert(1, {"type": "image", "image": image})
            messages.append([{"role": "user", "content": content}])

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        fps = video_kwargs["fps"][0]

        # if video_inputs:
        #     for i, v_input in enumerate(video_inputs):
        #         print(f"Video {i+1} tensor shape: {v_input.shape}")

        if image_inputs:
            return [
                {
                    "prompt": query,
                    "multi_modal_data": {
                        "video": v_input.numpy(),
                        "image": np.array(i_input),
                    },
                }
                for i_input, v_input, query in zip(image_inputs, video_inputs, texts)
            ], fps

        return [
            {
                "prompt": query,
                "multi_modal_data": {"video": v_input.numpy()},
            }
            for v_input, query in zip(video_inputs, texts)
        ], fps

    def __call__(self, video_path, query, query_image, **kwargs):
        if isinstance(video_path, list) and isinstance(query, list):
            inputs, fps = self.get_batch_messages(video_path, query, query_image)
        else:
            raise ValueError("video_path and query must be list or str")
        # print(inputs[0]["prompt"])

        frame_shape = (inputs[0]["multi_modal_data"]["video"].shape[3], inputs[0]["multi_modal_data"]["video"].shape[2])
        frames = inputs[0]["multi_modal_data"]["video"]
        # print(inputs[0]["multi_modal_data"]["video"].shape)
        
        outputs = self.llm.generate(inputs, sampling_params=self.sample_params)
        if self.rt_shape:
            return outputs, frames, fps, frame_shape
        return outputs, frames, fps
    
    def run_images_scorer(self, msgs, images):
        query = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        inputs = [{
                    "prompt": query,
                    "multi_modal_data": {
                        "image": images
                    },
                }]
        outputs = self.llm.generate(inputs, sampling_params=self.sample_params)
        # print(outputs[0].outputs[0].text)
        if outputs[0].outputs[0].text in ["0","1","2"]:
            return int(outputs[0].outputs[0].text)
        else:
            return -1
    
    def inference_wo_process(self, inputs):
        outputs = self.llm.generate(inputs, sampling_params=self.sample_params)
        return outputs[0].outputs[0].text
