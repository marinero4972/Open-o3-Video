from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_from_disk

SYSTEM_PROMPT = {
    "visual QA": "A conversation between user and assistant. The user provides an image and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant must localize the object with bounding box coordinates between <box> and </box>. The answer must strictly follow the following format:`<obj>object_name</obj><box>bounding_box</box>'.",
    "temporal-spatial free-form QA": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed.",
    "temporal QA": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant determines the precise time period that answers the question. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When mentioning time during the reasoning process, the assistant must use the format: `<t>time_in_seconds</t>s'.The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'.",
    "temporal QA (MCQ)": "A conversation between user and assistant. The user provides a video and a multiple-choice question, and the Assistant determines the precise time period that answers the question and selects the correct option. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When mentioning time during the reasoning process, the assistant must use the format: `<t>time_in_seconds</t>s'. The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s.\nCorrect Option: [ONLY THE LETTER]'.",
    "General video QA MCQ": "A conversation between user and assistant. The user provides a video and asks a multiple-choice question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. Only output the correct option in the <answer> </answer> section.",
    "General video QA Free-form": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed."
}

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
}

def make_conversation_image_and_video(example):

    task = example.get('task')

    if task == 'visual QA':
        system_message = SYSTEM_PROMPT['visual QA']
        content_list = [{"type": "image"}, {"type": "text", "text": example['question']}]
    elif task in ['temporal-spatial free-form QA', 'temporal QA', 'temporal QA (MCQ)', 'General video QA MCQ', 'General video QA Free-form']:
        system_message = SYSTEM_PROMPT[task]
        content_list = [{"type": "video"}, {"type": "text", "text": example['question']}]
    else:
        raise ValueError(f"Unknown task: {task}")

    prompt_list = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": content_list}
    ]

    example['prompt'] = prompt_list   
    return example


def get_data(script_args):
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
            dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    dataset = dataset.map(make_conversation_image_and_video)

    train_dataset = dataset['train']
    num_to_keep = len(train_dataset) - (len(train_dataset) % 4)
    dataset['train'] = train_dataset.select(range(num_to_keep))
    print(f"Dataset 'train' split size: {num_to_keep}")
    print(dataset)

    return dataset
