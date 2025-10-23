import os
import re
import sys
from collections import defaultdict
import numpy as np
from PIL import Image
import io
import time
import pandas as pd


class VideoMME_Bench:
    def __init__(self, data_dir, add_asr=False, asr_dir=None, think_mode=False):
        self.data_dir = data_dir
        # add_asr: whether to add subtitles to the text input
        self.add_asr = add_asr
        self.asr_dir = asr_dir
        self.data_dir = data_dir
        self.think_mode = think_mode
        print("think mode:", self.think_mode)

    def get_data(self):
        print("Loading data...")
        self.docs = []
        all_docs = []
        filename = os.path.join(self.data_dir, "videomme/test-00000-of-00001.parquet")
        self.docs.append(read_parquet_file(filename))
        num_docs = sum([len(docs) for docs in self.docs])
        count = 0
        video_paths, image_input, text_input, answers = [], [], [], []
        for docs in self.docs:
            for index, row in docs.iterrows():
                doc = row.to_dict()
                all_docs.append(doc)
                video_p, img, txt = self.process_data(doc)
                video_paths.extend(video_p)
                image_input.extend(img)
                text_input.extend(txt)
                answers.append(doc["answer"])
                count += 1
        print(f"Data loaded: {count}/{num_docs}")

        return video_paths, image_input, text_input, all_docs

    def process_data(self, doc):
        video_path, image = videomme_doc_to_visual(doc, self.data_dir)
        text = videomme_doc_to_text(doc, self.think_mode)
        # add subtitles
        if self.add_asr:
            asr_path = os.path.join(self.asr_dir, doc["videoID"] + ".mp4.txt")
            asr = ""
            if os.path.exists(asr_path):
                with open(asr_path, "r", encoding="utf-8") as f:
                    asr = f.read()
            if asr == "":
                text_input = [
                    "Audio transcripts of the video:\n"
                    + "This video does not have audio transcripts.\nQuestion:"
                    + text[0]
                ]
            else:
                text_input = [
                    "Audio transcripts of the video:\n" + asr + "\nQuestion:" + text[0]
                ]
        else:
            text_input = text

        return video_path, image, text_input


def read_parquet_file(file_path):
    df = pd.read_parquet(file_path)
    return df



import os
import re
import sys
import cv2
import numpy as np
from loguru import logger as eval_logger


VIDEO_TYPE = ["short", "medium", "long"]
CATEGORIES = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual",
]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    res = float(float(h) * 3600 + float(m) * 60 + float(s) + float(ms) / 1000)
    return res


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text, start_time, end_time))

    return subtitle_frames, total_frame


def videomme_doc_to_visual(doc, cache_dir):

    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path], [None]


def videomme_doc_to_text(doc, think_mode=False):
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option

    if not think_mode:
        option_prompt = "Select the best answer to the multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
        full_prompt = option_prompt + "\n" + question + "\n"
    else:
        option_prompt = "Select the best answer to the multiple-choice question based on the video. You must first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The reasoning process MUST NOT be longer than 100 words. In the answer part, respond with only the letter (A, B, C, or D) of the correct option."
        full_prompt = "Question:" + question + "\n" + option_prompt
    return [full_prompt]


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
        "Final answer:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    """    
    # 查找所有匹配
    matches = re.findall(r"[ABCD]", s)
    print(matches)  # 打印所有匹配的字符

    if not matches:  # 如果没有匹配
        return ""
    
    return matches[-1]  # 返回最后一个匹配的字符
    """
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def videomme_process_results_new(doc, pred, think=None, frame_shape=None):
    """
    Args:
        doc: a instance of the eval dataset
        pred: the prediction of the model
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    index2ans, all_choices = get_multi_choice_info((doc["options"]))
    pred_ans = parse_multi_choice_response(pred, all_choices, index2ans)

    category = doc["domain"]
    sub_category = doc["sub_category"]
    task_category = doc["task_type"]
    data_dict = {
        "question_id": doc["question_id"],
        "duration": doc["duration"],
        "category": category,
        "sub_category": sub_category,
        "task_category": task_category,
        "pred_answer": pred_ans,
        "answer": doc["answer"],
        "response": pred,
        "reasoning_process": think,
        "frame_shape": frame_shape,
        "video_id": doc["videoID"],
    }

    return data_dict


def videomme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for video_type in VIDEO_TYPE:
        for category in CATEGORIES:
            for sub_category in SUB_CATEGORIES:
                for task_category in TASK_CATEGORIES:
                    key = f"{video_type}_{category}_{sub_category}_{task_category}"
                    category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        video_type = result["duration"]
        category = result["category"]
        sub_category = result["sub_category"]
        task_category = result["task_category"]
        key = f"{video_type}_{category}_{sub_category}_{task_category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for video_type in VIDEO_TYPE:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_type in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(
            f"Evaluation on video Type: {video_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
        )

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(
            f"Evaluation on Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
        )

    for sub_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if sub_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(
            f"Evaluation on Video Sub Categories: {sub_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
        )

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(
            f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
        )

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(
        f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
    )
    return 100 * total_correct / total_answered if total_answered > 0 else 0


def parse_answer(pred, doc):
    index2ans, all_choices = get_multi_choice_info((doc["options"]))
    pred_ans = parse_multi_choice_response(pred, all_choices, index2ans)
    return pred_ans
    
"""______________________________________________________________________________________________________________________________________________________________________"""


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if response == "API Error" or response == "":
        return "API Error"

    # Step 1: Clean up punctuation from the response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    ans_with_period = False
    ans_with_colon = False
    candidates = []

    # Step 2: If no candidates, look for choices with a period after (A. B. C. D.)
    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            # print(f"Found choice with period after: {choice}")
            candidates.append(f"{choice}.")
            ans_with_period = True
    # print("Candidates found:", candidates)
    # Step 2.1: If no candidates, look for choices with a colon after (A: B: C: D:)
    for choice in all_choices:  # e.g., A: B: C: D:
        if f"{choice}:" in response:
            # print(f"Found choice with semicolon after: {choice}")
            candidates.append(f"{choice}:")
            ans_with_colon = True
    # print("Candidates found:", candidates)
    # Step 3: Look for choices with parentheses e.g., (A) (B) (C) (D)
    # if len(candidates) == 0:
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            # print(f"Found choice with parentheses: {choice}")
            candidates.append(f"({choice})")
            ans_with_brack = True
    # print("Candidates found:", candidates)
    # Step 4: If no candidates, look for choices with a space after (A B C D)
    # if len(candidates) == 0:
    for choice in all_choices:  # e.g., A B C D
        if f"{choice} " in response:
            # print(f"Found choice without parentheses (space after): {choice}")
            candidates.append(f"{choice} ")
            ans_with_space = True
    # print("Candidates found:", candidates)

    # Check for choices with newlines around them e.g., \nD\n
    for choice in all_choices:  # e.g., D
        if f"\n{choice}\n" in response:
            # print(f"Found choice with newlines around: {choice}")
            candidates.append(f"\n{choice}\n")

    for choice in all_choices:  # e.g., D
        if f" {choice}\n" in response:
            candidates.append(f" {choice}\n")

    for choice in all_choices:  # e.g., D
        if f"\n{choice} " in response:
            candidates.append(f"\n{choice} ")
    for choice in all_choices:  # e.g., D
        if f": {choice}" in response:
            candidates.append(f": {choice}")
    for choice in all_choices:  # e.g., D
        if f":{choice}" in response:
            candidates.append(f":{choice}")
    for choice in all_choices:  # e.g., D
        if f":\n{choice}" in response:
            candidates.append(f":\n{choice}")
    for choice in all_choices:  # e.g., D
        if f"\n\n{choice}" in response:
            candidates.append(f"\n\n{choice}")
    # 补充两个新的
    for choice in all_choices:  # e.g., **D**
        if f"**{choice}**" in response:
            candidates.append(f"**{choice}**")
    for choice in all_choices:  # e.g., {D}
        if f"{{{choice}}}" in response:
            candidates.append(f"{{{choice}}}")

    # Step 5: If no candidates and response has more than 5 tokens, try parsing based on content
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                # print(f"Found answer content match: {ans}")
                candidates.append(index)
                index_ans = False  # It's content answer, not an index
    # print("candidates:", candidates)
    # Step 6: If still no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = "No Answer Found"
        # print(f"No candidates found.")
    # Step 7: If multiple candidates found, use the one appearing last

    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            for can in candidates:
                index = response.rfind(can)
                # print(f"Checking position of choice: {can} at {index}")
                start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                # print(f"Checking position of content match: {can} at {index}")
                start_indexes.append(index)
        # Get the last one (max index)
        # rint("start_indexes:", start_indexes)
        pred_index = candidates[np.argmax(start_indexes)]
        for choice in all_choices:
            if choice in pred_index:
                pred_index = choice
                break
        # print(f"Multiple candidates, selected based on last occurrence: {pred_index}")
    else:
        # If only one candidate, use it
        pred_index = candidates[0]
        # print(f"Only one candidate found, selected: {pred_index}")
        for choice in all_choices:
            if choice in pred_index:
                pred_index = choice
                break
    return pred_index


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices
