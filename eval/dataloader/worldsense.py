import os
import re
import sys
from collections import defaultdict
import numpy as np
from PIL import Image
import io
import time
import pandas as pd
import os.path as osp
import pickle
import json
import pandas as pd
import csv

BASE_SYS = 'Carefully watch this video and pay attention to every detail. '
SYS = BASE_SYS + 'Based on your observations, select the best option that accurately addresses the question.'


FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Audio transcripts of the video:\nThis video does not have audio transcripts. \
"""

REQUIREMENTS = """
Select the best answer to the multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option. \
"""

REQUIREMENTS_THK = """
Select the best answer to the multiple-choice question based on the video. \
You must first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The reasoning process MUST NOT be longer than 100 words.\
In the answer part, respond with only the letter (A, B, C, or D) of the correct option. \
"""

FRAMES_TMPL_SUB = """
These are the frames of a video. \
Audio transcripts of the video:\n
{}
"""

FRAMES_TMPL_AUDIO = """
These are the frames of a video and the corresponding audio. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option. \
"""

class WorldSense_Bench:
    def __init__(self, data_dir=None, add_asr=True, asr_dir=None, think_mode=False):
        self.data_dir = data_dir
        # add_asr: whether to add subtitles to the text input
        self.add_asr = add_asr
        self.asr_dir = asr_dir
        self.think_mode = think_mode
        print("think mode:", self.think_mode)

    def get_data(self):
        print("Loading data...")
        self.docs = []
        all_docs = []
        filename = os.path.join(self.data_dir,'WorldSense.tsv')
        self.docs.append(pd.read_csv(filename, sep="\t"))
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
                answers.append(doc['answer'])
                count += 1
        print(f"Data loaded: {count}/{num_docs}")
        

        return video_paths, image_input, text_input, all_docs

    def process_data(self, line):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        asr_path = osp.join(self.asr_dir, line['video'] + '.wav.txt')
        if osp.exists(asr_path) and self.add_asr:
            with open(asr_path, 'r') as f:
                subtitles = f.read()
        else:
            subtitles = ''
        video_path = osp.join(self.data_dir, line['video'] + '.mp4')
        text_input = SYS + FRAMES_TMPL_SUB.format(subtitles) if subtitles != '' else SYS + FRAMES_TMPL_NOSUB
 
        question_str = line['question'] + '\n' + '\n'.join(eval(line['candidates']))
        prompt = 'Question: {}\n'.format(question_str)
        text_input+=prompt
        if self.think_mode:
            text_input+=REQUIREMENTS_THK
        else:
            text_input+=REQUIREMENTS
        
        return [video_path], [None], [text_input]

def read_parquet_file(file_path):
    df = pd.read_parquet(file_path)
    return df

import json
import os
import re
import sys
import cv2
import numpy as np


"""______________________________________________________________________________________________________________________________________________________________________"""



def parse_multi_choice_response(response, all_choices=['A', 'B', 'C', 'D'], index2ans=True):
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
    import random
    random.seed(42)

    index_ans = True
    ans_with_brack = False
    ans_with_period = False
    ans_with_colon = False
    candidates = []

    # Step 2: If no candidates, look for choices with a period after (A. B. C. D.)
    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            #print(f"Found choice with period after: {choice}")
            candidates.append(f"{choice}.")
            ans_with_period = True
    #print("Candidates found:", candidates)
    # Step 2.1: If no candidates, look for choices with a colon after (A: B: C: D:)
    for choice in all_choices:  # e.g., A: B: C: D:
        if f"{choice}:" in response:
            #print(f"Found choice with semicolon after: {choice}")
            candidates.append(f"{choice}:")
            ans_with_colon = True
    #print("Candidates found:", candidates)
    # Step 3: Look for choices with parentheses e.g., (A) (B) (C) (D)
    # if len(candidates) == 0:
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            #print(f"Found choice with parentheses: {choice}")
            candidates.append(f"({choice})")
            ans_with_brack = True
    #print("Candidates found:", candidates)
    # Step 4: If no candidates, look for choices with a space after (A B C D)
    # if len(candidates) == 0:
    for choice in all_choices:  # e.g., A B C D
        if f"{choice} " in response:
            #print(f"Found choice without parentheses (space after): {choice}")
            candidates.append(f"{choice} ")
            ans_with_space = True
    #print("Candidates found:", candidates)

    # Check for choices with newlines around them e.g., \nD\n
    for choice in all_choices:  # e.g., D
        if f"\n{choice}\n" in response:
            #print(f"Found choice with newlines around: {choice}")
            candidates.append(f"\n{choice}\n")
    
    for choice in all_choices:  # e.g., D
        if f" {choice}\n" in response:
            candidates.append(f" {choice}\n")

    for choice in all_choices:  # e.g., D
        if f"\n{choice} " in response:
            candidates.append(f"\n{choice} ")
    for choice in all_choices:  # e.g., D
        if f": {choice}" in response:
            candidates.append( f": {choice}")
    for choice in all_choices:  # e.g., D
        if f":{choice}" in response:
            candidates.append(f":{choice}")
    for choice in all_choices:  # e.g., D
        if f":\n{choice}" in response:
            candidates.append(f":\n{choice}")
    for choice in all_choices:  # e.g., D
        if f"\n\n{choice}" in response:
            candidates.append(f"\n\n{choice}")
    
    for choice in all_choices:  # e.g., **D**
        if f"**{choice}**" in response:
            candidates.append(f"**{choice}**")
    for choice in all_choices:  # e.g., {D}
        if f"{{{choice}}}" in response:  
            candidates.append(f"{{{choice}}}")

    # print("candidates:", candidates)
    # Step 6: If still no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = "No Answer Found"
        #print(f"No candidates found.")
    # Step 7: If multiple candidates found, use the one appearing last
    
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            for can in candidates:
                index = response.rfind(can)
                #print(f"Checking position of choice: {can} at {index}")
                start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                #print(f"Checking position of content match: {can} at {index}")
                start_indexes.append(index)
        # Get the last one (max index)
        #print("start_indexes:", start_indexes)
        pred_index = candidates[np.argmax(start_indexes)]
        for choice in all_choices:
            if choice in pred_index:
                pred_index = choice
                #print("index is", pred_index,"!!!!!!!!")
                break
        #print(f"Multiple candidates, selected based on last occurrence: {pred_index}")
    else:
        # If only one candidate, use it
        pred_index = candidates[0]
        # print(f"Only one candidate found, selected: {pred_index}")
        for choice in all_choices:
            if choice in pred_index:
                #print("index is", pred_index,"!!!!!!!!")
                pred_index = choice
                break
    if pred_index not in all_choices:
        # print("Random!!")
        # print("Original Response", response)
        pred_index = random.choice(all_choices)
    return pred_index



FAIL_MSG = 'Failed to obtain answer via API.'

DURATIONS = [
    "<1min",
    "1-2min",
    "2-4min",
    "4-6min",
    "6-8min",
    ">8min"
]

DOMAINS = [
    'Tech & Science',
    'Culture & Politics',
    'Daily Life',
    'Film & TV',
    'Performance',
    'Games',
    'Sports',
    'Music',
]

SUB_CATEGORIES = [
    "Academic Lectures",
    "Auto",
    "Software",
    "Physics",
    "Climate Change",
    "Space Missions",
    "Chemistry",
    "Engineering Projects",
    "Biology",
    "Science Explainers",
    "Artificial Intelligence",
    "Astronomy",
    "Tech Reviews",
    "Editorials",
    "Politics",
    "Historical Analysis",
    "Social Commentary",
    "Book Reviews",
    "Cultural Explainers",
    "Drawing Tutorials",
    "Celebrity Interviews",
    "Art Exhibitions",
    "Fashion",
    "Travel",
    "Daily Vlogs",
    "Cooking",
    "Pranks",
    "Camping",
    "Nutrition & Health",
    "Home Improvement",
    "Painting & Photography",
    "Unboxing Videos",
    "Family Vlogs",
    "DIY & Crafts",
    "Skincare & Makeup",
    "Documentaries",
    "Film Trailers",
    "Event Livestreams",
    "Short Films",
    "Documentary Profiles",
    "Movie Reviews",
    "World News",
    "Talks",
    "Parodies",
    "Storytime",
    "Stand-up",
    "Sketches",
    "FPS Game",
    "Casual Game",
    "Role Playing Game",
    "Sports Game",
    "Basketball",
    "Racing",
    "Football",
    "Bowling Ball",
    "Soccer",
    "Motorsport",
    "swimming",
    "Boxing",
    "Other Sports",
    "Fitness",
    "Fishing",
    "Hiking",
    "Covers",
    "Music Videos",
    "Remixes",
    "Walkthroughs"
]

TASK_DOMAINS = [
    'Recognition',
    'Understanding',
    'Reasoning'
]

TASK_CATEGORIES = [
    "Anomaly Recognition",
    "Event Recognition",
    "Attribute Recognition",
    "Human Interaction",
    "Temporal Localization",
    "Video Emotions",
    "Event Sorting",
    "Hallucination",
    "Text and Diagram Understanding",
    "Attribute Reasoning",
    "Causal Reasoning",
    "Object Counting",
    "Action Counting",
    "Temporal Prediction",
    "Emotion Change",
    "Audio Counting",
    "Scene Recognition",
    "Human-object Interaction",
    "Human Emotions",
    "Object State Change",
    "Relation Reasoning",
    "Spatial Relation",
    "Audio Source Localization",
    "Audio Recognition",
    "Object Existence Recognition",
    "Audio Change"
]

AUDIO_CLASSES = [
    "Speech",
    "Event",
    "Music",
]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)

# LOAD & DUMP
def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)

def get_dimension_rating(data_path):
    data = load(data_path)

    duration_rating = {k: {} for k in DURATIONS}
    for duration in DURATIONS + ['overall']:
        duration_rating[duration] = {
            'overall': '',
            'domain': {k: [] for k in DOMAINS},
            'sub_category': {k: [] for k in SUB_CATEGORIES},
            'task_domain': {k: [] for k in TASK_DOMAINS},
            'task_type': {k: [] for k in TASK_CATEGORIES},
            'audio_class': {k: [] for k in AUDIO_CLASSES},
        }

    for i in range(len(data)):

        domain = data.iloc[i]['domain']
        sub_ctg = data.iloc[i]['sub_category']
        task_domain_ctg = data.iloc[i]['task_domain']
        task_ctg = data.iloc[i]['task_type']
        audio_ctg = eval(data.iloc[i]['audio_class'])

        duration = data.iloc[i]['duration']
        score = float(data.iloc[i]['score'])

        duration_rating['overall']['domain'][domain].append(score)
        duration_rating['overall']['sub_category'][sub_ctg].append(score)
        duration_rating['overall']['task_domain'][task_domain_ctg].append(score)
        duration_rating['overall']['task_type'][task_ctg].append(score)

        duration_rating[duration]['domain'][domain].append(score)
        duration_rating[duration]['sub_category'][sub_ctg].append(score)
        duration_rating[duration]['task_domain'][task_domain_ctg].append(score)
        duration_rating[duration]['task_type'][task_ctg].append(score)

        for _audio_ctg in audio_ctg:
            duration_rating['overall']['audio_class'][_audio_ctg].append(score)
            duration_rating[duration]['audio_class'][_audio_ctg].append(score)
            
    for duration in ['overall'] + DURATIONS:

        overall_res_dur = f'{np.mean([x for x in sum(duration_rating[duration]["domain"].values(), []) if x >= 0]):.3f}'
        duration_rating[duration]['overall'] = overall_res_dur

        for domain in DOMAINS:
            domain_res_dur = f'{np.mean([x for x in duration_rating[duration]["domain"][domain] if x >= 0]):.3f}'
            duration_rating[duration]['domain'][domain] = domain_res_dur

        for sub_ctg in SUB_CATEGORIES:
            sub_res_dur = f'{np.mean([x for x in duration_rating[duration]["sub_category"][sub_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['sub_category'][sub_ctg] = sub_res_dur

        for task_ctg in TASK_DOMAINS:
            task_res_dur = f'{np.mean([x for x in duration_rating[duration]["task_domain"][task_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['task_domain'][task_ctg] = task_res_dur
        
        for task_ctg in TASK_CATEGORIES:
            task_res_dur = f'{np.mean([x for x in duration_rating[duration]["task_type"][task_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['task_type'][task_ctg] = task_res_dur

        for audio_ctg in AUDIO_CLASSES:
            audio_res_dur = f'{np.mean([x for x in duration_rating[duration]["audio_class"][audio_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['audio_class'][audio_ctg] = audio_res_dur

    return duration_rating

def worldsense_process_result(eval_file, **judge_kwargs):

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')


        res = {} if not osp.exists(tmp_file) else load(tmp_file)
        res = {k: v for k, v in res.items() if FAIL_MSG not in v}

        data = load(eval_file)
        data_un = data[~pd.isna(data['prediction'])]

        for idx in data['index']:
            ans = data.loc[data['index'] == idx, 'answer'].values[0]
            pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
            extracted_pred = parse_multi_choice_response(pred)
            # 找到这个 idx 对应的 dataframe 行索引
            row_idx = data.index[data['index'] == idx][0]
            data.at[row_idx, 'score'] = int(extracted_pred == ans)
            # data.loc[idx, 'score'] = int(extracted_pred == ans)

        rejected = [x for x in data['score'] if x == -1]

        print(
            f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
            f'failed to obtain the score for another {len(rejected)} questions. '
            f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
        )

        dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating