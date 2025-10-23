import json
import numpy as np
from tqdm import tqdm
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
import argparse
import os

parser = argparse.ArgumentParser(description="Process video data with a specified model.")
parser.add_argument('--result_file', type=str, default="none.json", 
                    help='Path to save the output result JSON file.')
parser.add_argument('--model_path', type=str, default="/path/to/Qwen2.5-72B-Instruct", 
                    help='Path to save the output result JSON file.')
args = parser.parse_args()

result_file = args.result_file
model_name = args.model_path

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

system_prompt = """
As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.
Your assessment should range from 0 to 3, \
based solely on the semantic similarity between the groundtruth and the candidate answer, \
disregarding any grammatical differences.
A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.
Your response should be a single integer from 0, 1, 2, or 3.
"""

# tmpl = 'Groundtruth answer: {}\nCandidate answer: {}\nYour response: '
tmpl = 'Question: {}\nGroundtruth answer: {}\nCandidate answer: {}\nYour response: '

def qwen2_5_evaluation(question, gt, candidate):
    user_prompt=tmpl.format(question, gt, candidate)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    score = response
    # print(score)
    # breakpoint()
    try:
        score = int(score)
    except (ValueError, TypeError):
        score = -1
    return score

def refined_timestamps(result):
    import re
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
    
def calculate_temporal_iou(gt_range, pred_range):
    """ calculate Temporal IoU"""
    if not pred_range:
        return 0.0  
    
    if isinstance(pred_range, str):
        try:
            pred_range = ast.literal_eval(pred_range)
        except (ValueError, SyntaxError):
            return 0.0

    if not isinstance(pred_range, (list, tuple)) or len(pred_range) != 2 or \
    not all(isinstance(x, (int, float)) for x in pred_range):
        return 0.0

    gt_start, gt_end = gt_range
    pred_start, pred_end = pred_range
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    return intersection / union if union > 0 else 0.0


def compute_iou(gt_bbox, pred_bbox):
    """calculate 2 bbox IoU"""
    if not isinstance(pred_bbox, (list, tuple)) or len(pred_bbox) != 4:
        return 0.0
    
    # GT bbox
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox['xmin'], gt_bbox['ymin'], gt_bbox['xmax'], gt_bbox['ymax']
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_bbox

    # Intersection
    x1 = max(gt_xmin, pred_xmin)
    y1 = max(gt_ymin, pred_ymin)
    x2 = min(gt_xmax, pred_xmax)
    y2 = min(gt_ymax, pred_ymax)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union
    gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    union = gt_area + pred_area - intersection

    return intersection / union if union > 0 else 0.0

def calculate_bbox_iou(gt_bbox, pred_bboxes):
    """Calculate single BBox IoU, support multiple prediction frames to get maximum IoU"""
    try:
        if not pred_bboxes:
            return 0.0

        if isinstance(pred_bboxes[0], (int, float)) and len(pred_bboxes) == 4:
            pred_bboxes = [pred_bboxes]

        return max([compute_iou(gt_bbox, pred_bbox) for pred_bbox in pred_bboxes])
    except:
        return 0.0

def calculate_spatial_metrics(gt_bboxes, pred_bboxes):
    """calculate Spatial IoU and mAP"""
    if not pred_bboxes:
        return [0.0] * 5, 0.0

    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    ious = []
    aps = []
    for box in gt_bboxes:
        frame_id = str(box["timestamp"])
        if isinstance(pred_bboxes, dict) and frame_id in pred_bboxes:
            pred_bbox = pred_bboxes[frame_id]
            gt_bbox = {
                "xmin": box["xmin"],
                "ymin": box["ymin"],
                "xmax": box["xmax"],
                "ymax": box["ymax"]
            }
            iou = calculate_bbox_iou(gt_bbox, pred_bbox)
            ious.append(iou)
        else:
            ious.append(0.0)
    mIoU = np.mean(ious) if ious else 0.0

    for threshold in iou_thresholds:
        scores = [1 if iou >= threshold else 0 for iou in ious]
        if len(ious) > 0:
            aps.append(np.mean(scores))
        else:
            aps.append(0.0)
    return aps, mIoU

def calculate_spatial_random(gt_bboxes, w, h):
    """calculate Spatial IoU and mAP"""
    pred_bbox = [0, 0, w, h]
    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    ious = []
    aps = []
    for gt_bbox_entry in gt_bboxes:
        for frame_id, gt_bbox in gt_bbox_entry.items():
            iou = calculate_bbox_iou(gt_bbox, pred_bbox)
            ious.append(iou)
    mIoU = np.mean(ious) if ious else 0.0

    for threshold in iou_thresholds:
        scores = [1 if iou >= threshold else 0 for iou in ious]
        if len(ious) > 0:
            aps.append(np.mean(scores))
        else:
            aps.append(0.0)
    return aps, mIoU

# evaluate the json file
def evaluate_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    model_name = "qwen"
    domains = {}
    durations = {}
    overall_stats = {"all_rating":[], "valid_rating": [], "correct_num":0, "temporal_ious": [], "temporal_ious_2": [], "spatial_aps": [[] for _ in range(5)],
                    "spatial_aps_2": [[] for _ in range(5)], "spatial_mious": [], "spatial_mious_2": [], "random_tious": [], "random_aps": [[] for _ in range(5)], "random_vious":[],
                    "vqa_temporal_idx":[], "vqa_spatial_idx":[], "temporal_spatial_idx":[],"vqa_temp_spatial_idx":[],
                    "vqa_temporal_idx_2":[], "vqa_spatial_idx_2":[], "temporal_spatial_idx_2":[],"vqa_temp_spatial_idx_2":[]}


    for idx, item in enumerate(tqdm(data, desc=f"Evaluating {model_name} results", unit="item")):
        video_length = round(item['frame_count']/item['fps'], 2)
        w, h = item['width'], item['height']
        domain = item.get("domain", "unknown")
        if domain not in domains:
            domains[domain] = {"all_rating":[], "valid_rating": [], "correct_num":0, "temporal_ious": [], "temporal_ious_2": [], "spatial_aps": [[] for _ in range(5)],
                    "spatial_aps_2": [[] for _ in range(5)], "spatial_mious": [], "spatial_mious_2": [], "random_tious": [], "random_aps": [[] for _ in range(5)], "random_vious":[],
                    "vqa_temporal_idx":[], "vqa_spatial_idx":[], "temporal_spatial_idx":[],"vqa_temp_spatial_idx":[],
                    "vqa_temporal_idx_2":[], "vqa_spatial_idx_2":[], "temporal_spatial_idx_2":[],"vqa_temp_spatial_idx_2":[]}
        
        if video_length < 60:
            duration = "Short"
        elif 60 <= video_length < 180:
            duration = "Medium"
        else:
            duration = "Long"
        if duration not in durations:
            durations[duration] = {"all_rating":[], "valid_rating": [], "correct_num":0, "temporal_ious": [], "temporal_ious_2": [], "spatial_aps": [[] for _ in range(5)],
                    "spatial_aps_2": [[] for _ in range(5)], "spatial_mious": [], "spatial_mious_2": [], "random_tious": [], "random_aps": [[] for _ in range(5)], "random_vious":[],
                    "vqa_temporal_idx":[], "vqa_spatial_idx":[], "temporal_spatial_idx":[],"vqa_temp_spatial_idx":[],
                    "vqa_temporal_idx_2":[], "vqa_spatial_idx_2":[], "temporal_spatial_idx_2":[],"vqa_temp_spatial_idx_2":[]}

        if 'answer_vqa' in item and item['answer_vqa']:
            score = qwen2_5_evaluation(item['question'], item['answer'], item['answer_vqa'])
        else:
            continue
        overall_stats["all_rating"].append(score if score != -1 else 0)
        domains[domain]["all_rating"].append(score if score != -1 else 0)
        durations[duration]["all_rating"].append(score if score != -1 else 0)
        if score != -1:
            overall_stats["valid_rating"].append(score)
            domains[domain]["valid_rating"].append(score)
            durations[duration]["valid_rating"].append(score)
        if score >= 2:
            overall_stats["correct_num"] += 1
            domains[domain]["correct_num"] += 1
            durations[duration]["correct_num"] += 1
        data[idx]["VQA_score"] = score
        # answer_temporal
        if item['answer_temporal'] == []:
            item['answer_temporal'] = refined_timestamps(item['answer_temporal_pre'])
            # print(item['answer_temporal_pre'])
            # print(item['answer_temporal'])
        
        if 'answer_temporal' in item and item['answer_temporal']:
            temporal_iou = calculate_temporal_iou(item['timestamps'], item['answer_temporal'])
        else:
            temporal_iou = 0.0

        overall_stats["temporal_ious"].append(temporal_iou)
        domains[domain]["temporal_ious"].append(temporal_iou)
        durations[duration]["temporal_ious"].append(temporal_iou)
        data[idx]["temporal_IoU"] = temporal_iou

        # answer_temporal_2
        if item['answer_temporal_2'] == []:
            item['answer_temporal_2'] = refined_timestamps(item['answer_temporal_pre_2'])
            # print(item['answer_temporal_pre_2'])
            # print(item['answer_temporal_2'])

        if 'answer_temporal_2' in item and item['answer_temporal_2']:
            temporal_iou_2 = calculate_temporal_iou(item['timestamps'], item['answer_temporal_2'])
        else:
            temporal_iou_2 = 0.0
        
        overall_stats["temporal_ious_2"].append(temporal_iou_2)        
        domains[domain]["temporal_ious_2"].append(temporal_iou_2)
        durations[duration]["temporal_ious_2"].append(temporal_iou_2)
        data[idx]["temporal_IoU_2"] = temporal_iou_2

        random_iou = calculate_temporal_iou(item['timestamps'],[0, video_length])
        overall_stats["random_tious"].append(random_iou)
        domains[domain]["random_tious"].append(random_iou)
        durations[duration]["random_tious"].append(random_iou)

        # answer_spatial
        if 'answer_spatial' in item and item['answer_spatial']:
            aps, mIoU = calculate_spatial_metrics(item['bboxes'], item['answer_spatial'])
        else:
            aps, mIoU = [0.0] * 5, 0.0
        for i, ap in enumerate(aps):
            domains[domain]["spatial_aps"][i].append(ap)
            durations[duration]["spatial_aps"][i].append(ap)
            overall_stats["spatial_aps"][i].append(ap)
        domains[domain]["spatial_mious"].append(mIoU)
        durations[duration]["spatial_mious"].append(mIoU)
        overall_stats["spatial_mious"].append(mIoU)
        data[idx]["AP1@0.1:0.9"] = aps
        data[idx]["spatial_mIoU"] = mIoU

        # answer_spatial_2
        if 'answer_spatial_2' in item and item['answer_spatial_2']:
            aps_2, mIoU_2 = calculate_spatial_metrics(item['bboxes'], item['answer_spatial_2'])
        else:
            aps_2, mIoU_2 = [0.0] * 5, 0.0
        for i, ap in enumerate(aps_2):
            domains[domain]["spatial_aps_2"][i].append(ap)
            durations[duration]["spatial_aps_2"][i].append(ap)
            overall_stats["spatial_aps_2"][i].append(ap)
        domains[domain]["spatial_mious_2"].append(mIoU_2)
        durations[duration]["spatial_mious_2"].append(mIoU_2)
        overall_stats["spatial_mious_2"].append(mIoU_2)
        data[idx]["AP2@0.1:0.9"] = aps_2
        data[idx]["spatial_mIoU_2"] = mIoU_2


        random_aps, random_mIoU = calculate_spatial_random(item['bboxes'], w, h)
        for i, ap in enumerate(random_aps):
            domains[domain]["random_aps"][i].append(ap)
            durations[duration]["random_aps"][i].append(ap)
            overall_stats["random_aps"][i].append(ap)
        domains[domain]["random_vious"].append(random_mIoU)
        durations[duration]["random_vious"].append(random_mIoU)
        overall_stats["random_vious"].append(random_mIoU)
        
        if score >= 2 and temporal_iou >= 0.3:
            domains[domain]["vqa_temporal_idx"].append(idx)
            durations[duration]["vqa_temporal_idx"].append(idx)
            overall_stats["vqa_temporal_idx"].append(idx)
        if score >= 2 and temporal_iou_2 >= 0.3:
            domains[domain]["vqa_temporal_idx_2"].append(idx)
            durations[duration]["vqa_temporal_idx_2"].append(idx)
            overall_stats["vqa_temporal_idx_2"].append(idx)
        if score >= 2 and mIoU >= 0.1:
            domains[domain]["vqa_spatial_idx"].append(idx)
            durations[duration]["vqa_spatial_idx"].append(idx)
            overall_stats["vqa_spatial_idx"].append(idx)
        if score >= 2 and mIoU_2 >= 0.1:
            domains[domain]["vqa_spatial_idx_2"].append(idx)
            durations[duration]["vqa_spatial_idx_2"].append(idx)
            overall_stats["vqa_spatial_idx_2"].append(idx)
        if temporal_iou >= 0.3 and mIoU >= 0.1:
            domains[domain]["temporal_spatial_idx"].append(idx)
            durations[duration]["temporal_spatial_idx"].append(idx)
            overall_stats["temporal_spatial_idx"].append(idx)
        if temporal_iou_2 >= 0.3 and mIoU_2 >= 0.1:
            domains[domain]["temporal_spatial_idx_2"].append(idx)
            durations[duration]["temporal_spatial_idx_2"].append(idx)
            overall_stats["temporal_spatial_idx_2"].append(idx)
        if score >= 2 and temporal_iou >= 0.3 and mIoU >= 0.1:
            domains[domain]["vqa_temp_spatial_idx"].append(idx)
            durations[duration]["vqa_temp_spatial_idx"].append(idx)
            overall_stats["vqa_temp_spatial_idx"].append(idx)
        if score >= 2 and temporal_iou_2 >= 0.3 and mIoU_2 >= 0.1:
            domains[domain]["vqa_temp_spatial_idx_2"].append(idx)
            durations[duration]["vqa_temp_spatial_idx_2"].append(idx)
            overall_stats["vqa_temp_spatial_idx_2"].append(idx)

    def print_stats(label, stats, total_samples):
        avg_all_score = np.mean(stats["all_rating"])
        avg_valid_score = np.mean(stats["valid_rating"]) if stats["valid_rating"] else 0
        acc_vqa = stats["correct_num"] / total_samples

        r1_iou30 = np.mean([1 if iou >= 0.3 else 0 for iou in stats["temporal_ious"]])
        r1_iou50 = np.mean([1 if iou >= 0.5 else 0 for iou in stats["temporal_ious"]])
        r1_iou70 = np.mean([1 if iou >= 0.7 else 0 for iou in stats["temporal_ious"]])
        mean_temporal_iou = np.mean(stats["temporal_ious"])

        r1_iou30_2 = np.mean([1 if iou >= 0.3 else 0 for iou in stats["temporal_ious_2"]])
        r1_iou50_2 = np.mean([1 if iou >= 0.5 else 0 for iou in stats["temporal_ious_2"]])
        r1_iou70_2 = np.mean([1 if iou >= 0.7 else 0 for iou in stats["temporal_ious_2"]])
        mean_temporal_iou_2 = np.mean(stats["temporal_ious_2"])

        mean_aps = [np.mean(ar_list) for ar_list in stats["spatial_aps"]]
        mean_miou = np.mean(stats["spatial_mious"])

        mean_aps_2 = [np.mean(ar_list) for ar_list in stats["spatial_aps_2"]]
        mean_miou_2 = np.mean(stats["spatial_mious_2"])


        vqa_temp = len(stats["vqa_temporal_idx"]) / total_samples
        vqa_temp_2 = len(stats["vqa_temporal_idx_2"]) / total_samples
        vqa_spat = len(stats["vqa_spatial_idx"]) / total_samples
        vqa_spat_2 = len(stats["vqa_spatial_idx_2"]) / total_samples
        temp_spat = len(stats["temporal_spatial_idx"]) / total_samples
        temp_spat_2 = len(stats["temporal_spatial_idx_2"]) / total_samples
        vqa_temp_spat = len(stats["vqa_temp_spatial_idx"]) / total_samples
        vqa_temp_spat_2 = len(stats["vqa_temp_spatial_idx_2"]) / total_samples

        print(f"{label}:")
        print(f"VQA: Avg All Score: {avg_all_score:.4f}, Avg Valid Score: {avg_valid_score:.4f}, Accuracy: {acc_vqa:.4f}")
        print("Chain 1:")
        print(f"Temporal Answer: R1@IoU=0.3: {r1_iou30:.4f}, R1@IoU=0.5: {r1_iou50:.4f}, R1@IoU=0.7: {r1_iou70:.4f}, Mean IoU: {mean_temporal_iou:.4f}")
        print(f"Spatial Answer: mAP@0.1: {mean_aps[0]:.4f}, mAP@0.3: {mean_aps[1]:.4f}, mAP@0.5: {mean_aps[2]:.4f}, mAP@0.7: {mean_aps[3]:.4f}, mAP@0.9: {mean_aps[4]:.4f}, Mean mIoU: {mean_miou:.4f}")
        print("\n")  
        print("Chain 2:")
        print(f"Temporal Answer: R1@IoU=0.3: {r1_iou30_2:.4f}, R1@IoU=0.5: {r1_iou50_2:.4f}, R1@IoU=0.7: {r1_iou70_2:.4f}, Mean IoU: {mean_temporal_iou_2:.4f}")
        print(f"Spatial Answer: mAP@0.1: {mean_aps_2[0]:.4f}, mAP@0.3: {mean_aps_2[1]:.4f}, mAP@0.5: {mean_aps_2[2]:.4f}, mAP@0.7: {mean_aps_2[3]:.4f}, mAP@0.9: {mean_aps_2[4]:.4f}, Mean mIoU: {mean_miou_2:.4f}")
        print("\n")

        AM = (acc_vqa + mean_temporal_iou + mean_miou)/3
        AM2 = (acc_vqa + mean_temporal_iou_2 + mean_miou_2)/3
        mAM = (AM + AM2) / 2

        LGM = -(math.log(1 - acc_vqa) + math.log(1 - mean_temporal_iou) + math.log(1 - mean_miou)) / 3     
        LGM2 = -(math.log(1 - acc_vqa) + math.log(1 - mean_temporal_iou_2) + math.log(1 - mean_miou_2)) / 3
        mLGM = (LGM + LGM2) / 2

        print(f"AM1:{AM:.4f}, AM2:{AM2:.4f}, mAM:{mAM:.4f}")
        print(f"LGM1:{LGM:.4f}, LGM2:{LGM2:.4f}, mLGM:{mLGM:.4f}\n")

        print("Combined resutls:")
        print(f"VQA & Temp:  Chain 1: {vqa_temp:.4f}, Chain 2: {vqa_temp_2:.4f}")
        print(f"VQA & Spat: Chain 1: {vqa_spat:.4f} Chain 2: {vqa_spat_2:.4f}")
        print(f"Temp & Spat:  Chain 1: {temp_spat:.4f} Chain 2: {temp_spat_2:.4f}")
        print(f"VQA & Temp & Spat:  Chain 1:{vqa_temp_spat:.4f} Chain 2: {vqa_temp_spat_2:.4f}")
        print(f"VQA & Temp list: \n Chain 1:{stats['vqa_temporal_idx']} \nChain 2:{stats['vqa_temporal_idx_2']}")
        print(f"VQA & Spat list: \n Chain 1:{stats['vqa_spatial_idx']} \n Chain 2: {stats['vqa_spatial_idx_2']}")
        print(f"Temp & Spat list:  \n Chain 1:{stats['temporal_spatial_idx']} \n Chain 2: {stats['temporal_spatial_idx_2']}")
        print(f"VQA & Temp & Spat list: \n Chain 1:{stats['vqa_temp_spatial_idx']} \n Chain 2:{stats['vqa_temp_spatial_idx_2']}\n")

    print_stats("Overall Statistics", overall_stats, len(data))
    for duration, stats in durations.items():
        print_stats(f"Video Length: {duration}", stats, len(stats["all_rating"]))
    for domain, stats in domains.items():
        print_stats(f"Domain: {domain}", stats, len(stats["all_rating"]))

print(result_file)
evaluate_json(result_file)