# test_time_scaling.py
import os
import re
import cv2
import json
import logging
import numpy as np

LOG = logging.getLogger("tts_video")
LOG.setLevel(logging.INFO)

pattern = r"<obj>(.*?)</obj><box>(\[.*?\])</box>at<t>(.*?)</t>s"

def parse_box(box_str):
    clean = box_str.strip().replace(" ", "")
    clean = clean.replace("[", "")
    clean = clean.replace("]", "")
    parts = clean.split(",")
    try:
        vals = [float(p) for p in parts]
    except Exception:
        return None
    if len(vals) != 4:
        return None
    x1, y1, x2, y2 = vals
    if x2 >= x1 and y2 >= y1:
        return vals
    else:
        return None


def parse_patterns(text):
    out = []
    for match in re.finditer(pattern, text, re.DOTALL):
        obj = match.group(1).strip()
        box_raw = match.group(2)
        t_raw = match.group(3).strip()
        try:
            t_sec = round(float(t_raw),2)
        except Exception:
            t_sec = None
        box_xyxy = parse_box(box_raw)
        if t_sec is not None and box_xyxy is not None:
            out.append({"obj": obj, "box_xyxy": box_xyxy, "t_sec": t_sec})
    return out

def read_frame_at_time(frames, fps, t_sec):
    # print("t_sec, fps:", t_sec, fps)
    if round(t_sec*fps) < len(frames):
        return frames[round(t_sec*fps)]
    else:
        return None

def crop_box(frame, box_xyxy):

    frame_hwc = np.transpose(frame, (1, 2, 0))  # H, W, 3
    H, W, _ = frame_hwc.shape

    x1, y1, x2, y2 = map(int, box_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    try:
        cropped = frame_hwc[y1:y2, x1:x2]
    except:
        return None

    if cropped.size == 0:
        return None

    cropped_resized = cv2.resize(cropped.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    cropped_resized = cropped_resized.astype(np.uint8)
    
    cropped_frame = np.transpose(cropped_resized, (2, 0, 1))
    return cropped_frame



def relevance_mapping(score_0_1_2):
    if score_0_1_2 == 2:
        return 1.0
    if score_0_1_2 == 1:
        return 0.6
    if score_0_1_2 == 0:
        return 0.2
    return 0.2


def extract_and_crop(frames, fps, think_info):
    image_list = []
    for i, info in enumerate(think_info):
        frame = read_frame_at_time(frames, fps, info["t_sec"])
        if frame is None:
            continue
        crop = crop_box(frame, info["box_xyxy"])
        if crop is None:
            continue
        image_list.append(crop)
    if len(image_list) > 10:
        return []
    return image_list

def build_image_scorer_msgs(images, question):
    SYSTEM = "You are a helpful assistant. Only reply with a single digit: 0, 1, or 2."
    USER_TMPL = (
        "You will be given a video question and a set of cropped images extracted from the video.\n"
        "Score how related these images are to answering the question.\n\n"
        "Scoring rules:\n"
        "2 = clearly relevant to answering the question\n"
        "1 = might be useful but uncertain\n"
        "0 = not relevant at all\n\n"
        "Only output one of: 0, 1, or 2. No other text.\n"
        "Question: {question}"
    )

    content = [{"type": "text", "text": USER_TMPL.format(question=question)}]
    for p in images:
        content.append({"type": "image", "image": p})
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]
    return messages
       