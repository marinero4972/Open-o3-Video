#!/usr/bin/env python3
import cv2
import re
from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np
import ast
import os

class VideoQAGIFGenerator:
    """
    Generates a GIF demonstrating a video-based QA process.
    """
    def __init__(self, font_path=None, font_size=16, box_color=(255, 50, 50)):
        self.box_color = box_color
        self.text_color = 'black'
        self.highlight_colors = [
            (23, 115, 233),  # Blue
            (217, 30, 24),   # Red
            (240, 173, 78),  # Orange
            (40, 167, 69),   # Green
            (108, 117, 125)  # Grey
        ]
        self.bg_color = 'white'
        self.font_regular = ImageFont.load_default(size=font_size)
        self.font_bold = self.font_regular
        

    def _create_text_panel(self, steps_to_draw, width, height):
        panel = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(panel)
        
        margin = 15
        x, y = margin, 10

        try:
            ascent, descent = self.font_regular.getmetrics()
            line_height = ascent + descent + 5
        except AttributeError:
            line_height = self.font_regular.getbbox("A")[3] + 5
        
        space_width = draw.textlength(" ", font=self.font_regular)

        for step_type, content, *extra_info in steps_to_draw:
            text_to_draw, color, font = "", self.text_color, self.font_regular
            is_bold_header = False

            if step_type == 'header':
                text_to_draw = content.lstrip('\n')
                if content.startswith('\n') and x > margin:
                    x = margin
                    y += line_height
                
                # 检查并分离粗体标题
                for header in ["Question:", "Reasoning Process:", "Answer:"]:
                    if text_to_draw.startswith(header):
                        is_bold_header = True
                        header_width = draw.textlength(header, font=self.font_bold)
                        if x + header_width > width - margin:
                            x, y = margin, y + line_height
                        draw.text((x, y), header, font=self.font_bold, fill=self.text_color)
                        x += header_width
                        text_to_draw = text_to_draw[len(header):]
                        break

            elif step_type == 'word':
                text_to_draw = content
            elif step_type == 'action':
                text_to_draw = extra_info[0]
                color = extra_info[1] if len(extra_info) > 1 else self.highlight_colors[0]

            words = text_to_draw.split(' ')
            for i, word in enumerate(words):
                if not word: continue
                word_width = draw.textlength(word, font=font)
                if x + word_width > width - margin:
                    x, y = margin, y + line_height
                if y + line_height > height: return panel
                
                draw.text((x, y), word, font=font, fill=color)
                x += word_width + (space_width if i < len(words) - 1 else 0)
            
            x += space_width if not is_bold_header else 0
        return panel
    
    def _draw_boxes_on_frame(self, frame_pil, objects):
        draw = ImageDraw.Draw(frame_pil)
        for obj in objects:
            box = obj['box']
            name = obj['name']
            draw.rectangle(box, outline=self.box_color, width=3)
            
            text_bbox = self.font_regular.getbbox(name)
            text_height = text_bbox[3] - text_bbox[1]
            text_position = (box[0] + 2, box[1] - text_height - 6)
            
            if text_position[1] < 0: text_position = (box[0] + 2, box[1] + 2)
            
            text_width = self.font_regular.getlength(name)
            bg_bbox = [text_position[0]-2, text_position[1]-2, text_position[0] + text_width + 2, text_position[1] + text_height + 2]
            draw.rectangle(bg_bbox, fill=self.box_color)
            draw.text(text_position, name, font=self.font_regular, fill=self.bg_color)
        return frame_pil

    def _draw_timestamp_and_progress(self, frame_pil, current_time, total_duration):
        draw = ImageDraw.Draw(frame_pil)
        width, height = frame_pil.size
        bar_height, bar_y_start = 8, height - 8
        progress_ratio = current_time / total_duration if total_duration > 0 else 0
        progress_width = int(progress_ratio * width)
        
        draw.rectangle([0, bar_y_start, width, height], fill=(80, 80, 80))
        draw.rectangle([0, bar_y_start, progress_width, height], fill=self.box_color)
        
        time_text = f"{current_time:.1f}s / {total_duration:.1f}s"
        try:
            time_font = self.font_regular.font_variant(size=max(10, self.font_regular.size - 2))
        except TypeError:
            time_font = self.font_regular
        
        text_width, text_bbox = time_font.getlength(time_text), time_font.getbbox(time_text)
        text_height = text_bbox[3] - text_bbox[1]
        text_x, text_y = 10, bar_y_start - text_height - 8
        
        draw.rectangle([text_x-3, text_y-3, text_x + text_width + 3, text_y + text_height + 3], fill=(0,0,0))
        draw.text((text_x, text_y), time_text, font=time_font, fill=(255,255,255))
        
        return frame_pil

    def _parse_action_tag(self, tag_string):
        obj_pattern = re.search(r"<obj>(.*?)</obj><box>(\[.*?\])</box>at<t>(\d+\.?\d*)</t>s", tag_string)
        if obj_pattern:
            try:
                name, box_str, time_str = obj_pattern.groups()
                return {'time': float(time_str), 'objects': [{'name': name, 'box': ast.literal_eval(box_str)}]}
            except (ValueError, SyntaxError): return None
        time_pattern = re.search(r"<t>(\d+\.?\d*)</t>", tag_string)
        if time_pattern:
            return {'time': float(time_pattern.group(1)), 'objects': []}
        return None

    def _build_step_list(self, text):
        steps = []
        pattern = re.compile(r"(<obj>.*?<\/obj><box>.*?<\/box>at<t>.*?<\/t>s|<t>.*?<\/t>)")
        segments = pattern.split(text)
        for segment in segments:
            if not segment: continue
            action = self._parse_action_tag(segment)
            if action:
                steps.append(('action', action, segment.strip()))
            else:
                words = segment.strip().split()
                for word in words:
                    steps.append(('word', word))
        return steps
    
    def create_demo_gif(self, video_path, question, reasoning, answer, output_path,
                        target_size=(640, 360), gif_fps=15, frames_per_word=2):
        if not os.path.exists(video_path): raise FileNotFoundError(f"视频文件未找到: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"无法打开视频文件: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = frame_count / video_fps if video_fps > 0 else 0
        
        print(f"视频总时长: {total_duration:.2f}s, GIF帧率: {gif_fps} FPS, 文字速度: {frames_per_word} 帧/词")

        video_w, video_h = target_size
        text_panel_h = 250
        total_w, total_h = video_w, video_h + text_panel_h
        gif_frames = []

        ret, frame = cap.read()
        if not ret: raise ValueError("无法读取视频的第一帧。")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame_pil = Image.fromarray(frame_rgb).resize((video_w, video_h), Image.Resampling.LANCZOS)

        question_header_steps = [('header', f'Question: {question}')]
        text_panel = self._create_text_panel(question_header_steps, total_w, text_panel_h)
        combined_img = Image.new('RGB', (total_w, total_h))
        combined_img.paste(current_frame_pil, (0, 0))
        combined_img.paste(text_panel, (0, video_h))
        for _ in range(int(gif_fps * 2.5)): gif_frames.append(np.array(combined_img))

        def add_colors_to_steps(steps, start_index=0):
            colored_steps, color_idx = [], start_index
            for step in steps:
                if step[0] == 'action':
                    color = self.highlight_colors[color_idx % len(self.highlight_colors)]
                    colored_steps.append((*step, color))
                    color_idx += 1
                else:
                    colored_steps.append(step)
            return colored_steps, color_idx
        
        reasoning_steps, color_index = add_colors_to_steps(self._build_step_list(reasoning))
        answer_steps, _ = add_colors_to_steps(self._build_step_list(answer), color_index)

        all_steps = question_header_steps.copy()
        all_steps.append(('header', '\nReasoning Process: '))
        all_steps.extend(reasoning_steps)
        all_steps.append(('header', '\nAnswer: '))
        all_steps.extend(answer_steps)

        # print(all_steps)
        
        for i in range(len(all_steps)):
            current_display_list = all_steps[:i+1]
            step_type, content, *_ = all_steps[i]
            frame_to_show, pause_duration = current_frame_pil, 0
            
            if step_type == 'action':
                action_data = content
                current_time = action_data['time']
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    current_frame_pil = Image.fromarray(frame_rgb).resize((video_w, video_h), Image.Resampling.LANCZOS)
                    frame_to_show = current_frame_pil.copy()
                    if action_data['objects']: frame_to_show = self._draw_boxes_on_frame(frame_to_show, action_data['objects'])
                    frame_to_show = self._draw_timestamp_and_progress(frame_to_show, current_time, total_duration)
                else:
                    print(f"警告: 无法跳转到视频的 {current_time}秒。")
                pause_duration = 1.5

            text_panel = self._create_text_panel(current_display_list, total_w, text_panel_h)
            combined_img = Image.new('RGB', (total_w, total_h))
            combined_img.paste(frame_to_show, (0, 0))
            combined_img.paste(text_panel, (0, video_h))
            
            # --- 核心改动：使用 frames_per_word 控制速度 ---
            typing_frames = frames_per_word if step_type != 'header' else 1
            pause_frames = int(gif_fps * pause_duration)
            num_frames_to_add = typing_frames + pause_frames

            for _ in range(num_frames_to_add): gif_frames.append(np.array(combined_img))

        for _ in range(int(gif_fps * 4)):
            if gif_frames: gif_frames.append(gif_frames[-1])

        scale = 2
        gif_frames = [cv2.resize(frame, (total_w*scale, total_h*scale), 
                         interpolation=cv2.INTER_LINEAR) for frame in gif_frames]

        print(f"正在保存包含 {len(gif_frames)} 帧的GIF文件到 {output_path}...")
        imageio.mimsave(output_path, gif_frames, fps=gif_fps, loop=0)
        print(f"✅ GIF文件已成功保存到 '{output_path}'")
        cap.release()

def convert_coord_format_gemini(coords, image_size):
    norm_x_min, norm_y_min, norm_x_max, norm_y_max = coords
    width, height = image_size
    real_x_min = norm_x_min * width
    real_y_min = norm_y_min * height
    real_x_max = norm_x_max * width
    real_y_max = norm_y_max * height
    return [real_x_min, real_y_min, real_x_max, real_y_max]

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


if __name__ == '__main__':

    import json
    import os
    import re
    import shutil

    font_size = 14
    root = "..."
    vid = "..."
    image_size = (672, 364)  # (w, h)  # 设置可视化demo时的长宽比
    question = "..."
    answer = "..."
    think = "..."
    
    video_path = vid + ".mp4"
    os.makedirs('./output_gif', exist_ok=True)
    video_file_path = os.path.join(root, video_path)
    item_id = "example_" + vid
    out_path = f"./output_gif/demo_{item_id}.gif"

    try:
        generator = VideoQAGIFGenerator(font_size=font_size)  
        generator.create_demo_gif(
            video_path=video_file_path,
            question=question,
            reasoning=think,
            answer=answer,
            output_path=out_path,
            target_size=image_size,
            gif_fps=12,
            frames_per_word=3,
        )
    except Exception as e:
        print(f"发生错误: {e}")
