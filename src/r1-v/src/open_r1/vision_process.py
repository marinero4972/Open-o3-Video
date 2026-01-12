from __future__ import annotations

import base64
import hashlib
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Optional

logger = logging.getLogger(__name__)

# ============== VIDEO CACHING CONFIGURATION ==============
# Persistent cache on workspace (survives between jobs)
VIDEO_CACHE_DIR = os.environ.get(
    "VIDEO_CACHE_DIR",
    "/hkfs/work/workspace/scratch/uzivy-open-o3-data/video_cache"
)
VIDEO_CACHE_ENABLED = os.environ.get("VIDEO_CACHE_ENABLED", "1") == "1"
VIDEO_CACHE_JPEG_QUALITY = int(os.environ.get("VIDEO_CACHE_JPEG_QUALITY", "85"))

# Create cache directory if caching is enabled
if VIDEO_CACHE_ENABLED:
    os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)


def _get_cache_path(video_path: str, ele: dict) -> str:
    """Generate a unique cache path for a video based on its path and config."""
    # Include relevant config in hash to handle different sampling params
    config_str = f"{ele.get('nframes', '')}{ele.get('fps', '')}{ele.get('min_frames', '')}{ele.get('max_frames', '')}"
    hash_input = f"{video_path}:{config_str}"
    hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]
    # Use video filename + hash for easier debugging
    video_name = Path(video_path).stem
    return os.path.join(VIDEO_CACHE_DIR, f"{video_name}_{hash_id}.cache")


def _save_video_cache(cache_path: str, video: torch.Tensor, sample_fps: float):
    """Save video frames as compressed JPEGs to cache."""
    try:
        frames_data = []
        for frame in video:  # video shape: (T, C, H, W)
            # Convert to PIL Image
            frame_np = frame.permute(1, 2, 0).numpy().astype('uint8')
            img = Image.fromarray(frame_np)
            # Compress to JPEG
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=VIDEO_CACHE_JPEG_QUALITY)
            frames_data.append(buf.getvalue())

        cache_data = {
            'frames': frames_data,
            'sample_fps': sample_fps,
            'shape': list(video.shape),
        }
        torch.save(cache_data, cache_path)
        logger.info(f"Cached video to {cache_path} ({len(frames_data)} frames, {os.path.getsize(cache_path) / 1024:.1f} KB)")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")


def _load_video_cache(cache_path: str) -> tuple[torch.Tensor, float] | None:
    """Load video frames from compressed cache."""
    try:
        cache_data = torch.load(cache_path, weights_only=False)
        frames = []
        for jpg_bytes in cache_data['frames']:
            img = Image.open(BytesIO(jpg_bytes))
            frame = torch.tensor(list(img.getdata()), dtype=torch.uint8)
            frame = frame.reshape(img.height, img.width, 3).permute(2, 0, 1)
            frames.append(frame)
        video = torch.stack(frames)
        return video, cache_data['sample_fps']
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None
# ============== END VIDEO CACHING ==============


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
MAX_RATIO = 200

# VIDEO_MIN_PIXELS = 128 * 28 * 28
# VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 128 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 16

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def _load_image_sequence_from_dir(dir_path: str, ele: dict) -> tuple[torch.Tensor, float]:
    """Load image sequence from a directory as a video tensor."""
    import glob
    import re

    # Try common frame naming patterns
    patterns = ["im_*.jpg", "im_*.png", "frame_*.jpg", "frame_*.png", "*.jpg", "*.png"]
    image_files = []

    for pattern in patterns:
        image_files = glob.glob(os.path.join(dir_path, pattern))
        if image_files:
            break

    if not image_files:
        raise ValueError(f"No image files found in directory: {dir_path}")

    # Sort numerically by extracting number from filename
    def extract_number(path):
        basename = os.path.basename(path)
        numbers = re.findall(r'\d+', basename)
        return int(numbers[-1]) if numbers else 0

    image_files = sorted(image_files, key=extract_number)

    total_frames = len(image_files)
    # Assume default FPS for image sequences (configurable via ele)
    video_fps = ele.get("source_fps", 10.0)

    # Calculate number of frames to sample using smart_nframes logic
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    # Sample frames uniformly
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    sampled_files = [image_files[i] for i in idx]

    # Load images and stack into tensor (T, C, H, W)
    frames = []
    for img_path in sampled_files:
        img = Image.open(img_path).convert("RGB")
        frame = torch.tensor(list(img.getdata()), dtype=torch.uint8)
        frame = frame.reshape(img.height, img.width, 3).permute(2, 0, 1)
        frames.append(frame)

    video = torch.stack(frames)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    return video, sample_fps


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_path = ele["video"]
        is_directory = os.path.isdir(video_path)
        cache_path = _get_cache_path(video_path, ele) if VIDEO_CACHE_ENABLED else None

        # Try loading from cache first
        cached_result = None
        if cache_path and os.path.exists(cache_path):
            cached_result = _load_video_cache(cache_path)

        if cached_result is not None:
            video, sample_fps = cached_result
            logger.info(f"Loaded video from cache: {cache_path}")
        else:
            if is_directory:
                # Load from image sequence directory
                video, sample_fps = _load_image_sequence_from_dir(video_path, ele)
            else:
                # Decode video (slow path)
                video_reader_backend = get_video_reader_backend()
                try:
                    video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
                except Exception as e:
                    logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
                    video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

            # Save to cache for next time (before resize, after frame sampling)
            if cache_path:
                _save_video_cache(cache_path, video, sample_fps)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    image_val = ele.get("image") or ele.get("image_url")
                    video_val = ele.get("video")
                    if image_val or video_val:
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        image_val = vision_info.get("image") or vision_info.get("image_url")
        video_val = vision_info.get("video")
        if image_val:
            image_inputs.append(fetch_image(vision_info))
        elif video_val:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs
