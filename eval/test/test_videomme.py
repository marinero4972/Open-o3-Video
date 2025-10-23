import os

os.environ["DECORD_EOF_RETRY_MAX"] = "40960"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import json
from datetime import datetime
from dataloader.videomme import (
    VideoMME_Bench,
    videomme_aggregate_results,
    videomme_process_results_new,
    parse_answer
)
import argparse
import multiprocessing
from tts import parse_patterns, extract_and_crop, build_image_scorer_msgs, relevance_mapping

def parse_args():
    parser = argparse.ArgumentParser(description="Model Configuration Parameters")

    # Basic Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/path/to/Video-MME",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/path/to/your/model",
        help="Path to the model.",
    )
    parser.add_argument(
        "--asr_dir",
        type=str,
        default="/path/to/your/asr_data",
        help="Path to the ASR data directory.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="debug", help="Experiment name."
    )

    # Model Configuration
    parser.add_argument("--models_per_gpu", type=int, default=1)
    parser.add_argument(
        "--disable_asr",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default=None,
        help="Path to YAML file containing model keyword arguments.",
    )
    parser.add_argument(
        "--vote", type=str, default="confidence_voting", help="Experiment name."
    )
    parser.add_argument("--think_mode", action="store_true", help="Use Chain-of-Thought prompting.")
    parser.add_argument("--N", type=int, default=8, help="number of paths")

    args = parser.parse_args()

    try:
        import yaml

        with open(args.model_kwargs, "r") as f:
            args.model_kwargs = yaml.safe_load(f)
        if not isinstance(args.model_kwargs, dict):
            raise ValueError("YAML file must contain a dictionary")
    except ImportError:
        parser.error(
            "PyYAML is required for YAML parsing. Install with: pip install pyyaml"
        )
    except FileNotFoundError:
        parser.error(f"Model kwargs file not found: {args.model_kwargs}")
    except Exception as e:
        parser.error(f"Error parsing YAML file: {e}")
    return args


def get_cuda_visible_devices():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible_devices:
        return []
    gpu_list = [
        int(gpu_id.strip())
        for gpu_id in cuda_visible_devices.split(",")
        if gpu_id.strip()
    ]
    return gpu_list


def build_model(
    model_path,
    temperature,
    max_tokens,
    video_max_pixels,
    video_max_frames,
):
    from models.model_vllm import QwenVL_VLLM

    model = QwenVL_VLLM(
        model_path,
        rt_shape=True,
        temperature=temperature,
        max_tokens=max_tokens,
        video_max_pixels=video_max_pixels,
        video_max_frames=video_max_frames,
    )

    return model


def evaluate_chunk(video_paths, image, text_input, docs, gpu_id, args, queue):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {str(gpu_id)}] Processing {len(video_paths)} examples.")

        results = []
        metrics = []

        bs = 1  # Batch size
        model = build_model(args.model_path, **args.model_kwargs)

        think_contents = []
        frame_shape_list = []

        def process_batch(batch_video_paths, batch_text_input, batch_image, batch_doc):

            N = args.N
            score_list = []
            pred_list = []
            n_think = ["" for _ in range(N)]
            frame_shape = None

            for path_idx in range(N):
                think_mode = args.think_mode

                output_list, frames, fps, shape = model(
                    batch_video_paths,
                    batch_text_input,
                    query_image=batch_image,
                )
                request_output = output_list[0]
                completions = request_output.outputs
                pred_text = completions[0].text
                frame_shape = shape

                # print(pred_text)

                if think_mode:
                    import re
                    match = re.search(r"<answer>(.*?)</answer>", pred_text, re.DOTALL)
                    ans = None
                    if match:
                        ans = match.group(1).strip()
                        if ans not in ["A","B","C","D"]:
                            score = 0
                            pred_list.append("NA")
                            score_list.append(score)
                            print("exit since pred_text not ABCD:", ans)
                            continue
                        else:
                            pred_list.append(ans)
                    else:
                        score = 0
                        pred_list.append("NA")
                        score_list.append(score)
                        print("exit since pred_text not match:", pred_text)
                        continue
                    
                    match = re.search(r"<think>(.*?)</think>", pred_text, re.DOTALL)
                    if match:
                        think_process = match.group(1).strip()
                        n_think[path_idx] = think_process
                    else:
                        score_list.append(0)
                        print("exit since think not match:", ans)
                        continue

                    # majority voting
                    if args.vote == "majority_voting":
                        score = 1.0
                    elif args.vote == "confidence_voting":
                        # confidence-based voting
                        think_info = parse_patterns(think_process)
                        # print(think_info)
                        if think_info is not None:
                            image_list = extract_and_crop(frames, fps, think_info)
                            # print(image_list)
                            if len(image_list) > 0:
                                doc = batch_doc[0]
                                question = doc["question"]
                                option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
                                question = question + "\n" + option
                                msgs = build_image_scorer_msgs(image_list, question)
                                score = relevance_mapping(model.run_images_scorer(msgs, image_list))
                            else:
                                score = 0.2
                            
                            # print("score:",score)
                            if score == 1.0:
                                outstr = "Example start\nThink process:" + think_process + "\nAnswer:" + ans + "\nQuestion:" +  question + "\nVideoID:" + batch_doc[0]["videoID"] +"\nExample end."
                                # print(outstr)
                    else:
                        score = 0.2
                    
                else:
                    ans = parse_answer(pred_text, batch_doc[0])
                    if ans in ["A","B","C","D"]:
                        pred_list.append(ans)
                        score = 1.0
                    else:
                        pred_list.append("NA")
                        score = 0.0
                score_list.append(score) 

            print(pred_list, score_list)
            choice_score = {"A": 0, "B": 0, "C": 0, "D": 0 }
            for i in range(N):
                if pred_list[i] == "NA":
                    continue
                choice_score[pred_list[i]] += score_list[i]
            print("Choice Score:", choice_score)
            pred_text = max(choice_score, key=choice_score.get)
 
            print(
                batch_doc[0]["videoID"],
                "GT:",
                batch_doc[0]["answer"],
                "Pred:",
                pred_text,
            )

            think_text = ""

            for idx in range(len(pred_list)):
                if pred_list[idx]==pred_text:
                    target_idx = idx
                    think_text = n_think[idx]
                    break
            
            results.append(pred_text)
            think_contents.append(think_text)
            frame_shape_list.append(frame_shape)

        idx = 0
        while idx < len(video_paths):
            batch_size = min(bs, len(video_paths) - idx)
            batch_video_paths = video_paths[idx : idx + batch_size]
            batch_text_input = text_input[idx : idx + batch_size]
            batch_image = image[idx : idx + batch_size]
            batch_doc = docs[idx : idx + batch_size]

            process_batch(batch_video_paths, batch_text_input, batch_image, batch_doc)
            idx += batch_size
            print(f"GPU ID:{gpu_id},{idx}/{len(video_paths)}")

        metrics = [
            videomme_process_results_new(docs[i], results[i], think_contents[i], frame_shape_list[i]) for i in range(len(docs))
        ]
        queue.put((metrics, results, None))
        print(f"[GPU {gpu_id}] Finished processing.")
    except Exception as e:
        import traceback

        error_msg = traceback.format_exc()
        queue.put((None, None, error_msg))


def evaluate(args, num_gpus, gpu_list):

    videomme = VideoMME_Bench(
        args.data_dir, add_asr=not args.disable_asr, asr_dir=args.asr_dir, think_mode=args.think_mode
    )

    if len(gpu_list) == 0:
        gpu_list = list(range(num_gpus))

    video_paths, image, text_input, docs = videomme.get_data()
    total = len(video_paths)

    models_per_gpu = args.models_per_gpu
    gpu_list_new = [
        gpu_list[i] for i in range(len(gpu_list)) for j in range(models_per_gpu)
    ]
    gpu_list = gpu_list_new

    print(f"Total examples: {total}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"GPU list: {gpu_list}")

    num_gpus = num_gpus * models_per_gpu

    chunk_size = (
        total + num_gpus - 1
    ) // num_gpus  # ceiling division to cover all examples
    chunks = []

    for i in range(num_gpus):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        if start >= end:
            break
        chunks.append(
            (
                video_paths[start:end],
                image[start:end],
                text_input[start:end],
                docs[start:end],
            )
        )

    queue = multiprocessing.Queue()
    processes = []

    for i, (vp_chunk, img_chunk, txt_chunk, docs_chunk) in enumerate(chunks):
        p = multiprocessing.Process(
            target=evaluate_chunk,
            args=(vp_chunk, img_chunk, txt_chunk, docs_chunk, gpu_list[i], args, queue),
        )
        p.start()
        processes.append(p)

    # Collect the results from each process.
    all_metrics = []
    all_results = []
    for _ in processes:
        metrics, results, error = queue.get()
        if error is not None:
            print(f"子进程出错: {error}")
            for p in processes:
                p.terminate()
            exit(1)
        all_metrics.extend(metrics)
        all_results.extend(results)

    for p in processes:
        p.join()

    acc = videomme_aggregate_results(all_metrics)
    print("Final accuracy:", acc)

    queue.close()

    return all_metrics, all_results


def main():
    print("Start Time:", datetime.now())
    args = parse_args()
    print(args)

    num_gpus = int(os.getenv("NUM_GPUS"))
    gpu_list = get_cuda_visible_devices()

    metrics, results = evaluate(args, num_gpus=num_gpus, gpu_list=gpu_list)

    metrics_path = f"./logs/videomme_logs/metrics_{args.exp_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    results_path = f"./logs/videomme_logs/results_{args.exp_name}.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    print("End Time:", datetime.now())


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
