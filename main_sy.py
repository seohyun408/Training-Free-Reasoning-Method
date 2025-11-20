import os
import json
import argparse
import torch
from datasets import load_dataset, Image as HFImage
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from PIL import Image
from collections import defaultdict
from multiprocessing import Process

from process import process_dataset_sample


def build_parser():
    p = argparse.ArgumentParser(description="Run reasoning + thought anchor extraction without shell exports.")
    # ===== Dataset / paths =====
    p.add_argument("--dataset-name", default="Xkev/LLaVA-CoT-100k", help="HuggingFace dataset name")
    p.add_argument("--dataset-split", default="train", help="Dataset split slice, e.g. train or train[:1000]")
    p.add_argument("--images-root", default="/mnt/hdd/llava/llava-cot-100k/extracted",
                   help="Directory containing images (root with ai2d/, coco/, ...)")
    p.add_argument("--hf-cache", default="/mnt/hdd/llava/llava-cot-100k", help="HF datasets cache dir")
    p.add_argument("--model-cache", default=os.getenv("HF_MODEL_CACHE", "/mnt/hdd/huggingface-models"),
                   help="HF model cache dir")

    # ===== Model / device =====
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct", help="Model name to load")
    p.add_argument("--gpu-id", type=int, default=0,
                   help="Base GPU index (0-based). If using multiple GPUs, this is the first id.")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Number of GPUs to use in parallel (each GPU gets a shard of the tasks). "
                        "If 0 or CUDA not available, runs on CPU.")

    # ===== Sampling / limits =====
    p.add_argument("--max-samples", type=int, default=0,
                   help="Global max number of examples to process (0 = no global limit). "
                        "Applied after per-subset sampling.")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Generation max_new_tokens")
    p.add_argument("--do-sample", action="store_true", help="Enable sampling (temperature/top-p)")

    # ===== Thought anchor / suppression options =====
    p.add_argument("--skip-suppression", action="store_true", help="Skip suppression KL computation")
    p.add_argument("--suppression-strategy", choices=["attn", "embed"], default="attn",
                   help="Counterfactual suppression strategy")
    p.add_argument("--anchor-method", choices=["outgoing", "incoming", "combined"], default="outgoing",
                   help="Aggregate KL for anchor vector")
    p.add_argument("--sparse-top-p", type=float, default=0.0,
                   help="If >0 apply nucleus sparsification before KL (paper-style)")

    # ===== Output / extra features =====
    p.add_argument("--output-dir", default="anchor_vectors_output",
                   help="Directory to store JSON outputs")
    p.add_argument("--enable-contrastive", action="store_true",
                   help="Enable contrastive positive/negative sentence generation")
    p.add_argument("--contrastive-samples", type=int, default=5,
                   help="Number of samples for contrastive generation")
    p.add_argument("--test-pca-context", action="store_true",
                   help="Test PCA context vector effect on answer accuracy")
    p.add_argument("--pca-num-trials", type=int, default=3,
                   help="Number of trials per context scale for PCA testing")

    # ===== New: per-subset control =====
    p.add_argument("--per-subset-max", type=int, default=20,
                   help="Max number of examples to process per top-level image folder "
                        "(ai2d, chartqa, coco, ...).")
    p.add_argument("--subset-prefixes", type=str, default="",
                   help="Comma-separated list of top-level folder names to include "
                        "(e.g. 'ai2d,chartqa,coco'). Empty = use all subsets found.")

    # NEW: local json file for offline dataset loading
    p.add_argument(
        "--local-json",
        type=str,
        default="",
        help="If provided, load dataset from this local JSON/JSONL file using "
             "datasets.load_dataset('json', ...) instead of the HF hub dataset-name."
    )

    return p


# Environment // API (optional)
HF_DATASETS_CACHE = os.getenv('HF_DATASETS_CACHE')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
# If you want W&B tracking set WANDB_API_KEY (preferred) or WANDB_TOKEN
WANDB_API_KEY = os.getenv('WANDB_API_KEY') or os.getenv('WANDB_TOKEN')
USE_WANDB = bool(WANDB_API_KEY)

if USE_WANDB:
    try:
        import wandb
        wandb.login(key=WANDB_API_KEY)
    except Exception as e:
        print(f"[wandb] Login skipped: {e}")
        # Do not mutate USE_WANDB to avoid scope issues; use local flag in main


def inject_env_from_args(args):
    os.environ["MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["DO_SAMPLE"] = "1" if args.do_sample else "0"
    os.environ["SKIP_SUPPRESSION"] = "1" if args.skip_suppression else "0"
    os.environ["SUPPRESSION_STRATEGY"] = args.suppression_strategy
    os.environ["ANCHOR_METHOD"] = args.anchor_method
    if args.sparse_top_p > 0:
        os.environ["SPARSE_TOP_P"] = str(args.sparse_top_p)
    # Always enable contrastive generation
    os.environ["ENABLE_CONTRASTIVE"] = "1" if args.enable_contrastive else "0"
    os.environ["CONTRASTIVE_SAMPLES"] = str(args.contrastive_samples)
    # PCA context testing
    os.environ["TEST_PCA_CONTEXT"] = "1" if args.test_pca_context else "0"
    os.environ["PCA_NUM_TRIALS"] = str(args.pca_num_trials)


# ===== Worker: runs on one GPU (or CPU) with a shard of tasks =====
def run_worker(gpu_index: Optional[int], tasks: List[Dict], args: argparse.Namespace):
    DATASET_NAME = args.dataset_name
    DATASET_SPLIT = args.dataset_split
    HF_CACHE = args.hf_cache
    IMAGES_ROOT = args.images_root
    MODEL_CACHE = args.model_cache
    model_name = args.model_name

    # Ensure datasets/model cache location
    os.environ["HF_DATASETS_CACHE"] = HF_CACHE
    os.environ.setdefault("HF_HOME", MODEL_CACHE)

    # Device selection for this worker
    if torch.cuda.is_available() and gpu_index is not None:
        if gpu_index >= torch.cuda.device_count():
            print(f"[worker] Requested GPU {gpu_index} >= available {torch.cuda.device_count()}, falling back to cuda:0")
            gpu_index = 0
        device = f"cuda:{gpu_index}"
    else:
        device = "cpu"

    rank_str = f"GPU-{gpu_index}" if "cuda" in device else "CPU"
    # print(f"[worker {rank_str}] Loading dataset: {DATASET_NAME} ({DATASET_SPLIT})...")
    # dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, cache_dir=HF_CACHE)
    # print(f"[worker {rank_str}] Loaded {len(dataset)} examples")

    # Dataset loading (HF hub vs local json)
    if getattr(args, "local_json", ""):
        local_json = args.local_json
        print(f"[worker {rank_str}] Loading LOCAL json dataset from: {local_json} "
              f"({DATASET_SPLIT})...")
        data_files = {"train": local_json}
        dataset = load_dataset(
            "json",
            data_files=data_files,
            split=DATASET_SPLIT,
            cache_dir=HF_CACHE,
        )
    else:
        print(f"[worker {rank_str}] Loading HF dataset: {DATASET_NAME} ({DATASET_SPLIT})...")
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, cache_dir=HF_CACHE)

    print(f"[worker {rank_str}] Loaded {len(dataset)} examples")

    # Optional: print GPU memory info
    if "cuda" in device:
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mem /= 1024 ** 3
            total_mem /= 1024 ** 3
            print(f"[worker {rank_str}] GPU memory free/total: {free_mem:.2f} / {total_mem:.2f} GB")
        except Exception as e:
            print(f"[worker {rank_str}] Could not query GPU memory: {e}")

    print(f"[worker {rank_str}] Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=MODEL_CACHE)
    load_kwargs = {
        "trust_remote_code": True,
        "cache_dir": MODEL_CACHE,
        "low_cpu_mem_usage": True,
    }

    # Optional quantization flags
    load_8bit = os.getenv("LOAD_8BIT", "0") in {"1", "true", "True"}
    load_4bit = os.getenv("LOAD_4BIT", "0") in {"1", "true", "True"}
    if load_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["quantization_config"] = quant_cfg
        except Exception as e:
            print(f"[worker {rank_str}][quant] 4bit quantization requested but failed to import BitsAndBytes: {e}")
    elif load_8bit:
        load_kwargs["load_in_8bit"] = True

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    load_kwargs["dtype"] = torch_dtype  # use new arg name instead of deprecated torch_dtype

    print(f"[worker {rank_str}][model] Loading with kwargs: "
          f"{ {k: type(v) if k=='quantization_config' else v for k, v in load_kwargs.items()} }")

    device_map = {"": device} if device.startswith("cuda") else None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,
        **load_kwargs
    )
    model.eval()
    tokenizer = getattr(processor, "tokenizer", None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # W&B: for simplicity, only allow when using single GPU / single worker
    use_wandb = USE_WANDB and (args.num_gpus == 1)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="qwen3vl_reasoning",
                config={
                    "model_name": model_name,
                    "dataset": DATASET_NAME,
                    "split": DATASET_SPLIT,
                    "hf_cache": HF_CACHE,
                    "device": device,
                },
                name=f"worker_{rank_str}"
            )
        except Exception as e:
            print(f"[worker {rank_str}][wandb] init failed: {e}")
            use_wandb = False

    print(f"[worker {rank_str}] Starting processing of {len(tasks)} tasks...")
    for task in tqdm(tasks, desc=f"Worker {rank_str}"):
        idx = task["dataset_idx"]
        subset = task["subset"]
        subset_idx = task["subset_local_idx"]
        image_path = task["image_path"]

        example = dataset[int(idx)]  # HF dataset supports integer indexing

        result = process_dataset_sample(
            example=example,
            model=model,
            processor=processor,
            tokenizer=tokenizer if tokenizer is not None else processor,
            device=device,
            model_name=model_name,
            images_root=IMAGES_ROOT,
            generate_reasoning=True,
        )

        result["example_idx"] = int(idx)
        result["subset"] = subset
        result["subset_local_idx"] = int(subset_idx)
        result["image_path"] = image_path

        # File name: <subset>_<local_idx>.json, e.g. ai2d_4.json
        file_name = f"{subset}_{subset_idx}.json"
        out_path = output_dir / file_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[worker {rank_str}][save] Wrote: {out_path}")

        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "example_idx": int(idx),
                    "subset": subset,
                    "subset_local_idx": int(subset_idx),
                    "total_pairs": result.get("total_pairs", 0),
                    "successful_pairs": result.get("successful_pairs", 0),
                    "num_chunks_first_pair": len(result.get("qa_pairs", [{}])[0].get("chunks", []))
                    if result.get("qa_pairs") else 0,
                })
            except Exception as e:
                print(f"[worker {rank_str}][wandb] log failed for example {idx}: {e}")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception as e:
            print(f"[worker {rank_str}][wandb] finish failed: {e}")

    print(f"[worker {rank_str}] Finished.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    inject_env_from_args(args)

    DATASET_NAME = args.dataset_name
    DATASET_SPLIT = args.dataset_split
    HF_CACHE = args.hf_cache

    # Ensure datasets cache location for the enumeration step
    os.environ["HF_DATASETS_CACHE"] = HF_CACHE

    # print(f"Loading dataset for task enumeration: {DATASET_NAME} ({DATASET_SPLIT})...")
    # dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, cache_dir=HF_CACHE)
    # print(f"Loaded {len(dataset)} examples")

    # Dataset loading for task enumeration (HF hub vs local json)
    if args.local_json:
        local_json = args.local_json
        print(f"Loading LOCAL json dataset for task enumeration from: {local_json} "
              f"({DATASET_SPLIT})...")
        data_files = {"train": local_json}
        dataset = load_dataset(
            "json",
            data_files=data_files,
            split=DATASET_SPLIT,
            cache_dir=HF_CACHE,
        )
    else:
        print(f"Loading HF dataset for task enumeration: {DATASET_NAME} ({DATASET_SPLIT})...")
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, cache_dir=HF_CACHE)

    print(f"Loaded {len(dataset)} examples")

    # Decide which subsets (top-level folders) to include
    subset_filter = None
    if args.subset_prefixes.strip():
        subset_filter = {s.strip() for s in args.subset_prefixes.split(",") if s.strip()}
        print(f"[main] Restricting to subsets: {sorted(subset_filter)}")
    else:
        print("[main] Using all subsets (top-level folders found in image paths).")

    per_subset_max = args.per_subset_max
    per_subset_counts = defaultdict(int)
    tasks = []

    # Build task list: each task has dataset index + subset name + local index
    for idx, example in enumerate(tqdm(dataset, desc="Enumerating examples")):
        image_field = example.get("image", "")
        if isinstance(image_field, str):
            image_path = image_field
        else:
            # If image is already an Image object or something else, skip (unlikely in this dataset)
            continue

        parts = image_path.split("/")
        subset = parts[0] if parts else "unknown"

        if subset_filter and subset not in subset_filter:
            continue

        if per_subset_counts[subset] >= per_subset_max:
            continue

        local_idx = per_subset_counts[subset]
        per_subset_counts[subset] += 1

        tasks.append({
            "dataset_idx": idx,
            "subset": subset,
            "subset_local_idx": local_idx,
            "image_path": image_path,
        })

    print(f"[main] Built {len(tasks)} tasks across {len(per_subset_counts)} subsets.")
    for s, c in sorted(per_subset_counts.items()):
        print(f"  - {s}: {c} examples")

    # Apply global max-samples if requested
    if args.max_samples > 0 and len(tasks) > args.max_samples:
        tasks = tasks[:args.max_samples]
        print(f"[main] Truncated tasks to first max_samples={args.max_samples}")

    # ===== Device / multi-GPU dispatch =====
    num_available = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_available == 0 or args.num_gpus <= 0:
        print("[main][device] No CUDA or num_gpus <= 0. Running on CPU only.")
        run_worker(gpu_index=None, tasks=tasks, args=args)
        return

    # We use GPUs [gpu_id, gpu_id + num_to_use - 1], clipped to available range
    max_usable = max(0, num_available - args.gpu_id)
    if max_usable <= 0:
        print(f"[main][device] gpu-id {args.gpu_id} >= device_count {num_available}. Using GPU 0 only.")
        run_worker(gpu_index=0, tasks=tasks, args=args)
        return

    num_to_use = min(args.num_gpus, max_usable)
    if num_to_use == 1:
        gpu_id = args.gpu_id
        print(f"[main][device] Using single GPU: {gpu_id}")
        run_worker(gpu_index=gpu_id, tasks=tasks, args=args)
        return

    gpu_ids = list(range(args.gpu_id, args.gpu_id + num_to_use))
    print(f"[main][device] Using multiple GPUs: {gpu_ids}")

    # Split tasks round-robin across GPUs
    shards: List[List[Dict]] = [[] for _ in gpu_ids]
    for i, task in enumerate(tasks):
        shards[i % len(gpu_ids)].append(task)

    procs: List[Process] = []
    for gpu_id, shard in zip(gpu_ids, shards):
        p = Process(target=run_worker, args=(gpu_id, shard, args), daemon=False)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("[main] All workers finished.")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
