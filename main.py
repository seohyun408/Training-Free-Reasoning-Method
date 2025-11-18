import os
import json
import argparse
import torch
from datasets import load_dataset, Image as HFImage
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from PIL import Image
from pathlib import Path

from process import process_dataset_sample

def build_parser():
    p = argparse.ArgumentParser(description="Run reasoning + thought anchor extraction without shell exports.")
    p.add_argument("--dataset-name", default="Xkev/LLaVA-CoT-100k", help="HuggingFace dataset name")
    p.add_argument("--dataset-split", default="train[:50]", help="Dataset split slice, e.g. train[:50]")
    p.add_argument("--images-root", default="/mnt/hdd/llava/llava-cot-100k/extracted", help="Directory containing images")
    p.add_argument("--hf-cache", default="/mnt/hdd/llava/llava-cot-100k", help="HF datasets cache dir")
    p.add_argument("--model-cache", default=os.getenv("HF_MODEL_CACHE", "/mnt/hdd/huggingface-models"), help="HF model cache dir")
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct", help="Model name to load")
    p.add_argument("--gpu-id", type=int, default=1, help="GPU index (0-based)")
    p.add_argument("--max-samples", type=int, default=15, help="Stop after N successful samples (0=all)")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Generation max_new_tokens")
    p.add_argument("--do-sample", action="store_true", help="Enable sampling (temperature/top-p)")
    p.add_argument("--skip-suppression", action="store_true", help="Skip suppression KL computation")
    p.add_argument("--suppression-strategy", choices=["attn", "embed"], default="attn", help="Counterfactual suppression strategy")
    p.add_argument("--anchor-method", choices=["outgoing", "incoming", "combined"], default="outgoing", help="Aggregate KL for anchor vector")
    p.add_argument("--sparse-top-p", type=float, default=0.0, help="If >0 apply nucleus sparsification before KL (paper-style)")
    p.add_argument("--output-dir", default="anchor_vectors_output", help="Directory to store JSON outputs")
    p.add_argument("--enable-contrastive", action="store_true", help="Enable contrastive positive/negative sentence generation")
    p.add_argument("--contrastive-samples", type=int, default=5, help="Number of samples for contrastive generation")
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
    os.environ["ENABLE_CONTRASTIVE"] = "1"
    os.environ["CONTRASTIVE_SAMPLES"] = str(args.contrastive_samples)


def main():
    parser = build_parser()
    args = parser.parse_args()
    inject_env_from_args(args)

    DATASET_NAME = args.dataset_name
    DATASET_SPLIT = args.dataset_split
    HF_CACHE = args.hf_cache
    IMAGES_ROOT = args.images_root
    MODEL_CACHE = args.model_cache

    # Ensure datasets/model cache location
    os.environ["HF_DATASETS_CACHE"] = HF_CACHE
    os.environ.setdefault("HF_HOME", MODEL_CACHE)

    print(f"Loading dataset: {DATASET_NAME} ({DATASET_SPLIT})...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, cache_dir=HF_CACHE)
    print(f"Loaded {len(dataset)} examples")

    # Model (override with env MODEL_NAME) - use Instruct for cleaner outputs
    model_name = args.model_name
    # GPU selection
    gpu_id_env = str(args.gpu_id)
    if torch.cuda.is_available():
        try:
            gpu_index = int(gpu_id_env)
        except ValueError:
            gpu_index = 1
        if gpu_index >= torch.cuda.device_count():
            print(f"[device] Requested GPU_ID={gpu_index} exceeds available count {torch.cuda.device_count()}, falling back to 0")
            gpu_index = 0
        device = f"cuda:{gpu_index}"
    else:
        device = "cpu"
    print(f"[device] Using device: {device} (total CUDA: {torch.cuda.device_count() if torch.cuda.is_available() else 0})")
    if "cuda" in device:
        try:
            free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
            total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
            print(f"[device] GPU memory free/total: {free_mem:.2f} / {total_mem:.2f} GB")
        except Exception as e:
            print(f"[device] Could not query GPU memory: {e}")
    
    print(f"Loading model: {model_name}")
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
            print(f"[quant] 4bit quantization requested but failed to import BitsAndBytes: {e}")
    elif load_8bit:
        load_kwargs["load_in_8bit"] = True

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    load_kwargs["dtype"] = torch_dtype  # use new arg name instead of deprecated torch_dtype

    print(f"[model] Loading with kwargs: { {k: type(v) if k=='quantization_config' else v for k,v in load_kwargs.items()} }")
    # Explicit single-GPU device_map to avoid post-load .to() and accelerate offload conflict
    device_map = {"": device} if device.startswith("cuda") else None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,
        **load_kwargs
    )
    model.eval()
    tokenizer = getattr(processor, "tokenizer", None)
    
    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    use_wandb = USE_WANDB
    if use_wandb:
        try:
            wandb.init(project="qwen3vl_reasoning", config={
                "model_name": model_name,
                "dataset": DATASET_NAME,
                "split": DATASET_SPLIT,
                "hf_cache": HF_CACHE,
                "device": device,
            })
        except Exception as e:
            print(f"[wandb] init failed: {e}")
            use_wandb = False

    max_samples = args.max_samples if args.max_samples > 0 else None

    processed = 0
    for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        # Filter: only process ai2d images
        image_field = example.get('image', '')
        if isinstance(image_field, str):
            image_path = image_field
        else:
            image_path = ''

        if 'ai2d' not in image_path:
            print(f"\nSkipping example {idx+1}/{len(dataset)} (not ai2d): {image_path}")
            continue

        print(f"\nProcessing example {idx+1}/{len(dataset)}")

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
        
        result["example_idx"] = idx
        
        # Save individual result (always save for traceability)
        out_path = output_dir / f"example_{idx}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[save] Wrote: {out_path}")

        if use_wandb:
            try:
                wandb.log({
                    "example_idx": idx,
                    "total_pairs": result.get("total_pairs", 0),
                    "successful_pairs": result.get("successful_pairs", 0),
                    "num_chunks_first_pair": len(result.get("qa_pairs", [{}])[0].get("chunks", [])) if result.get("qa_pairs") else 0,
                })
            except Exception as e:
                print(f"[wandb] log failed for example {idx}: {e}")

        # Count only successful examples toward MAX_SAMPLES
        if result.get("successful_pairs", 0) > 0 and result.get("error") is None:
            processed += 1
            if max_samples is not None and processed >= max_samples:
                print(f"Reached MAX_SAMPLES={max_samples} (successful examples). Stopping early.")
                break
        else:
            print("[info] Example skipped or produced no successful pairs; not counted toward MAX_SAMPLES.")

    if use_wandb:
        try:
            wandb.finish()
        except Exception as e:
            print(f"[wandb] finish failed: {e}")
        


if __name__ == "__main__":
    main()
