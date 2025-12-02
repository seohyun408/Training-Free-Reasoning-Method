import os
import json
import wandb
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import pdb

from datasets import load_dataset, Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from transformers import BitsAndBytesConfig
from PIL import Image
from pathlib import Path

from process import DatasetProcessor


HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
HF_HOME = os.getenv('HF_HOME')
DATASETS_CACHE = os.getenv('HF_DATASETS_CACHE')
MODEL_CACHE = os.getenv('HF_MODEL_CACHE')
IMAGES_ROOT = os.getenv('IMAGES_ROOT')

WANDB_TOKEN = os.getenv('WANDB_TOKEN')
USE_WANDB = bool(WANDB_TOKEN)


def build_parser():

    p = argparse.ArgumentParser(description="🔥Team Galaxy🔥")
    
    ## DATASET
    p.add_argument("--dataset_name", default="Xkev/LLaVA-CoT-100k", help="HuggingFace dataset name")
    p.add_argument("--dataset_split", default="train[:100]", help="Dataset split slice, e.g. train[:50]")
    
    ## MODEL
    p.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct", help="Model name to load")
    p.add_argument("--load_4bit", action="store_true", help="Enable 4-bit quantization")
    p.add_argument("--load_8bit", action="store_true", help="Enable 8-bit quantization")

    ## Setting
    p.add_argument("--max_new_tokens", type=int, default=256, help="Generation max_new_tokens")
    p.add_argument("--do-sample", action="store_true", default=True, help="Enable sampling (temperature/top-p) (default: enabled)")
    p.add_argument("--no-sampling", dest="do_sample", action="store_false", help="Disable sampling (use greedy decoding)")
    p.add_argument("--max_attempts", type=int, default=2, help="Attempt loop with fallback prompts")
    p.add_argument("--contrastive_samples", type=int, default=5, help="Number of samples for contrastive generation")
    p.add_argument("--output_dir", default="outputs", help="Directory to save contrastive results")
    p.add_argument("--output_dir2", default="outputs2", help="Directory to save PCA results")

    ## ENVIRONMENT (etc)
    p.add_argument("--images_root", default="/scratch/tjgus0408/huggingface/datasets/LLaVA-CoT-100k", help="Directory containing images")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (0-based)")
    p.add_argument('--debug', action='store_true')

    return p


def main():

    parser = build_parser()
    args = parser.parse_args()

    os.environ["DO_SAMPLE"] = "1" if args.do_sample else "0"

    if USE_WANDB:
        wandb.login(key=WANDB_TOKEN)
        wandb.init(
            project="CMU_Project_NLP", 
            name = "qwen3vl_reasoning",
            config=args,
            reinit=True
        )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


    print(f"\n>>>>>> Loading Dataset: {args.dataset_name}")
    dataset = load_dataset("Xkev/LLaVA-CoT-100k", split="train", cache_dir=DATASETS_CACHE)
    dataset = dataset.filter(lambda x: 'coco' in x.get('image', '').lower())
    print(f"Loaded {len(dataset)} examples")


    print(f"\n>>>>>> Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
                    args.model_name,
                    trust_remote_code=True, 
                    cache_dir=MODEL_CACHE
                )
    load_kwargs = {
        "trust_remote_code": True,
        "cache_dir": MODEL_CACHE,
        "low_cpu_mem_usage": True,
    }
    if args.load_4bit:
        quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
        load_kwargs["quantization_config"] = quant_cfg
    elif args.load_8bit:
        load_kwargs["load_in_8bit"] = True
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    load_kwargs["dtype"] = torch_dtype 

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map=None,  #"auto",
        **load_kwargs
    )
    model.eval()

    model = model.to(device)
    tokenizer = getattr(processor, "tokenizer", None)

    print(f"\n>>>>>> Start Processing")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_dir2 = Path(args.output_dir2)
    output_dir2.mkdir(exist_ok=True)

    proc = DatasetProcessor(
        args=args,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        images_root=args.images_root,
        generate_reasoning=True,
        model_name=args.model_name
    )

    cnt = 0
    for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        print(f"\nProcessing example {idx+1}/{len(dataset)}")
        result = proc.process_sample(example)
        
        result["example_idx"] = idx

        results = result["qa_pairs"]
        print("+"*10)
        cnt += 1
        print(cnt)
        print("+"*10)

        with open(output_dir / f"example_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
        # 여기서 Context Vector 만들 데이터셋 갯수 조절하세요 ! ><
        if cnt > 3:
            print(cnt)
            break


    print(f"\n>>>>>> Start Vector Generation")
    pca_data = np.load('pca_data/vector_generation.npy', allow_pickle=True)
    context_vector = proc.process_pca(pca_data)


    print(f"\n>>>>>> Evaluation with Context Vector")
    for idx, example in enumerate(tqdm(dataset, desc="Evaluation")):
        pca_results = proc.evaluate_with_context_vector(context_vector, example)

        with open(output_dir2 / f"example_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(pca_results, f, indent=2, ensure_ascii=False)
    
    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
