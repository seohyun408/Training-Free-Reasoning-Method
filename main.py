import os
import json
import wandb
import argparse
import numpy as np
import torch
import torch.nn.functional as F

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


## Environment Setting
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

    ## DIRECTORY // ENVIRONMENT
    p.add_argument("--images_root", default="/scratch/tjgus0408/huggingface/datasets/LLaVA-CoT-100k", help="Directory containing images")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (0-based)")

    ## 아직 수정중
    p.add_argument("--max-samples", type=int, default=10, help="Stop after N successful samples (0=all)")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Generation max_new_tokens")
    p.add_argument("--do-sample", action="store_true", default=True, help="Enable sampling (temperature/top-p) (default: enabled)")
    p.add_argument("--no-sampling", dest="do_sample", action="store_false", help="Disable sampling (use greedy decoding)")
    p.add_argument("--skip-suppression", action="store_true", help="Skip suppression KL computation")
    p.add_argument("--suppression-strategy", choices=["attn", "embed"], default="attn", help="Counterfactual suppression strategy")
    p.add_argument("--anchor-method", choices=["outgoing", "incoming", "combined"], default="outgoing", help="Aggregate KL for anchor vector")
    p.add_argument("--sparse-top-p", type=float, default=0.0, help="If >0 apply nucleus sparsification before KL (paper-style)")
    p.add_argument("--enable-contrastive", action="store_true", help="Enable contrastive positive/negative sentence generation")
    p.add_argument("--contrastive-samples", type=int, default=5, help="Number of samples for contrastive generation")
    p.add_argument("--test-pca-context", action="store_true", default=True, help="Test PCA context vector effect on answer accuracy (default: enabled)")
    p.add_argument("--no-pca", dest="test_pca_context", action="store_false", help="Disable PCA context vector testing")
    p.add_argument("--pca-num-trials", type=int, default=5, help="Number of trials per context scale for PCA testing (default: 5)")
    return p


def main():

    parser = build_parser()
    args = parser.parse_args()

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
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
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
    results = []
    output_dir = Path("anchor_vectors_output")
    output_dir.mkdir(exist_ok=True)

    proc = DatasetProcessor(
        args=args,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=args.device,
        images_root=args.images_root,
        generate_reasoning=args.generate_reasoning,
        model_name=args.model_name
    )
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        print(f"\nProcessing example {idx+1}/{len(dataset)}")
        
        result = proc.process_sample(example)
        
        result["example_idx"] = idx

        # Save individual result
        with open(output_dir / f"example_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


    # ======================================================
    # 25.11.03 :: Sample for Result 
    # ======================================================

    # Convert to list to ensure data is loaded (Only one)

    # one_sample = "ai2d/images/2067.png"

        
    # sample_size = 20
    # sample_data = []

    # for i in range(min(sample_size, len(dataset))):
    #     example = dataset[i]
    #     if example['image'] == one_sample:
    #         sample_data.append(example)
    
    # print(f"Loaded {len(sample_data)} examples from cached dataset")

    # model_name = "llava-hf/llava-1.5-7b-hf"

    # print(f"Loading model: {model_name}")
    # processor = LlavaProcessor.from_pretrained(model_name)
    # model = LlavaForConditionalGeneration.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    #     device_map="auto" if device == "cuda" else None
    # )
    # model.eval()
    
    # tokenizer = processor.tokenizer
    # model = model.to(device)
    
    # results = []
    # output_dir = Path("anchor_vectors_output")
    # output_dir.mkdir(exist_ok=True)
    
    # for idx, example in enumerate(tqdm(sample_data, desc="Processing examples")):
    #     print(f"\nProcessing example {idx+1}/{len(sample_data)}")
        
    #     result = process_dataset_sample(   
    #         example=example,
    #         model=model,
    #         processor=processor,
    #         tokenizer=tokenizer,
    #         device=device,
    #         model_name=model_name,
    #     )
        
    #     result["example_idx"] = idx
        
    #     # Save individual result
    #     with open(output_dir / f"example_{idx}.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, indent=2, ensure_ascii=False)
        


    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
