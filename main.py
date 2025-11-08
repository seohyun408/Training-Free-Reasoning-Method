import os
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
from pathlib import Path

from process import process_dataset_sample

# Environment // API
HF_DATASETS_CACHE = os.getenv('HF_DATASETS_CACHE')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
WANDB_TOKEN = os.getenv('WANDB_TOKEN')


def main():

    print("Loading LLaVA-CoT-100k dataset...")
    dataset = load_dataset("Xkev/LLaVA-CoT-100k", split="train")
    
    sample_size = 20
    sample_data = []
    
    # ======================================================
    # 25.11.03 :: Sample for Result 
    # ======================================================

    # Convert to list to ensure data is loaded (Only one)

    one_sample = "ai2d/images/2067.png"

    for i in range(min(sample_size, len(dataset))):
        example = dataset[i]
        if example['image'] == one_sample:
            sample_data.append(example)
    
    print(f"Loaded {len(sample_data)} examples from cached dataset")


    # model_name = "deepseek-ai/deepseek-r1-distill-qwen-7b" # 14b -> 7b
    model_name = "llava-hf/llava-1.5-7b-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    processor = LlavaProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()
    
    tokenizer = processor.tokenizer
    model = model.to(device)
    
    results = []
    output_dir = Path("anchor_vectors_output")
    output_dir.mkdir(exist_ok=True)
    
    for idx, example in enumerate(tqdm(sample_data, desc="Processing examples")):
        print(f"\nProcessing example {idx+1}/{len(sample_data)}")
        
        result = process_dataset_sample(   
            example=example,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model_name=model_name,
        )
        
        result["example_idx"] = idx
        
        # Save individual result
        with open(output_dir / f"example_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        


if __name__ == "__main__":
    main()
