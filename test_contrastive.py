#!/usr/bin/env python3
"""
Test contrastive positive/negative sentence generation.

Usage:
    python test_contrastive.py
"""

import torch
from datasets import load_dataset
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from PIL import Image
from process import process_dataset_sample
import os
import json

# Setup environment
os.environ['MAX_NEW_TOKENS'] = '256'
os.environ['DO_SAMPLE'] = '1'
os.environ['ENABLE_CONTRASTIVE'] = '1'
os.environ['CONTRASTIVE_SAMPLES'] = '5'

# Load dataset (one example with working image)
dataset = load_dataset('Xkev/LLaVA-CoT-100k', split='train[8:9]', cache_dir='/mnt/hdd/llava/llava-cot-100k')
model_name = 'Qwen/Qwen3-VL-8B-Instruct'
device = 'cuda:1'

print('Loading model...')
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir='/mnt/hdd/huggingface-models')
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map={'': device},
    trust_remote_code=True,
    cache_dir='/mnt/hdd/huggingface-models',
    low_cpu_mem_usage=True,
    dtype=torch.float16
)
model.eval()
tokenizer = processor.tokenizer

print('\nProcessing example with contrastive generation...')
result = process_dataset_sample(
    example=dataset[0],
    model=model,
    processor=processor,
    tokenizer=tokenizer,
    device=device,
    model_name=model_name,
    images_root='/mnt/hdd/llava/llava-cot-100k/extracted',
    generate_reasoning=True
)

print('\n' + '='*80)
print('📊 RESULTS')
print('='*80)

if result.get('qa_pairs'):
    for qa in result['qa_pairs']:
        print(f"\nQuestion: {qa['question'][:100]}...")
        print(f"Reasoning: {qa['reasoning_text'][:200]}...")

        if qa.get('contrastive'):
            cont = qa['contrastive']
            print(f"\n✅ POSITIVE sentence (prob={cont['positive_probability']:.4f}):")
            print(f"   {cont['positive_sentence']}")
            print(f"\n❌ NEGATIVE sentence (prob={cont['negative_probability']:.4f}):")
            print(f"   {cont['negative_sentence']}")

            print(f"\n📈 All {len(cont['all_samples'])} samples:")
            for i, sample in enumerate(cont['all_samples']):
                print(f"   Sample {i+1}: prob={sample['answer_probability']:.4f}, answer={sample['final_answer']}")
        else:
            print("\n⚠️ No contrastive results")

# Save results
output_path = 'test_contrastive_output.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to: {output_path}")
