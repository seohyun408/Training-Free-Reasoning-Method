#!/usr/bin/env python3
"""
Run PCA context vector test on existing contrastive results.

Uses already generated contrastive results from anchor_vectors_output/*.json
"""

import torch
import numpy as np
import json
import glob
import os
from pathlib import Path
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

from contrastive_generation import (
    extract_hidden_states,
    compute_pca_context_vector,
    test_context_vector_effect
)


def main():
    # Setup cache directories
    MODEL_CACHE = "/mnt/hdd/huggingface-models"
    os.environ["HF_HOME"] = MODEL_CACHE
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE

    # Setup device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model cache: {MODEL_CACHE}")

    # Load model (same as main.py: Qwen3-VL-8B-Instruct)
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"\nLoading model: {model_name}")

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE
    )

    load_kwargs = {
        "trust_remote_code": True,
        "cache_dir": MODEL_CACHE,
        "low_cpu_mem_usage": True,
        "dtype": torch.float16,
    }

    device_map = {"": device} if device.startswith("cuda") else None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,
        **load_kwargs
    )
    model.eval()
    tokenizer = processor.tokenizer

    print("✅ Model loaded")

    # Find existing results with contrastive data
    result_files = glob.glob("anchor_vectors_output/example_*.json")

    if not result_files:
        print("❌ No result files found")
        return

    print(f"\nFound {len(result_files)} result files")

    # Process each file
    for result_file in sorted(result_files):
        print(f"\n{'='*80}")
        print(f"Processing: {result_file}")
        print(f"{'='*80}")

        with open(result_file, 'r') as f:
            data = json.load(f)

        if not data.get('qa_pairs'):
            print("  ⚠️ No QA pairs")
            continue

        qa_pair = data['qa_pairs'][0]
        contrastive = qa_pair.get('contrastive')

        if not contrastive or not contrastive.get('positive_full'):
            print("  ⚠️ No contrastive data")
            continue

        # Check if PCA already done
        if contrastive.get('pca_context'):
            print("  ⚠️ PCA already computed, skipping")
            continue

        # Extract data
        question = qa_pair['question']
        chunks = qa_pair['chunks']
        anchor_vector = qa_pair['anchor_vector']
        correct_answer = contrastive['correct_answer']
        positive_full = contrastive['positive_full']
        negative_full = contrastive['negative_full']

        print(f"\n  Question: {question[:80]}...")
        print(f"  Correct answer: {correct_answer}")

        # Reconstruct prefix
        from contrastive_generation import extract_anchor_prefix
        prefix_text, anchor_idx, anchor_sentence = extract_anchor_prefix(
            question=question,
            reasoning_text="",
            chunks=chunks,
            anchor_vector=anchor_vector
        )

        print(f"  Anchor (idx={anchor_idx}): {anchor_sentence[:80]}...")

        # Clean texts
        def clean_text(text):
            for vision_token in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]:
                text = text.replace(vision_token, "")
            return text.strip()

        clean_prefix = clean_text(prefix_text)
        positive_text = clean_prefix + "\n\nContinue the next reasoning step:" + positive_full
        negative_text = clean_prefix + "\n\nContinue the next reasoning step:" + negative_full

        # Extract hidden states
        print("\n  [PCA] Extracting hidden states...")
        positive_hidden = extract_hidden_states(
            model=model,
            tokenizer=tokenizer,
            text=positive_text,
            device=device
        )

        negative_hidden = extract_hidden_states(
            model=model,
            tokenizer=tokenizer,
            text=negative_text,
            device=device
        )

        # Compute PCA context vector
        print("  [PCA] Computing PCA context vector...")
        context_vector = compute_pca_context_vector(
            positive_hidden=positive_hidden,
            negative_hidden=negative_hidden,
            n_components=1
        )

        # Test context vector effect
        print("  [PCA] Testing context vector effect...")
        pca_results = test_context_vector_effect(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            prefix_text=prefix_text,
            positive_full=positive_full,
            negative_full=negative_full,
            context_vector=context_vector,
            correct_answer=correct_answer,
            device=device,
            num_trials=5,
            context_scales=[0.0, 0.5, 1.0, 2.0, 5.0]
        )

        # Add PCA results to data
        contrastive['pca_context'] = {
            "context_vector_shape": list(context_vector.shape),
            "results": {}
        }

        for scale, result in pca_results.items():
            contrastive['pca_context']['results'][str(scale)] = {
                "accuracy": result["accuracy"],
                "correct_count": result["correct_count"],
                "total_trials": result["total_trials"],
                "avg_probability": result["avg_probability"],
                "generated_answers": result["generated_answers"],
                "answer_probabilities": [float(p) for p in result["answer_probabilities"]]
            }

        # Save updated results
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n  ✅ Updated: {result_file}")

        # Print summary
        print("\n  📊 Summary:")
        for scale in sorted(pca_results.keys()):
            result = pca_results[scale]
            print(f"    Scale {scale:4.1f}: Accuracy = {result['accuracy']:5.1%} "
                  f"({result['correct_count']}/{result['total_trials']}), "
                  f"Avg Prob = {result['avg_probability']:.4f}")

        print(f"\n  🏆 Best scale: {max(pca_results.keys(), key=lambda s: pca_results[s]['accuracy'])}")


if __name__ == "__main__":
    main()
