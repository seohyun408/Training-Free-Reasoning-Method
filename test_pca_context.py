#!/usr/bin/env python3
"""
Test PCA context vector approach.

Extract context vector from (positive - negative) hidden states using PCA,
then test if adding this vector to decoder hidden states improves answer accuracy.
"""

import torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import json

from contrastive_generation import (
    generate_from_anchor,
    extract_anchor_prefix,
    extract_hidden_states,
    compute_pca_context_vector,
    test_context_vector_effect
)


def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    print(f"\nLoading model: {model_name}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer

    print("✅ Model loaded")

    # Load a test example with contrastive results
    import glob
    result_files = glob.glob("anchor_vectors_output/example_*.json")

    if not result_files:
        print("❌ No result files found. Please run main.py first.")
        return

    # Load first example with contrastive data
    test_data = None
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
            if data.get('qa_pairs') and data['qa_pairs'][0].get('contrastive'):
                test_data = data
                print(f"\n📄 Using example: {result_file}")
                break

    if not test_data:
        print("❌ No examples with contrastive data found.")
        return

    qa_pair = test_data['qa_pairs'][0]
    contrastive = qa_pair['contrastive']

    # Extract data
    question = qa_pair['question']
    chunks = qa_pair['chunks']
    anchor_vector = qa_pair['anchor_vector']
    correct_answer = contrastive['correct_answer']

    positive_full = contrastive['positive_full']
    negative_full = contrastive['negative_full']

    print(f"\n📋 Question: {question[:100]}...")
    print(f"✅ Positive: {contrastive['positive_sentence'][:100]}...")
    print(f"❌ Negative: {contrastive['negative_sentence'][:100]}...")
    print(f"🎯 Correct answer: {correct_answer}")

    # Extract prefix (up to anchor)
    prefix_text, anchor_idx, anchor_sentence = extract_anchor_prefix(
        question=question,
        reasoning_text="",
        chunks=chunks,
        anchor_vector=anchor_vector
    )

    print(f"\n⚓ Anchor sentence (idx={anchor_idx}): {anchor_sentence[:100]}...")

    # Step 1: Extract hidden states from positive and negative continuations
    print("\n" + "="*80)
    print("📊 STEP 1: Extracting Hidden States")
    print("="*80)

    # Clean texts (remove vision tokens)
    def clean_text(text):
        for vision_token in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]:
            text = text.replace(vision_token, "")
        return text.strip()

    clean_prefix = clean_text(prefix_text)
    positive_text = clean_prefix + "\n\nContinue the next reasoning step:" + positive_full
    negative_text = clean_prefix + "\n\nContinue the next reasoning step:" + negative_full

    print(f"Extracting hidden states from positive continuation...")
    positive_hidden = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        text=positive_text,
        device=device
    )
    print(f"  Positive hidden states shape: {positive_hidden.shape}")

    print(f"Extracting hidden states from negative continuation...")
    negative_hidden = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        text=negative_text,
        device=device
    )
    print(f"  Negative hidden states shape: {negative_hidden.shape}")

    # Step 2: Compute PCA context vector
    print("\n" + "="*80)
    print("📊 STEP 2: Computing PCA Context Vector")
    print("="*80)

    context_vector = compute_pca_context_vector(
        positive_hidden=positive_hidden,
        negative_hidden=negative_hidden,
        n_components=1
    )

    print(f"✅ Context vector computed: shape {context_vector.shape}")

    # Step 3: Test context vector effect
    print("\n" + "="*80)
    print("📊 STEP 3: Testing Context Vector Effect")
    print("="*80)

    results = test_context_vector_effect(
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

    # Save results
    output_file = "pca_context_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for scale, data in results.items():
            serializable_results[str(scale)] = {
                "accuracy": data["accuracy"],
                "correct_count": data["correct_count"],
                "total_trials": data["total_trials"],
                "avg_probability": data["avg_probability"],
                "generated_answers": data["generated_answers"],
                "answer_probabilities": [float(p) for p in data["answer_probabilities"]]
            }

        json.dump({
            "question": question,
            "correct_answer": correct_answer,
            "anchor_sentence": anchor_sentence,
            "context_vector_shape": context_vector.shape,
            "results": serializable_results
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)

    for scale in sorted(results.keys()):
        data = results[scale]
        print(f"Scale {scale:4.1f}: Accuracy = {data['accuracy']:5.1%} ({data['correct_count']}/{data['total_trials']}), "
              f"Avg Prob = {data['avg_probability']:.4f}")


if __name__ == "__main__":
    main()
