#!/usr/bin/env python3
"""
Contrastive sentence generation after thought anchor.

Given a thought anchor, generate multiple continuations and identify:
- Positive sentence: leads to higher probability of correct answer
- Negative sentence: leads to lower probability of correct answer
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoProcessor
from PIL import Image
import re


def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer from generated text."""
    # Try <final>...</final>
    if "<final>" in text and "</final>" in text:
        start = text.find("<final>") + len("<final>")
        end = text.find("</final>")
        return text[start:end].strip()

    # Try patterns like "Final Answer: X" or "The final answer is X"
    import re
    patterns = [
        r'(?:[Ff]inal [Aa]nswer|[Tt]he (?:correct |final )?answer is|[Aa]nswer):?\s*(.+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).strip()
            # Remove markdown bold (**text**)
            answer = re.sub(r'\*\*([^\*]+)\*\*', r'\1', answer)
            # Remove trailing punctuation
            answer = re.sub(r'[.,!?]+$', '', answer)
            # Remove any remaining asterisks
            answer = answer.strip('*').strip()
            return answer

    # Try to find answer at the end (last line)
    lines = text.strip().split("\n")
    if lines:
        last_line = lines[-1].strip()
        # Remove trailing punctuation
        last_line = re.sub(r'[.,!?]+$', '', last_line)
        return last_line

    return None


def get_answer_logits(
    model,
    processor,
    tokenizer,
    prefix_text: str,
    continuation_text: str,
    image: Image.Image,
    device: str,
    correct_answer: str
) -> float:
    """
    Calculate probability score for continuation based on:
    1. Whether it produces the correct answer
    2. The fluency (average token probability) of the continuation

    Args:
        model: The VLM model
        processor: Model processor
        tokenizer: Tokenizer
        prefix_text: Text up to and including the anchor sentence
        continuation_text: Generated continuation after anchor
        image: Input image (not used, text-only scoring)
        device: Device
        correct_answer: Expected correct answer (e.g., "full moon", "K")

    Returns:
        Probability (0.0 to 1.0) - higher if continuation leads to correct answer
    """
    # Extract the final answer from continuation
    generated_answer = extract_final_answer(continuation_text)

    # Check if answer is correct (normalize for comparison)
    def normalize_answer(ans):
        if ans is None:
            return ""
        return ans.lower().strip()

    is_correct = normalize_answer(generated_answer) == normalize_answer(correct_answer)

    # If no final answer was generated, check if answer appears in text
    if not generated_answer:
        is_correct = correct_answer.lower() in continuation_text.lower()

    # Calculate continuation fluency (average token probability)
    clean_prefix = prefix_text
    for vision_token in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]:
        clean_prefix = clean_prefix.replace(vision_token, "")
    clean_prefix = clean_prefix.strip()

    full_text = clean_prefix + "\n\nContinue the next reasoning step:" + continuation_text
    full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)

    prefix_only = clean_prefix + "\n\nContinue the next reasoning step:"
    prefix_tokens = tokenizer(prefix_only, return_tensors="pt")["input_ids"].to(device)
    prefix_len = prefix_tokens.shape[1]

    # Get model logits
    with torch.no_grad():
        outputs = model(input_ids=full_tokens, return_dict=True)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Calculate fluency: average log-likelihood of continuation tokens
    total_log_prob = 0.0
    count = 0

    for i in range(prefix_len, min(full_tokens.shape[1], logits.shape[1])):
        if i > 0 and i - 1 < logits.shape[1]:
            token_logits = logits[0, i-1, :]
            probs = torch.softmax(token_logits, dim=-1)

            actual_token = full_tokens[0, i].item()
            prob = probs[actual_token].item()

            if prob > 0:
                total_log_prob += np.log(prob)
                count += 1

    if count == 0:
        fluency = 0.1
    else:
        avg_log_prob = total_log_prob / count
        fluency = np.exp(avg_log_prob)

    # Final score: fluency if correct answer, very low if incorrect
    if is_correct:
        return float(fluency)
    else:
        return float(fluency * 0.01)  # Penalize incorrect answers heavily


def generate_from_anchor(
    model,
    processor,
    tokenizer,
    prefix_text: str,  # Question + sentences up to anchor
    image: Image.Image,
    device: str,
    correct_answer: str,
    num_samples: int = 5,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.95
) -> Dict:
    """
    Generate multiple continuations from anchor and identify positive/negative sentences.

    Args:
        model: The VLM model
        processor: Model processor
        tokenizer: Tokenizer
        prefix_text: Text up to and including the anchor sentence
        image: Input image
        device: Device
        correct_answer: Expected correct answer
        num_samples: Number of generations to sample (default: 5)
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Dict with positive_sentence, negative_sentence, and all samples
    """

    # Clean prefix_text - remove vision tokens if present
    clean_prefix = prefix_text
    for vision_token in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]:
        clean_prefix = clean_prefix.replace(vision_token, "")
    clean_prefix = clean_prefix.strip()

    # For contrastive generation, we use TEXT-ONLY continuation
    # The model has already "seen" the image through the prefix reasoning
    # Now we're just sampling different text continuations

    continuation_prompt = clean_prefix + "\n\nContinue the next reasoning step:"

    # Use tokenizer directly for text-only generation
    inputs = tokenizer(continuation_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get EOS token
    eos_id = None
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'eos_token_id'):
        eos_id = processor.tokenizer.eos_token_id

    samples = []

    print(f"\n{'='*80}")
    print(f"🎯 Generating {num_samples} continuations from anchor...")
    print(f"{'='*80}")

    for i in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                use_cache=True,
                output_scores=True,  # Get generation scores
                return_dict_in_generate=True  # Return dict with scores
            )

        # Decode
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs.sequences[:, input_len:]

        decoder = tokenizer if hasattr(tokenizer, "batch_decode") else processor
        continuation = decoder.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # Calculate answer probability: how likely is the correct answer given this continuation?
        answer_prob = get_answer_logits(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            prefix_text=prefix_text,
            continuation_text=continuation,
            image=image,
            device=device,
            correct_answer=correct_answer
        )

        # Extract first meaningful sentence from continuation
        sentences = re.split(r'(?<=[.!?])\s+', continuation.strip())

        # Find first sentence that has at least 10 characters (skip "3." etc.)
        first_sentence = ""
        for sent in sentences:
            if len(sent.strip()) >= 10:
                first_sentence = sent
                break

        # Fallback: use first 100 chars if no meaningful sentence found
        if not first_sentence:
            first_sentence = sentences[0] if sentences else continuation[:100]

        samples.append({
            "sample_idx": i,
            "continuation": continuation,
            "first_sentence": first_sentence,
            "answer_probability": answer_prob,
            "final_answer": extract_final_answer(continuation)
        })

        print(f"\n📝 Sample {i+1}/{num_samples}:")
        print(f"   First sentence: {first_sentence[:150]}...")
        print(f"   Final answer: {extract_final_answer(continuation)}")
        print(f"   Answer probability: {answer_prob:.4f}")

    # Sort by answer probability
    samples.sort(key=lambda x: x["answer_probability"], reverse=True)

    positive_sample = samples[0]  # Highest probability
    negative_sample = samples[-1]  # Lowest probability

    print(f"\n{'='*80}")
    print(f"✅ POSITIVE sentence (prob={positive_sample['answer_probability']:.4f}):")
    print(f"   {positive_sample['first_sentence']}")
    print(f"\n❌ NEGATIVE sentence (prob={negative_sample['answer_probability']:.4f}):")
    print(f"   {negative_sample['first_sentence']}")
    print(f"{'='*80}\n")

    return {
        "positive_sentence": positive_sample["first_sentence"],
        "negative_sentence": negative_sample["first_sentence"],
        "positive_full": positive_sample["continuation"],
        "negative_full": negative_sample["continuation"],
        "positive_probability": positive_sample["answer_probability"],
        "negative_probability": negative_sample["answer_probability"],
        "all_samples": samples,
        "correct_answer": correct_answer
    }


def extract_anchor_prefix(
    question: str,
    reasoning_text: str,
    chunks: List[str],
    anchor_vector: List[float]
) -> Tuple[str, int, str]:
    """
    Extract the prefix text up to and including the most important anchor sentence.

    Args:
        question: The question text
        reasoning_text: Full reasoning text
        chunks: List of sentence chunks
        anchor_vector: Anchor importance scores

    Returns:
        (prefix_text, anchor_idx, anchor_sentence)
    """
    # Find the most important anchor (highest score, excluding last sentence)
    valid_anchors = [(i, score) for i, score in enumerate(anchor_vector[:-1]) if score > 0]

    if not valid_anchors:
        # No valid anchor, use first sentence
        anchor_idx = 0
    else:
        # Get highest scoring anchor
        anchor_idx = max(valid_anchors, key=lambda x: x[1])[0]

    anchor_sentence = chunks[anchor_idx]

    # Build prefix: question + sentences up to and including anchor
    prefix_sentences = chunks[:anchor_idx + 1]
    prefix_reasoning = " ".join(prefix_sentences)

    # Clean up formatting
    prefix_reasoning = prefix_reasoning.replace("<reasoning>", "").replace("</reasoning>", "")
    prefix_reasoning = prefix_reasoning.replace("<think>", "").replace("</think>", "")

    prefix_text = question + "\n\nLet me think step by step:\n" + prefix_reasoning

    return prefix_text, anchor_idx, anchor_sentence
