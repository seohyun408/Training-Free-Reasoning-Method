import os
import json
import re
from typing import List, Tuple, Optional, Dict
from transformers import AutoTokenizer
import random
import numpy as np 
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from pathlib import Path



def extract_reasoning_from_response(gpt_response: str) -> str:

    if "<think>" in gpt_response and "</think>" in gpt_response:
        s = gpt_response.find("<think>") + len("<think>")
        e = gpt_response.find("</think>")
        if s < e:
            return gpt_response[s:e].strip()

    if "<REASONING>" in gpt_response and "</REASONING>" in gpt_response:
        start_idx = gpt_response.find("<REASONING>") + len("<REASONING>")
        end_idx = gpt_response.find("</REASONING>")
        if start_idx < end_idx:
            return gpt_response[start_idx:end_idx].strip()
    
    return gpt_response.strip()


def extract_qa_pairs(conversation: List[Dict]) -> List[Tuple[str, str]]:
    qa_pairs = []
    current_question = None
    
    for turn in conversation:
        if turn.get("from") == "human":
            current_question = turn.get("value", "")
        elif turn.get("from") == "gpt" and current_question is not None:
            gpt_response = turn.get("value", "")
            qa_pairs.append((current_question, gpt_response))
            current_question = None
    
    return qa_pairs


def split_solution_into_chunks(solution_text: str, tokenizer=None, min_tokens: int = 5) -> List[str]:

    sentence_ending_tokens = [".", "?", "!"]
    chunks = []
    current_chunk = ""
    
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]
        
        # Check for sentence endings
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if next_char == " " or next_char == "\n":
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        
        i += 1
    
    # Add the last chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Merge small chunks (less than 10 characters)
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    # Token-based merging if tokenizer is provided
    if tokenizer is not None and min_tokens > 0:
        i = 0
        while i < len(chunks):
            tokens = tokenizer.encode(chunks[i], add_special_tokens=False)
            token_count = len(tokens)

            # If chunk is too small, merge with next or previous
            if token_count < min_tokens:
                if i < len(chunks) - 1:
                    # Merge with next
                    chunks[i] = chunks[i] + " " + chunks[i + 1]
                    chunks.pop(i + 1)
                    # Don't increment i, check merged chunk again
                elif i > 0:
                    # Merge with previous (last chunk is small)
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
                    break
                else:
                    # Only one chunk and it's small, keep it
                    i += 1
            else:
                i += 1
    
    return chunks


def get_chunk_token_ranges(
    text: str, 
    chunks: List[str], 
    tokenizer: AutoTokenizer
) -> List[Tuple[int, int]]:
    chunk_token_ranges = []
    current_pos = 0
    
    for chunk in chunks:
        # Find chunk in text starting from current_pos
        chunk_start_char = text.find(chunk, current_pos)
        if chunk_start_char == -1:
            # Try normalized matching
            chunk_normalized = re.sub(r"\s+", " ", chunk).strip()
            chunk_start_char = text.find(chunk_normalized, current_pos)
        
        if chunk_start_char == -1:
            print(f"Warning: Chunk not found: {chunk[:50]}...")
            continue
        
        chunk_end_char = chunk_start_char + len(chunk)
        
        # Convert to token indices
        tokens_before_start = tokenizer.encode(text[:chunk_start_char], add_special_tokens=False)
        tokens_before_end = tokenizer.encode(text[:chunk_end_char], add_special_tokens=False)
        
        chunk_start_token = len(tokens_before_start)
        chunk_end_token = len(tokens_before_end)
        
        chunk_token_ranges.append((chunk_start_token, chunk_end_token))
        current_pos = chunk_end_char
    
    return chunk_token_ranges



############################################ // 여기까진 확인





def resample_anchor_sentences(
    question: str,
    chunks: List[str],
    anchor_indices: List[int],
    model,
    processor,
    tokenizer,
    image: Optional[Image.Image] = None,
    device: str = "cuda",
    num_resamples: int = 5,
    temperature_range: Tuple[float, float] = (0.7, 1.2)
) -> Dict:
    """
    Step 2: Positive/Negative Sampling
    
    Resample identified anchor sentences with different temperatures to generate
    variations, then select highest/lowest correctness probability versions.
    
    Args:
        question: The original question
        chunks: All reasoning chunks
        anchor_indices: Indices of high-impact chunks (anchors)
        model: The language model
        processor: The processor (for vision models)
        tokenizer: The tokenizer
        image: Optional image for VQA
        device: Device to use
        num_resamples: Number of resampling attempts (2-5)
        temperature_range: (min, max) temperature for sampling diversity
        
    Returns:
        Dictionary containing positive and negative samples for each anchor
    """
    import torch.nn.functional as F
    
    results = {}
    
    for anchor_idx in anchor_indices:
        anchor_chunk = chunks[anchor_idx]
        
        # Build context: question + chunks before anchor
        context = question + "\n\n"
        for i in range(anchor_idx):
            context += chunks[i] + " "
        
        # Generate multiple versions of this anchor sentence
        resampled_versions = []
        
        for resample_idx in range(num_resamples):
            # Use different temperatures for diversity
            temperature = random.uniform(temperature_range[0], temperature_range[1])
            
            # Prepare inputs
            if image is not None and processor is not None:
                # Add <image> token if not present (required for LLaVA)
                context_with_image = context if "<image>" in context else "<image>\n" + context
                inputs = processor(text=context_with_image, images=image, return_tensors="pt")
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            else:
                inputs = tokenizer(context, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with sampling
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=len(tokenizer.encode(anchor_chunk)) + 10,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Compute correctness probability (simplified version)
            # Use the model's confidence in the generated tokens
            correctness_score = compute_correctness_score(
                generated_text, anchor_chunk, model, processor, tokenizer, 
                context, image, device
            )
            
            resampled_versions.append({
                "text": generated_text,
                "correctness_score": correctness_score,
                "temperature": temperature
            })
        
        # Sort by correctness score
        resampled_versions.sort(key=lambda x: x["correctness_score"], reverse=True)
        
        # Select positive (highest) and negative (lowest)
        results[anchor_idx] = {
            "original_chunk": anchor_chunk,
            "positive_sample": resampled_versions[0],  # Highest correctness
            "negative_sample": resampled_versions[-1],  # Lowest correctness
            "all_samples": resampled_versions
        }
    
    return results


def compute_correctness_score(
    generated_text: str,
    reference_text: str,
    model,
    processor,
    tokenizer,
    context: str,
    image: Optional[Image.Image] = None,
    device: str = "cuda"
) -> float:
    """
    Compute correctness probability of generated text.
    
    Uses the model's log probability of generating the text as a proxy for correctness.
    Higher probability = more likely to be correct reasoning.
    
    Args:
        generated_text: The generated sentence
        reference_text: Original reference sentence
        model: The model
        processor: The processor
        tokenizer: The tokenizer
        context: Context before the sentence
        image: Optional image
        device: Device
        
    Returns:
        Correctness score (0-1, higher is better)
    """
    import torch
    import torch.nn.functional as F
    
    # Combine context + generated text
    full_text = context + generated_text
    
    # Prepare inputs
    if image is not None and processor is not None:
        # Add <image> token if not present (required for LLaVA)
        full_text_with_image = full_text if "<image>" in full_text else "<image>\n" + full_text
        inputs = processor(text=full_text_with_image, images=image, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
    else:
        inputs = tokenizer(full_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
    
    # Get tokens for the generated part
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    generated_tokens = full_tokens[len(context_tokens):]
    
    if len(generated_tokens) == 0:
        return 0.0
    
    # Calculate log probability of generated tokens
    log_probs = []
    for i, token_id in enumerate(generated_tokens):
        if len(context_tokens) + i < len(logits):
            token_logits = logits[len(context_tokens) + i]
            probs = F.softmax(token_logits, dim=0)
            log_prob = torch.log(probs[token_id] + 1e-10)
            log_probs.append(log_prob.item())
    
    if not log_probs:
        return 0.0
    
    # Average log probability
    avg_log_prob = np.mean(log_probs)
    
    # Convert to probability (normalized to 0-1 range)
    # Using sigmoid to map to [0, 1]
    correctness_score = 1 / (1 + np.exp(-avg_log_prob))
    
    return correctness_score




