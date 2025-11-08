import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set, Union
import time

from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# from pkld import pkld

from pytorch_models.model_config import model2layers_heads
from pytorch_models import analyze_text
from .logits_funcs import (
    analyze_text_get_p_logits,
    decompress_logits_for_position,
)
# from .receiver_head_funcs import (
#     get_problem_text_sentences,
#     get_model_rollouts_root,
# )
from .tokenizer_funcs import get_raw_tokens
from .attn_funcs import get_sentence_token_boundaries
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import Optional

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "whitebox-analyses"))
from pytorch_models.hooks import QwenAttentionHookManager, LlamaAttentionHookManager


# @pkld
# def get_suppression_KL_matrix(
#     problem_num: int,
#     p_nucleus: float = 0.9999,
#     model_name: str = "qwen-14b",
#     is_correct: bool = True,
#     only_first: Optional[int] = None,
#     take_log: bool = True,
# ) -> Optional[np.ndarray]:
#     try:
#         text, sentences = get_problem_text_sentences(problem_num, is_correct, model_name)
#     except Exception as e:
#         print(f"Error loading problem {problem_num}: {e}")
#         return None

#     layers, heads = model2layers_heads(model_name)
#     layers_to_mask = {i: list(range(heads)) for i in range(layers)}

#     # Convert sentences to token ranges for compatibility
#     sentence_boundaries = get_sentence_token_boundaries(text, sentences, model_name)
#     sentence2ranges = {i: boundary for i, boundary in enumerate(sentence_boundaries)}

#     text_tokens = get_raw_tokens(text, model_name)
#     device_map = "auto"

#     # Use CPU for very long sequences on Windows
#     if len(text_tokens) > 4200 and os.name == "nt":
#         device_map = "cpu"
#         print(f"Using CPU for long sequence ({problem_num}): {len(text_tokens)=}")

#     kw = {
#         "text": text,
#         "model_name": model_name,
#         "seed": 0,
#         "p_nucleus": p_nucleus,
#         "float32": model_name == "qwen-15b",
#         "token_range_to_mask": None,
#         "layers_to_mask": None,
#         "device_map": device_map,
#     }

#     try:
#         baseline_data = analyze_text_get_p_logits(**kw)
#     except (KeyError, torch.OutOfMemoryError, RuntimeError) as e:
#         print(f"CUDA failed: {e}. Sleeping for 10 seconds....")
#         time.sleep(10)
#         return None

#     sentence_sentence_scores = np.full((len(sentence2ranges), len(sentence2ranges)), np.nan)

#     for (
#         sentence_num,
#         token_range,
#     ) in tqdm(sentence2ranges.items(), desc=f"Examining sentence2sentence ({problem_num})"):
#         kw["token_range_to_mask"] = list(token_range)
#         kw["layers_to_mask"] = layers_to_mask
#         try:
#             s_data = analyze_text_get_p_logits(**kw)
#         except (KeyError, torch.OutOfMemoryError, RuntimeError) as e:
#             print(f"Probably CUDA failed: {e}. Sleeping for 10 seconds....")
#             time.sleep(10)
#             return

#         KL_log_l = []
#         for i in range(len(text_tokens)):
#             b_idxs, b_logits = decompress_logits_for_position(baseline_data, i)
#             s_idxs, s_logits = decompress_logits_for_position(s_data, i)

#             KL_sparse = calculate_kl_divergence_sparse(
#                 (b_idxs, b_logits), (s_idxs, s_logits), temperature=0.6, epsilon=1e-9
#             )
#             # if norm_entropy:

#             if take_log:
#                 KL_log = np.log(KL_sparse + 1e-9)
#             else:
#                 KL_log = KL_sparse
#             KL_log_l.append(KL_log)
#             if np.isnan(KL_log):
#                 raise ValueError(f"NaN KL log: {KL_log}")

#         KL_log_l = np.array(KL_log_l)
#         sentence_KL_logs = []
#         for sentence_idx_loop, token_range_loop in sentence2ranges.items():
#             # Ensure indices are within bounds of KL_log_l
#             start_idx = min(token_range_loop[0], len(KL_log_l))
#             end_idx = min(token_range_loop[1], len(KL_log_l))
#             if only_first is not None:
#                 if end_idx - start_idx > only_first:
#                     end_idx = start_idx + only_first

#             if start_idx < end_idx:
#                 mean_log_kl = np.nanmean(KL_log_l[start_idx:end_idx])  # Use nanmean for safety
#             else:
#                 mean_log_kl = np.nan  # Assign NaN if range is empty or invalid
#             sentence_KL_logs.append(mean_log_kl)
#             sentence_sentence_scores[
#                 sentence_idx_loop,
#                 sentence_num,
#             ] = mean_log_kl

#     return sentence_sentence_scores


# def calculate_kl_divergence_sparse(
#     baseline_data: Tuple[np.ndarray, np.ndarray],
#     suppressed_data: Tuple[np.ndarray, np.ndarray],
#     temperature: float = 0.6,
# ) -> float:
#     """
#     Calculates the KL divergence KL(P || Q) between two probability distributions
#     derived from sparse top-p logits. Clips small negative results to 0.

#     P is derived from baseline_data, Q from suppressed_data.

#     Args:
#         baseline_data (Tuple[np.ndarray, np.ndarray]): Tuple containing
#             (indices, logits) for the baseline distribution (P).
#             Indices should be int32, logits float16 or float32.
#         suppressed_data (Tuple[np.ndarray, np.ndarray]): Tuple containing
#             (indices, logits) for the suppressed distribution (Q).
#         temperature (float): Temperature to use for softmax conversion. Defaults to 0.6.
#         epsilon (float): Small value (currently unused here, handled internally).

#     Returns:
#         float: The calculated KL divergence KL(P || Q), guaranteed non-negative.
#                Returns np.nan if inputs are invalid or calculation yields NaN/inf.
#     """
#     b_idxs, b_logits = baseline_data
#     s_idxs, s_logits = suppressed_data

#     if b_idxs is None or b_logits is None or s_idxs is None or s_logits is None:
#         print("Warning: Invalid input data (None found). Cannot calculate KL divergence.")
#         return np.nan
#     if len(b_idxs) != len(b_logits) or len(s_idxs) != len(s_logits):
#         print(
#             "Warning: Mismatch between length of indices and logits. Cannot calculate KL divergence."
#         )
#         return np.nan
#     # Allow empty arrays for one side? KL(P||0) is inf if P>0, KL(0||Q) is 0.
#     # If both are empty, KL is 0 or NaN? Let's return 0 if both empty, NaN otherwise for now.
#     if len(b_idxs) == 0 and len(s_idxs) == 0:
#         return 0.0
#     if len(b_idxs) == 0 or len(s_idxs) == 0:
#         # If P is non-empty and Q is empty -> Div is Inf
#         # If P is empty and Q is non-empty -> Div is 0
#         # Let's return NaN for simplicity unless explicitly handled otherwise.
#         # Returning inf might be more correct if P is non-empty.
#         # For now, keep NaN to signal an edge case was hit.
#         print("Warning: One distribution has no tokens. Returning NaN.")
#         return np.nan

#     # Ensure logits are float32 for stable softmax
#     b_logits = b_logits.astype(np.float32)
#     s_logits = s_logits.astype(np.float32)

#     union_indices = np.union1d(b_idxs, s_idxs)
#     union_size = len(union_indices)

#     idx_to_union_pos = {idx: pos for pos, idx in enumerate(union_indices)}

#     min_logit_val = -1e9  # Approx -inf for softmax
#     b_logits_union = np.full(union_size, min_logit_val, dtype=np.float32)
#     s_logits_union = np.full(union_size, min_logit_val, dtype=np.float32)

#     for idx, logit in zip(b_idxs, b_logits):
#         b_logits_union[idx_to_union_pos[idx]] = logit
#     for idx, logit in zip(s_idxs, s_logits):
#         s_logits_union[idx_to_union_pos[idx]] = logit

#     b_logits_tensor = torch.from_numpy(b_logits_union)
#     s_logits_tensor = torch.from_numpy(s_logits_union)

#     log_p = F.log_softmax(b_logits_tensor / temperature, dim=0)
#     log_q = F.log_softmax(s_logits_tensor / temperature, dim=0)

#     # Check for immediate issues after log_softmax
#     if (
#         torch.isinf(log_p).any()
#         or torch.isnan(log_p).any()
#         or torch.isinf(log_q).any()
#         or torch.isnan(log_q).any()
#     ):
#         print("Warning: Inf or NaN detected in log-probabilities. Inputs might be too extreme.")
#         return np.nan

#     p_dist = torch.exp(log_p)  # Equivalent to softmax(logits/T)

#     if torch.isnan(p_dist).any():
#         print("Warning: NaN detected in P distribution.")
#         return np.nan

#     kl_terms = p_dist * (log_p - log_q)

#     kl_terms = torch.where(p_dist == 0, torch.tensor(0.0, dtype=kl_terms.dtype), kl_terms)

#     if torch.isnan(kl_terms).any():
#         print("Warning: NaN detected during KL term calculation.")
#         return np.nan

#     kl_div = torch.sum(kl_terms)

#     if torch.isinf(kl_div):
#         print("Warning: KL divergence is infinite.")
#         return kl_div.item()  # This will be float('inf')

#     kl_div_value = kl_div.item()

#     if kl_div_value < 0:
#         if kl_div_value < -1e-6:  # Adjust tolerance if needed
#             print(
#                 f"Warning: KL divergence significantly negative ({kl_div_value:.2e}). Clipping to 0.0. This might indicate an issue."
#             )
#         return 0.0
#     else:
#         return kl_div_value


def get_model_logits(
    model,
    processor,
    tokenizer,
    text: str,
    image: Optional[Image.Image] = None,
    token_range_to_mask: Optional[List[Tuple[int, int]]] = None,
    device: str = "cuda"
) -> List[torch.Tensor]:
    """
    Get model logits for each position in the text.
    
    Args:
        model: The model (LLaVA or text model)
        processor: Processor for vision models (optional for text models)
        tokenizer: The tokenizer
        text: Input text
        image: Optional PIL Image for vision models
        token_range_to_mask: Optional token ranges to mask via attention suppression
        device: Device to use
        
    Returns:
        List of logits tensors for each token position
    """
    # Prepare inputs - check if it's a vision model
    is_vision_model = hasattr(model, 'config') and hasattr(model.config, 'vision_config')
    
    if is_vision_model and image is not None and processor is not None:
        # Vision-language model (LLaVA)
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
    else:
        # Text-only model
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(device)
    
    # Apply attention suppression hooks if needed
    hook_manager = None
    if token_range_to_mask is not None:
        # Convert token ranges to the format expected by hooks
        # hooks expect list of lists: [[start, end], [start, end], ...]
        token_ranges_list = [[start, end] for start, end in token_range_to_mask]
        
        # For LLaVA models, use the language_model component
        model_to_hook = model.language_model if hasattr(model, 'language_model') else model
        model_type = type(model_to_hook).__name__
        
        print(f"[DEBUG] Token ranges to mask: {token_ranges_list}")
        
        # Use appropriate hook manager based on model type
        if 'Llama' in model_type:
            hook_manager = LlamaAttentionHookManager(
                model=model_to_hook,
                token_range=token_ranges_list,
                layer_2_heads_suppress=None  # Suppress all heads in all layers
            )
        else:
            # Default to Qwen for other models
            hook_manager = QwenAttentionHookManager(
                model=model_to_hook,
                token_range=token_ranges_list,
                layer_2_heads_suppress=None  # Suppress all heads in all layers
            )
        
        hook_manager.apply()
        
    
    try:
        # Forward pass
        with torch.no_grad():
            if is_vision_model and image is not None:
                # LLaVA model expects input_ids, pixel_values, attention_mask
                outputs = model(**inputs, output_attentions=False)
            else:
                # Text-only model
                outputs = model(input_ids, output_attentions=False)
            
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            logits_list = [logits[i] for i in range(logits.shape[0])]
            
        return logits_list
    finally:
        # Clean up hooks
        if hook_manager is not None:
            hook_manager.clear()



def calculate_kl_divergence_sparse2(
    baseline_logits: torch.Tensor,
    suppressed_logits: torch.Tensor,
    temperature: float = 0.6,
    top_p: float = 0.9999
) -> float:
    """
    Calculate KL divergence between baseline and suppressed logit distributions.
    Uses a simplified approach that's numerically stable.
    
    Args:
        baseline_logits: Baseline logits tensor [vocab_size]
        suppressed_logits: Suppressed logits tensor [vocab_size]
        temperature: Temperature for softmax
        top_p: Nucleus sampling threshold (not used, kept for compatibility)
        
    Returns:
        KL divergence value (always >= 0)
    """
    # Apply temperature
    baseline_logits_temp = baseline_logits / temperature
    suppressed_logits_temp = suppressed_logits / temperature
    
    # Compute log probabilities directly (more numerically stable)
    baseline_log_probs = F.log_softmax(baseline_logits_temp, dim=0)
    suppressed_log_probs = F.log_softmax(suppressed_logits_temp, dim=0)
    
    # Compute probabilities
    baseline_probs = torch.exp(baseline_log_probs)
    
    # KL divergence: KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    # Only compute for tokens where P > threshold to avoid numerical issues
    threshold = 1e-8
    mask = baseline_probs > threshold
    
    if mask.sum() == 0:
        # All probabilities are tiny, return 0
        return 0.0
    
    # Compute KL only on masked tokens
    kl_terms = baseline_probs[mask] * (baseline_log_probs[mask] - suppressed_log_probs[mask])
    kl_div = torch.sum(kl_terms)
    
    kl_value = kl_div.item()
    
    # Handle numerical errors
    if torch.isnan(kl_div) or torch.isinf(kl_div):
        print(f"Warning: Invalid KL divergence (nan or inf), returning 0.0")
        return 0.0
    
    if kl_value < 0:
        if kl_value > -1e-5:
            # Small negative due to numerical errors, clamp to 0
            kl_value = 0.0
        else:
            # Larger negative, something is wrong
            print(f"Warning: Negative KL divergence detected: {kl_value:.6e}, clamping to 0")
            kl_value = 0.0
    
    return kl_value


def compute_suppression_kl_matrix2(
    model,
    processor,
    tokenizer,
    text: str,
    image: Optional[Image.Image] = None,
    chunks: List[str] = None,
    chunk_token_ranges: List[Tuple[int, int]] = None,
    device: str = "cuda",
    p_nucleus: float = 0.9999,
    temperature: float = 0.6,
    take_log: bool = False
) -> np.ndarray:
    """
    Compute KL divergence matrix by suppressing each chunk and measuring effect.
        
    Returns:
        Matrix of shape [num_chunks, num_chunks] where [i, j] is effect of suppressing chunk i on chunk j
    """
    num_chunks = len(chunks)
    sentence_sentence_scores = np.full((num_chunks, num_chunks), np.nan)
    
    # Get baseline logits (no suppression)
    print("Computing baseline logits...")
    baseline_logits_list = get_model_logits(
        model, processor, tokenizer, text, image=image, token_range_to_mask=None, device=device
    )
    
    print(f"  Baseline logits length: {len(baseline_logits_list)}")
    
    # For each chunk, suppress it and measure effect on all positions
    for chunk_idx, (chunk, token_range) in enumerate(tqdm(
        zip(chunks, chunk_token_ranges), 
        desc="Suppressing chunks", 
        total=num_chunks
    )):
        print(f"\n[DEBUG] Suppressing chunk {chunk_idx}")
        
        # Suppress this chunk
        suppressed_logits_list = get_model_logits(
            model, processor, tokenizer, text, image=image,
            token_range_to_mask=[token_range], 
            device=device
        )
        
        # Debug: Check if logits changed
        if len(baseline_logits_list) == len(suppressed_logits_list):
            # Compare multiple positions to see if suppression worked
            check_positions = [min(10, len(baseline_logits_list) - 1), 
                             token_range[0], 
                             min(token_range[1], len(baseline_logits_list) - 1)]
            
            for check_idx in check_positions:
                if check_idx >= 0 and check_idx < len(baseline_logits_list):
                    max_diff = torch.max(torch.abs(baseline_logits_list[check_idx] - suppressed_logits_list[check_idx])).item()
                    mean_diff = torch.mean(torch.abs(baseline_logits_list[check_idx] - suppressed_logits_list[check_idx])).item()
                    
            # Overall check
            sample_idx = min(10, len(baseline_logits_list) - 1)
            if sample_idx > 0:
                max_diff = torch.max(torch.abs(baseline_logits_list[sample_idx] - suppressed_logits_list[sample_idx])).item()

        
        # Calculate KL divergence for each chunk position
        for target_chunk_idx, target_token_range in enumerate(chunk_token_ranges):
            # Get KL divergence over tokens in target chunk
            kl_values = []
            start_token = target_token_range[0]
            end_token = min(target_token_range[1], len(baseline_logits_list), len(suppressed_logits_list))
            
            if start_token >= end_token:
                continue
            
            for token_idx in range(start_token, end_token):
                if token_idx < len(baseline_logits_list) and token_idx < len(suppressed_logits_list):
                    kl = calculate_kl_divergence_sparse2(
                        baseline_logits_list[token_idx],
                        suppressed_logits_list[token_idx],
                        temperature=temperature,
                        top_p=p_nucleus
                    )
                    if not np.isnan(kl) and kl >= 0:
                        kl_values.append(kl)
            
            if kl_values:
                mean_kl = np.mean(kl_values)
                raw_mean_kl = mean_kl  # Save for debugging
                
                # Apply log transform if requested
                # Note: Only take log if KL > 0 to avoid log(0) = -inf
                if take_log:
                    if mean_kl > 1e-10:
                        mean_kl = np.log(mean_kl)
                    else:
                        # KL is essentially 0, keep as is or use small negative
                        mean_kl = np.log(1e-10)  # -23.03
                
                sentence_sentence_scores[target_chunk_idx, chunk_idx] = mean_kl
                
            else:
                # No valid KL values - set to nan
                sentence_sentence_scores[target_chunk_idx, chunk_idx] = np.nan
    
    return sentence_sentence_scores



if __name__ == "__main__":
    problem_num = 2238
    model_name = "qwen-15b"
    is_correct = True  # Use correct solutions

    output_file = get_suppression_KL_matrix(
        problem_num=problem_num,
        p_nucleus=0.9999,
        model_name=model_name,
        is_correct=is_correct,
    )

