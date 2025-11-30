import os
import numpy as np 
import torch
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "method"))

from contrastive_generation import extract_final_answer, get_answer_logits

def extract_hidden_states(
    model,
    tokenizer,
    text: str,
    device: str
) -> np.ndarray:
    """
    Extract hidden states from the model for given text.

    Args:
        model: The VLM model
        tokenizer: Tokenizer
        text: Input text
        device: Device

    Returns:
        Hidden states as numpy array, shape (seq_len, hidden_dim)
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # outputs.hidden_states is a tuple of (num_layers, batch_size, seq_len, hidden_dim)
        # We use the last layer
        last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

    # Convert to numpy and remove batch dimension
    hidden_states = last_hidden_state[0].cpu().numpy()  # (seq_len, hidden_dim)

    return hidden_states


def compute_pca_context_vector(
    positive_hidden: np.ndarray,
    negative_hidden: np.ndarray,
    n_components: int = 1
) -> np.ndarray:
    """
    Compute PCA on (positive - negative) hidden states to extract context vector.

    Args:
        positive_hidden: Hidden states from positive continuation, shape (seq_len_pos, hidden_dim)
        negative_hidden: Hidden states from negative continuation, shape (seq_len_neg, hidden_dim)
        n_components: Number of PCA components (default: 1 for first principal component)

    Returns:
        Context vector: first principal component, shape (hidden_dim,)
    """

    min_len = min(positive_hidden.shape[0], negative_hidden.shape[0])

    # Compute difference: positive - negative # token-level feature difference
    diff = positive_hidden[:min_len, :] - negative_hidden[:min_len, :]  # (min_len, hidden_dim)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(diff)

    # First principal component is the context vector
    context_vector = pca.components_[0]  # (hidden_dim,)

    print(f"[PCA] Explained variance ratio: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"[PCA] Context vector shape: {context_vector.shape}")

    return context_vector


def generate_with_context_vector(
    model,
    processor,
    tokenizer,
    prefix_text: str,
    context_vector: np.ndarray,
    context_scale: float,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7
) -> str:
    """
    Generate continuation with context vector added to hidden states.

    This is implemented using a forward hook that adds the context vector
    to the decoder's hidden states during generation.

    Args:
        model: The VLM model
        processor: Model processor
        tokenizer: Tokenizer
        prefix_text: Text prefix (up to anchor)
        context_vector: PCA context vector to add, shape (hidden_dim,)
        context_scale: Scaling factor for context vector (e.g., 1.0, 0.5)
        device: Device
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated continuation text
    """
    # Clean prefix
    clean_prefix = prefix_text
    for vision_token in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]:
        clean_prefix = clean_prefix.replace(vision_token, "")
    clean_prefix = clean_prefix.strip()

    full_text = clean_prefix + "\n\nContinue the next reasoning step:"

    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    # Convert context vector to tensor
    context_tensor = torch.from_numpy(context_vector).float().to(device)

    # Hook to add context vector to hidden states
    def add_context_hook(module, input, output):
        """Add context vector to hidden states."""
        # output is typically a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Add context vector to all positions
        # hidden_states shape: (batch_size, seq_len, hidden_dim)
        if hidden_states.shape[-1] == context_tensor.shape[0]:
            hidden_states = hidden_states + context_scale * context_tensor.unsqueeze(0).unsqueeze(0)

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        return output

    # Register hook on the last decoder layer
    # For Qwen models, this is typically model.model.layers[-1]
    hook_handle = None
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            hook_handle = model.model.layers[-1].register_forward_hook(add_context_hook)

        # Generate with context
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Extract only the generated part (after the prompt)
        if full_text in full_output:
            continuation = full_output.split(full_text, 1)[1].strip()
        else:
            continuation = full_output.strip()

    finally:
        # Remove hook
        if hook_handle is not None:
            hook_handle.remove()

    return continuation


def test_context_vector_effect(
    model,
    processor,
    tokenizer,
    prefix_text: str,
    positive_full: str,
    negative_full: str,
    context_vector: np.ndarray,
    correct_answer: str,
    device: str,
    num_trials: int,
    context_scales: float
) -> Tuple[str, int]:
    """
    Test the effect of context vector on answer accuracy.

    Generate multiple times with different context scales and measure
    how often the correct answer is produced.

    Args:
        model: The VLM model
        processor: Model processor
        tokenizer: Tokenizer
        prefix_text: Text prefix (up to anchor)
        positive_full: Full positive continuation (for reference)
        negative_full: Full negative continuation (for reference)
        context_vector: PCA context vector
        correct_answer: Expected correct answer
        device: Device
        num_trials: Number of generations per scale
        context_scales: List of scaling factors to test

    Returns:
        Dictionary with results for each scale
    """
    results = {}
    for scale in context_scales:
        print("\n" + "="*80)
        print(f"Testing Context Vector Effect == scale = {scale}")
        print("="*80)

        correct_count = 0
        generated_answers = ""

        continuation = generate_with_context_vector(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            prefix_text=prefix_text,
            context_vector=context_vector,
            context_scale=scale,
            device=device,
            max_new_tokens=256,
            temperature=0.9
        )

        generated_answers = extract_final_answer(continuation)

        # Check if correct
        def normalize_answer(ans):
            if ans is None:
                return ""
            return ans.lower().strip()

        is_correct = normalize_answer(generated_answers) == normalize_answer(correct_answer)

        status = "✅" if is_correct else "❌"
        print(f"{status} Answer={generated_answers}")

        results[scale] = generated_answers

    if is_correct:
        correct_count += 1

    return results, correct_count