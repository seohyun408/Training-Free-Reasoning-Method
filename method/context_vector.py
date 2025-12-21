import os
import numpy as np 
import torch
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional
from bert_score import score

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
    Extract hidden states from ALL layers of the model for given text.

    Args:
        model: The VLM model
        tokenizer: Tokenizer
        text: Input text
        device: Device

    Returns:
        Hidden states as numpy array, shape (num_layers, seq_len, hidden_dim)
        - num_layers: Number of transformer layers 
        - seq_len: Sequence length
        - hidden_dim: Hidden dimension size
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Get hidden states from ALL layers
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # Each tensor has shape (batch_size, seq_len, hidden_dim)
        # Index 0 is embedding layer output, indices 1~num_layers are transformer layer outputs
        all_hidden_states = outputs.hidden_states[1:]  # tuple of num_layers tensors
        stacked_hidden = torch.stack(all_hidden_states, dim=0)
        hidden_states = stacked_hidden[:, 0, :, :]

    # Convert to numpy
    hidden_states = hidden_states.cpu().numpy()  # (num_layers, seq_len, hidden_dim)
    
    print(f"[Hidden States] Extracted from {hidden_states.shape[0]} layers, "
          f"seq_len={hidden_states.shape[1]}, hidden_dim={hidden_states.shape[2]}")

    return hidden_states


def compute_pca_context_vector(
    pca_data: np.ndarray,
    n_components: int = 1
) -> np.ndarray:
    """
    Compute PCA context vector for each layer.
    
    Args:
        pca_data: numpy array with shape (num_samples, num_layers, hidden_dim)
                  or (num_samples, hidden_dim) for backward compatibility
        n_components: Number of PCA components
        
    Returns:
        Context vectors with shape (num_layers, hidden_dim) or (hidden_dim,)
    """

    if pca_data.ndim == 3:
        # Shape: (num_samples, num_layers, hidden_dim)
        num_samples, num_layers, hidden_dim = pca_data.shape
        print(f"[PCA] Computing context vectors for {num_layers} layers "
              f"with {num_samples} samples, hidden_dim={hidden_dim}")
        
        context_vectors = []
        explained_variances = []
        
        for layer_idx in range(num_layers):
            # Extract data for this layer: (num_samples, hidden_dim)
            layer_data = pca_data[:, layer_idx, :]
            
            pca = PCA(n_components=n_components)
            pca.fit(layer_data)
            
            # First principal component is the context vector for this layer
            layer_context_vector = pca.components_[0]  # (hidden_dim,)
            context_vectors.append(layer_context_vector)
            explained_variances.append(pca.explained_variance_ratio_[0])
        
        context_vectors = np.array(context_vectors)  # (num_layers, hidden_dim)
        
        return context_vectors
    
    else:
        pca = PCA(n_components=n_components)
        pca.fit(pca_data)

        # First principal component is the context vector
        context_vector = pca.components_[0]  # (hidden_dim,)

        print(f"[PCA] Explained variance ratio: {pca.explained_variance_ratio_[0]:.4f}")
        print(f"[PCA] Context vector shape: {context_vector.shape}")

        return context_vector


def generate_reasoning_with_context_vector(
    model,
    processor,
    tokenizer,
    question: str,
    image,
    context_vector: np.ndarray,
    context_scale: float,
    device: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7
) -> str:

    system_prompt = (
        "You are a helpful vision assistant analyzing images. "
        "Please provide your reasoning in <think>...</think> tags with AT LEAST 5 numbered steps (1., 2., 3., 4., 5., ...). "
        "You can include more steps if needed for thorough analysis. "
        "Each step should be a complete sentence (15-40 words) that describes specific details from the image. "
        "DO NOT use ellipsis (...) or placeholder text. Make each step concrete and informative. "
        "After reasoning, provide the final answer in <final>...</final> tags."
    )
    
    user_text = question.strip()
    
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": system_prompt}
        ]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text}
        ]}
    ]
    
    apply_tmpl = getattr(processor, "apply_chat_template", None)
    if apply_tmpl is None:
        template_text = "<image>\n" + user_text
    else:
        template_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    
    inputs = processor(text=[template_text], images=[image], return_tensors="pt")
    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs}
    else:
        inputs = dict(inputs)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # Handle both layer-wise (num_layers, hidden_dim) and single (hidden_dim,) context vectors
    context_tensor = torch.from_numpy(context_vector).float().to(device)
    is_layer_wise = context_tensor.ndim == 2  # (num_layers, hidden_dim)
    
    if is_layer_wise:
        print(f"[Context Vector] Layer-wise context vectors: {context_tensor.shape}")
    else:
        print(f"[Context Vector] Single context vector for all layers: {context_tensor.shape}")

    def make_layer_hook(layer_idx):
        """
        Create a hook for a specific layer that adds the appropriate context vector.
        """
        def add_context_hook(module, input, output):
            """
            Add layer-specific context vector to hidden states for all tokens.
            """
            # Handle different output types
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Get the context vector for this layer
            if is_layer_wise:
                layer_context = context_tensor[layer_idx]  # (hidden_dim,)
            else:
                layer_context = context_tensor  # (hidden_dim,)

            # Add context vector to all positions
            # hidden_states shape: (batch_size, seq_len, hidden_dim)
            if hidden_states.shape[-1] == layer_context.shape[0]:
                # Add scaled context vector to all tokens
                hidden_states = hidden_states + context_scale * layer_context.unsqueeze(0).unsqueeze(0)

                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states
            return output
        return add_context_hook

    # Get EOS token
    eos_id = None
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'eos_token_id'):
        eos_id = processor.tokenizer.eos_token_id

    # Register hooks on ALL decoder layers with layer-specific context vectors
    # For QWEN-VL: model.model.layers contains all transformer layers
    hook_handles = []
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            print(f"[Context Vector] Registering hooks on {num_layers} layers (scale={context_scale})")
            
            # Validate layer count if using layer-wise vectors
            if is_layer_wise and context_tensor.shape[0] != num_layers:
                print(f"[Warning] Context vector has {context_tensor.shape[0]} layers, "
                      f"but model has {num_layers} layers. Using min of both.")
            
            for layer_idx, layer in enumerate(model.model.layers):
                # Create layer-specific hook
                hook = make_layer_hook(layer_idx)
                handle = layer.register_forward_hook(hook)
                hook_handles.append(handle)

        # Generate reasoning with context vector applied to all layers
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                use_cache=True
            )

        # Decode (same as reasoning_generation.py)
        input_len = inputs.get("input_ids").shape[-1] if inputs.get("input_ids") is not None else 0
        decoder = tokenizer if hasattr(tokenizer, "batch_decode") else processor
        generated_text = decoder.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]

    finally:
        # Remove all hooks
        for handle in hook_handles:
            handle.remove()

    return generated_text


def test_context_vector_effect(
    model,
    processor,
    tokenizer,
    question: str,
    image,
    context_vector: np.ndarray,
    correct_answer: str,
    device: str,
    num_trials: int,
    context_scales: List[float]
) -> Tuple[Dict, float, float, float]:
 
    def normalize_answer(ans):
        """Normalize answer for comparison."""
        if ans is None:
            return ""
        return ans.lower().strip()

    all_results = {}
    
    print(f"\n{'='*80}")
    print(f"🎯 Testing Context Vector Effect")
    print(f"   Question: {question[:100]}...")
    print(f"   Ground Truth: {correct_answer}")
    print(f"{'='*80}\n")
    
    for scale in context_scales:
        print(f"\n{'='*80}")
        print(f"📊 Testing Scale = {scale}")
        print(f"{'='*80}")
        
        scale_results = []
        scale_correct = 0
        scale_precisions = []
        scale_recalls = []
        scale_f1s = []
        
        for trial_idx in range(num_trials):
            print(f"\n  Trial {trial_idx + 1}/{num_trials}:")
            
            # Generate reasoning with context vector applied to all layers
            reasoning = generate_reasoning_with_context_vector(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                question=question,
                image=image,
                context_vector=context_vector,
                context_scale=scale,
                device=device,
                max_new_tokens=512,
                temperature=0.7
            )
            
            generated_answer = extract_final_answer(reasoning)
            
            is_correct = normalize_answer(generated_answer) == normalize_answer(correct_answer)
            
            if is_correct:
                scale_correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} Generated: {generated_answer}")
            print(f"     Ground Truth: {correct_answer}")
            
            # Compute BERTScore
            precision, recall, f1 = compute_bertscore(generated_answer, correct_answer)
            
            print(f"  🌟 BERTScore - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
            
            scale_precisions.append(precision)
            scale_recalls.append(recall)
            scale_f1s.append(f1)
            
        accuracy = scale_correct / num_trials if num_trials > 0 else 0.0
        avg_precision = np.mean(scale_precisions) 
        avg_recall = np.mean(scale_recalls) 
        avg_f1 = np.mean(scale_f1s) 
        
        print(f"\n  📈 Scale {scale} Results:")
        print(f"     Accuracy: {scale_correct}/{num_trials} = {accuracy:.2%}")
        print(f"     BERTScore - P: {avg_precision:.4f}, R: {avg_recall:.4f}, F1: {avg_f1:.4f}")
        
        all_results[float(scale)] = {
            "accuracy": accuracy,
            "correct_count": scale_correct,
            "total_trials": num_trials,
            "bertscore": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            },
            "trials": scale_results
        }
    
    print(f"\n{'='*80}")
    print(f"📊 All Scales Completed")
    print(f"{'='*80}\n")
    
    return all_results


def compute_bertscore(sentence1: str, sentence2: str):

    cands = [sentence1]
    refs = [sentence2]

    P, R, F1 = score(cands, refs, lang="en", model_type="microsoft/deberta-large")

    return float(P[0]), float(R[0]), float(F1[0])