from typing import Dict, Optional

import os
import torch
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "whitebox-analyses"))

from attention_analysis.attn_supp_funcs import compute_suppression_kl_matrix2
from calculate_anchor import compute_anchor_vector, print_anchor_summary

from contrastive_generation import generate_from_anchor, extract_anchor_prefix
from contrastive_generation import extract_hidden_states, compute_pca_context_vector, test_context_vector_effect

from utils import extract_qa_pairs, extract_reasoning_from_response, split_solution_into_chunks, \
                get_chunk_token_ranges


class DatasetProcessor:
    def __init__(
        self,
        args,
        model,
        processor,
        tokenizer,
        device: str,
        images_root: str,
        generate_reasoning: bool,
        model_name: str
    ):
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.images_root = images_root
        self.generate_reasoning = generate_reasoning
        self.model_name = model_name


    def process_sample(self, example):
        """
        Input: {
            'id': ..., 
            'image': ...,
            'conversations': [{'from': 'human', 'value': ...}, {'from': 'gpt', 'value': ...}]
            }
        """
        conversation = example.get("conversations", [])
        qa_pairs = extract_qa_pairs(conversation)
        
        image_file = example.get("image")
        image_path = os.path.join(self.images_root, image_file)
        image = Image.open(image_path).convert("RGB")
        print("image_path >>> ", image_path)
        
        # Process each QA pair
        results = []
        for pair_idx, (question, gpt_response) in enumerate(qa_pairs):
            result = self.process_qa_pair(
                question=question,
                gpt_response=gpt_response,
                image=image,
                pair_idx=pair_idx
            )
            
        return {
            "total_pairs": len(qa_pairs),
            "successful_pairs": len(results),
            "image_path": image_path,
            "qa_pairs": results
        }


    def process_qa_pair(
        self,
        question: str,
        gpt_response: str,
        image,
        pair_idx,
    ) -> Dict:

        print("Generating reasoning with model...")
        if self.generate_reasoning:
            reasoning_source_text, raw_generations_local, template_text_used = _generate_reasoning_with_model(
                question=question,
                image=image,
                model=self.model,
                processor=self.processor,
                tokenizer=self.tokenizer,
                device=self.device,
                model_name=self.model_name
            )
        else:
            reasoning_source_text = gpt_response

        reasoning_text = extract_reasoning_from_response(reasoning_source_text)

        placeholder = _image_placeholder_for_model(self.model_name)
        if placeholder not in question:
            if placeholder != "<image>" and "<image>" in question:
                question = question.replace("<image>", placeholder)
            else:
                question = placeholder + "\n" + question

        full_text = question + "\n\n" + reasoning_text
        chunks = split_solution_into_chunks(reasoning_text, tokenizer=self.tokenizer, min_tokens=5)
        print(f"  Pair {pair_idx}: Split into {len(chunks)} chunks")

        # find where reasoning_text starts in full_text
        reasoning_start_idx = full_text.find(reasoning_text)
        if reasoning_start_idx == -1:
            if chunks:
                first_chunk_idx = full_text.find(chunks[0])
                if first_chunk_idx != -1:
                    reasoning_start_idx = first_chunk_idx

        # positions to token indices for chunks.
        chunk_token_ranges_reasoning = get_chunk_token_ranges(
            reasoning_text, 
            chunks, 
            self.tokenizer
        )
        if len(chunk_token_ranges_reasoning) != len(chunks):
            return {"error": f"Token range mismatch for pair {pair_idx}"}
        
        # Convert to token ranges in full_text
        tokens_before_reasoning = self.tokenizer.encode(full_text[:reasoning_start_idx], add_special_tokens=False)
        offset = len(tokens_before_reasoning)
        
        chunk_token_ranges = [(start + offset, end + offset) for start, end in chunk_token_ranges_reasoning]
        

        # ========================================
        #   STAGE 1: Anchor Detection  
        # ========================================

        print(f"  Pair {pair_idx}: Computing suppression KL matrix...")

        kl_matrix = compute_suppression_kl_matrix2(
            model=self.model,
            processor=self.processor,
            tokenizer=self.tokenizer,
            text=full_text,
            image=image,
            chunks=chunks,
            chunk_token_ranges=chunk_token_ranges,
            device=self.device
        )

        # KL Matrix[i, j] = 문장 j를 억제했을 때 문장 i에 미치는 영향 (KL divergence)
        # Anchor Vector[i] = 문장 i가 후속 문장들에 미치는 평균 영향 (높을수록 중요한 anchor)
        
        anchor_vector = compute_anchor_vector(kl_matrix, method="outgoing")
        
        # Print detailed anchor analysis
        print_anchor_summary(
            chunks=chunks,
            anchor_vector=anchor_vector,
            kl_matrix=kl_matrix,
            top_k=min(5, len(chunks))  # 상위 5개 또는 전체 문장 수
        )
        

        # ========================================
        #   Step 2: Contrastive Generation
        # ========================================

        # Debubgging하면서 데이터 확인

        contrastive_result = None

        # Extract correct answer from gpt_response or reasoning
        # Try to find answer in the original response or generated reasoning
        correct_answer = None
        # Check dataset ground truth first, then fallback to generated reasoning
        for source_text in [gpt_response, reasoning_source_text]:
            for tag_pair in [("<final>", "</final>"), ("<CONCLUSION>", "</CONCLUSION>")]:
                start_tag, end_tag = tag_pair
                if start_tag in source_text and end_tag in source_text:
                    start = source_text.find(start_tag) + len(start_tag)
                    end = source_text.find(end_tag)
                    correct_answer = source_text[start:end].strip()
                    break
            if correct_answer:
                break

        if correct_answer and len(chunks) > 1:
            print(f"\n[contrastive] Correct answer: {correct_answer}")

            # Get prefix up to anchor
            prefix_text, anchor_idx, anchor_sentence = extract_anchor_prefix(
                question=question,
                reasoning_text=reasoning_text,
                chunks=chunks,
                anchor_vector=anchor_vector
            )

            print(f"[contrastive] Anchor sentence (idx={anchor_idx}): {anchor_sentence[:100]}...")

            # Generate contrastive samples
            num_samples = int(os.getenv("CONTRASTIVE_SAMPLES", "5"))
            contrastive_result = generate_from_anchor(
                model=self.model,
                processor=self.processor,
                tokenizer=self.tokenizer,
                prefix_text=prefix_text,
                image=image,
                device=self.device,
                correct_answer=correct_answer,
                num_samples=num_samples,
                max_new_tokens=128,
                temperature=0.9,
                top_p=0.95
            )

            print(f"[contrastive] Generated {len(contrastive_result['all_samples'])} samples")

            # ========================================
            #   PCA Context Vector Testing
            # ========================================
            if os.getenv("TEST_PCA_CONTEXT", "0") in {"1", "true", "True"}:



                print("\n" + "="*80)
                print("🧪 PCA Context Vector Testing")
                print("="*80)

                # Get positive and negative full texts
                positive_full = contrastive_result['positive_full']
                negative_full = contrastive_result['negative_full']

                # Clean texts (remove vision tokens)
                def clean_text(text):
                    for vision_token in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]:
                        text = text.replace(vision_token, "")
                    return text.strip()

                clean_prefix = clean_text(prefix_text)
                positive_text = clean_prefix + "\n\nContinue the next reasoning step:" + positive_full
                negative_text = clean_prefix + "\n\nContinue the next reasoning step:" + negative_full

                # Extract hidden states
                print("[PCA] Extracting hidden states from positive continuation...")
                positive_hidden = extract_hidden_states(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    text=positive_text,
                    device=self.evice
                )

                print("[PCA] Extracting hidden states from negative continuation...")
                negative_hidden = extract_hidden_states(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    text=negative_text,
                    device=self.device
                )

                # Compute PCA context vector
                print("[PCA] Computing PCA context vector...")
                context_vector = compute_pca_context_vector(
                    positive_hidden=positive_hidden,
                    negative_hidden=negative_hidden,
                    n_components=1
                )

                # Test context vector effect
                pca_results = test_context_vector_effect(
                    model=self.model,
                    processor=self.processor,
                    tokenizer=self.tokenizer,
                    prefix_text=prefix_text,
                    positive_full=positive_full,
                    negative_full=negative_full,
                    context_vector=context_vector,
                    correct_answer=correct_answer,
                    device=self.device,
                    num_trials=int(os.getenv("PCA_NUM_TRIALS", "3")),
                    context_scales=[0.0, 0.5, 1.0, 2.0, 5.0]  # 5.0까지만 (더 크면 over-steering)
                )

                # Add PCA results to contrastive_result
                contrastive_result['pca_context'] = {
                    "context_vector_shape": context_vector.shape,
                    "results": {str(k): v for k, v in pca_results.items()}
                }


        return {
            "pair_idx": pair_idx,
            "question": question,
            "reasoning_text": reasoning_text,
            "raw_generation": raw_generations_local or None,
            "chunks": chunks,
            "chunk_token_ranges": chunk_token_ranges,
            "anchor_vector": anchor_vector.tolist() if hasattr(anchor_vector, "tolist") else anchor_vector,
            "kl_matrix_shape": getattr(kl_matrix, "shape", None),
            "contrastive": contrastive_result
        }


# ======================================================================================

def _image_placeholder_for_model(model_name: str) -> str:
    name = (model_name or "").lower()
    # Qwen3-VL expects explicit vision start/end + image pad placeholders in text when using processor(text=..., images=...)
    if "qwen3-vl" in name or "qwen" in name:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    # Default (e.g., LLaVA-style)
    return "<image>"


def _generate_reasoning_with_model(question, image, model, processor, tokenizer, device, model_name, max_new_tokens=256):
    """Generate reasoning with controlled prompt (<think>/<final>) and sanitation."""
    MAX_NEW_ENV = int(os.getenv("MAX_NEW_TOKENS", str(max_new_tokens)))
    # Remove hard 96 cap to allow longer reasoning when requested
    first_pass_max = MAX_NEW_ENV

    system_prompt = (
        "You are a helpful vision assistant analyzing images. "
        "Please provide your reasoning in <think>...</think> tags with AT LEAST 5 numbered steps (1., 2., 3., 4., 5., ...). "
        "You can include more steps if needed for thorough analysis. "
        "Each step should be a complete sentence (15-40 words) that describes specific details from the image. "
        "DO NOT use ellipsis (...) or placeholder text. Make each step concrete and informative. "
        "After reasoning, provide the final answer in <final>...</final> tags."
    )
    alt_system_prompt = (
        "Analyze the image carefully. Write at least 5 numbered reasoning steps (1., 2., 3., 4., 5., ...) inside <think> tags. "
        "More steps are welcome if they add value. "
        "Each step must be a full sentence with specific details from the image. "
        "Never use ... or generic phrases. Each sentence should be 15-40 words. "
        "End with your answer in <final> tags."
    )
    user_text = question.strip()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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

    eos_id = None
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'eos_token_id'):
        eos_id = processor.tokenizer.eos_token_id

    def _run_generate(cur_max_new: int, attempt: int, *, allow_sampling: bool):
        with torch.no_grad():
            try:
                return model.generate(
                    **inputs,
                    max_new_tokens=cur_max_new,
                    do_sample=allow_sampling,
                    temperature=0.8 if allow_sampling else None,
                    top_p=0.9 if allow_sampling else None,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                    use_cache=True
                )
            except RuntimeError as e:
                print(f"[gen-error] attempt={attempt}: {e}")
                raise

    is_thinking = "thinking" in (model_name or "").lower() or os.getenv("DO_SAMPLE", "0") == "1"
    # Attempt loop with fallback prompts
    max_attempts = int(os.getenv("REASONING_ATTEMPTS", "2"))
    raw_generations = []
    attempt_prompts = [system_prompt, alt_system_prompt]
    final_generated_text = ""
    for attempt_idx in range(max_attempts):
        cur_prompt = attempt_prompts[min(attempt_idx, len(attempt_prompts)-1)]
        # Rebuild messages with alternate prompt (keep image & question)
        messages[0]["content"][0]["text"] = cur_prompt
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
        gen_output = _run_generate(first_pass_max, attempt=attempt_idx, allow_sampling=is_thinking or attempt_idx>0)
        generated_ids = gen_output

        input_len = inputs.get("input_ids").shape[-1] if inputs.get("input_ids") is not None else 0
        decoder = tokenizer if hasattr(tokenizer, "batch_decode") else processor
        try:
            gen_new = decoder.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        except Exception:
            gen_new = ""
        raw_generations.append(gen_new)
        # Basic quality checks
        def count_numbered_steps(s: str) -> int:
            return sum(1 for n in ["1.", "2.", "3."] if n in s)
        if ("<think>" in gen_new and count_numbered_steps(gen_new) >= 3 and "..." not in gen_new) or attempt_idx == max_attempts-1:
            final_generated_text = gen_new
            break

    generated_text = final_generated_text or raw_generations[-1]
    input_len = inputs.get("input_ids").shape[-1] if inputs.get("input_ids") is not None else 0
    decoder = tokenizer if hasattr(tokenizer, "batch_decode") else processor
    try:
        gen_new = decoder.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
    except Exception:
        gen_new = ""

    def looks_garbage(s: str) -> bool:
        if not s:
            return True
        bad_mark = "\u001a"
        repl = "\ufffd"
        error_tokens = s.count("_error")
        non_printable = sum(ord(c) < 9 or (13 < ord(c) < 32) for c in s)
        return (error_tokens >= 3) or (repl in s) or (bad_mark in s) or (non_printable > 0)

    def too_short(s: str) -> bool:
        return len(s.strip()) < 40 or s.count('.') < 2

    # Quality fallback: if no <think> present, wrap numbered extraction attempt
    if "<think>" not in generated_text and count_numbered_steps(generated_text) >= 2:
        # Synthesize <think> wrapper for downstream extraction
        think_body = generated_text.strip()
        generated_text = f"<think>{think_body}</think>"
        print("[inject] Added synthetic <think> wrapper.")
    
    # Log raw generations for debugging (truncated)
    print("[gen-debug] Attempts:")
    for i, rg in enumerate(raw_generations):
        cleaned = rg[:180].replace('\n', ' ')
        print(f"  attempt {i}: {cleaned}")

    # Early truncation at </think>
    if "</think>" in generated_text:
        think_end = generated_text.find("</think>") + len("</think>")
        tail = generated_text[think_end:]
        if "<final>" in tail and "</final>" in tail:
            final_start = tail.find("<final>")
            final_end = tail.find("</final>") + len("</final>")
            generated_text = generated_text[:think_end] + tail[final_start:final_end]
        else:
            generated_text = generated_text[:think_end]

    return generated_text, raw_generations, template_text