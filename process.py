from typing import Dict, Optional

import os
from PIL import Image
import torch
import importlib.util
import pathlib

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "whitebox-analyses"))

# Robustly import modules from whitebox-analyses (directory has hyphen, not a package)
try:
    # Attempt absolute-style dynamic import (directory not a package, so may fail)
    from attention_analysis.attn_supp_funcs import compute_suppression_kl_matrix2  # type: ignore
    from calculate_anchor import compute_anchor_vector, print_anchor_summary  # type: ignore
except Exception:
    _base = os.path.join(os.path.dirname(__file__), "whitebox-analyses")
    _attn_path = os.path.join(_base, "attention_analysis", "attn_supp_funcs.py")
    _calc_path = os.path.join(_base, "calculate_anchor.py")

    def _load_from_path(module_name: str, file_path: str):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module

    _attn_mod = _load_from_path("attn_supp_funcs", _attn_path)
    _calc_mod = _load_from_path("calculate_anchor", _calc_path)

    compute_suppression_kl_matrix2 = getattr(_attn_mod, "compute_suppression_kl_matrix2")  # noqa: E305
    compute_anchor_vector = getattr(_calc_mod, "compute_anchor_vector")  # noqa: E305
    print_anchor_summary = getattr(_calc_mod, "print_anchor_summary")  # noqa: E305

from utils import extract_qa_pairs, extract_reasoning_from_response, split_solution_into_chunks, \
                get_chunk_token_ranges, get_chunk_token_ranges_with_offsets

def _image_placeholder_for_model(model_name: str) -> str:
    name = (model_name or "").lower()
    # Qwen3-VL expects explicit vision start/end + image pad placeholders in text when using processor(text=..., images=...)
    if "qwen3-vl" in name or "qwen" in name:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    # Default (e.g., LLaVA-style)
    return "<image>"

def process_dataset_sample(
    example: Dict,
    model,
    processor,
    tokenizer,
    device: str = "cuda",
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    cache_dir: Optional[str] = None,
    images_root: Optional[str] = None,
    generate_reasoning: bool = True,
) -> Dict:

    """
    Input: {
        'id': ..., 
        'image': ...,
        'conversations': [{'from': 'human', 'value': ...}, {'from': 'gpt', 'value': ...}]
        }
    """

    conversation = example.get("conversations", [])
    qa_pairs = extract_qa_pairs(conversation)
    
    # Resolve image (dataset may provide PIL image when cast; else resolve path)
    image_field = example.get("image")
    image = None
    image_path = None
    if isinstance(image_field, Image.Image):
        image = image_field.convert("RGB")
        print("image: provided as PIL Image from dataset")
    elif isinstance(image_field, str):
        # Attempt resolution via provided root and fallback searches
        candidates = []
        # Heuristic prefix remapping (dataset uses various roots)
        prefix_map = {
            "sqa/": "sqa",  # example: sqa/train/20839/image.png -> extracted/sqa/train/20839/image.png
            "chartqa/": "chartqa",  # chartqa/train/png/... -> extracted/chartqa/train/png/...
            "geoqa+/": "geoqa+",  # geoqa+/images/... -> extracted/geoqa+/images/...
        }
        remapped_field = image_field
        for pfx, sub in prefix_map.items():
            if image_field.startswith(pfx):
                remapped_field = image_field  # direct join already fine; kept for extensibility
                break

        if images_root:
            candidates.append(os.path.join(images_root, remapped_field))
        candidates.append(remapped_field)  # as-is
        hf_cache = os.getenv("HF_DATASETS_CACHE")
        if hf_cache:
            candidates.append(os.path.join(hf_cache, remapped_field))

        found_path = None
        for cand in candidates:
            if os.path.isfile(cand):
                found_path = cand
                break

        # Fallback: recursive search (expensive) limited to depth if not found
        if found_path is None and images_root:
            try:
                # Search by filename only (e.g., image.png or multi_col_100056.png)
                filename = os.path.basename(image_field)
                # Limit to first match
                for root, dirs, files in os.walk(images_root):
                    if filename in files:
                        found_path = os.path.join(root, filename)
                        break
            except Exception:
                pass
        if found_path is None:
            print(f"[warn] Image not found for field '{image_field}'. Tried: {candidates}. Skipping example.")
            return {
                "total_pairs": 0,
                "successful_pairs": 0,
                "image_path": image_field,
                "qa_pairs": [],
                "error": "image_not_found"
            }
        image_path = found_path
        image = Image.open(image_path).convert("RGB")
        print("image_path >>> ", image_path)
    else:
        raise ValueError(f"Unsupported image field type: {type(image_field)}")
    

    # Process each QA pair
    results = []
    for pair_idx, (question, gpt_response) in enumerate(qa_pairs):
        pair_result = process_qa_pair(
            question=question,
            gpt_response=gpt_response,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image=image,
            device=device,
            pair_idx=pair_idx,
            generate_reasoning=generate_reasoning,
            model_name=model_name,
        )
        
        if isinstance(pair_result, dict) and "error" not in pair_result:
            results.append(pair_result)
    return {
        "total_pairs": len(qa_pairs),
        "successful_pairs": len(results),
        "image_path": image_path,
        "qa_pairs": results
    }


def _generate_reasoning_with_model(question, image, model, processor, tokenizer, device, model_name, max_new_tokens=256):
    """Generate reasoning with controlled prompt (<think>/<final>) and sanitation."""
    MAX_NEW_ENV = int(os.getenv("MAX_NEW_TOKENS", str(max_new_tokens)))
    # Remove hard 96 cap to allow longer reasoning when requested
    first_pass_max = MAX_NEW_ENV

    system_prompt = (
        "You are a helpful vision assistant analyzing images. "
        "Please provide your reasoning in <think>...</think> tags with EXACTLY 3 numbered steps (1., 2., 3.). "
        "Each step should be a complete sentence (15-40 words) that describes specific details from the image. "
        "DO NOT use ellipsis (...) or placeholder text. Make each step concrete and informative. "
        "After reasoning, provide the final answer in <final>...</final> tags."
    )
    alt_system_prompt = (
        "Analyze the image carefully. Write 3 numbered reasoning steps (1., 2., 3.) inside <think> tags. "
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
                    use_cache=True,
                )
            except RuntimeError as e:
                print(f"[gen-error] attempt={attempt}: {e}")
                raise

    is_thinking = "thinking" in (model_name or "").lower()
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
        generated_ids = _run_generate(first_pass_max, attempt=attempt_idx, allow_sampling=is_thinking or attempt_idx>0)
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


def process_qa_pair(
    question: str,
    gpt_response: str,
    model,
    processor,
    tokenizer,
    image,
    device: str = "cuda",
    pair_idx: int = 0,
    generate_reasoning: bool = True,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> Dict:
    # Collect raw generations from reasoning model attempts (populated inside _generate if regeneration triggers)
    raw_generations_local = []

    # Reasoning 텍스트 생성 또는 기존 응답에서 추출
    if generate_reasoning:
        print("Generating reasoning with model...")
        reasoning_source_text, raw_generations_local, template_text_used = _generate_reasoning_with_model(
            question=question,
            image=image,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model_name=model_name
        )
    else:
        reasoning_source_text = gpt_response

    print("Total Text >>> ", reasoning_source_text, '\n')
    reasoning_text = extract_reasoning_from_response(reasoning_source_text)
    
    # Inject model-appropriate image placeholder into the question text (for later whitebox analysis tokenization)
    placeholder = _image_placeholder_for_model(model_name)
    if placeholder not in question:
        # If legacy token present and model is Qwen, replace it
        if placeholder != "<image>" and "<image>" in question:
            question = question.replace("<image>", placeholder)
        else:
            question = placeholder + "\n" + question
    
    # Early regeneration check BEFORE constructing full_text for downstream token alignment
    if reasoning_text.strip() == '...':
        print('[regen] Initial reasoning is placeholder "..." -> attempting regeneration.')
        try:
            regen_text, regen_raws, _ = _generate_reasoning_with_model(
                question=question,
                image=image,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                model_name=model_name,
                max_new_tokens=int(os.getenv('REGEN_MAX_NEW_TOKENS', '160'))
            )
            reasoning_text_candidate = extract_reasoning_from_response(regen_text)
            if reasoning_text_candidate.strip() != '...':
                reasoning_text = reasoning_text_candidate
                raw_generations_local.extend(regen_raws)
                print('[regen] First regeneration succeeded.')
            else:
                print('[regen] First regeneration still placeholder; will proceed (anchors disabled).')
        except Exception as e:
            print(f'[regen] Regeneration exception: {e}')

    full_text = question + "\n\n" + reasoning_text
    chunks = split_solution_into_chunks(reasoning_text, tokenizer=tokenizer, min_tokens=5)
    max_chunks_env = os.getenv("MAX_CHUNKS")
    if max_chunks_env is not None:
        try:
            k = int(max_chunks_env)
            if k >= 0:
                chunks = chunks[:k]
        except Exception:
            pass
    print(f"  Pair {pair_idx}: Split into {len(chunks)} chunks")
    print(f"  Chunks preview: {[c[:50] + '...' if len(c) > 50 else c for c in chunks]}")
    
    reasoning_start_idx = full_text.find(reasoning_text)
    if reasoning_start_idx == -1:
        if chunks:
            first_chunk_idx = full_text.find(chunks[0])
            if first_chunk_idx != -1:
                reasoning_start_idx = first_chunk_idx

    # positions to token indices for chunks.
    # Prefer offset-based alignment for accurate suppression ranges
    chunk_token_ranges_reasoning = get_chunk_token_ranges_with_offsets(
        full_text=full_text,
        reasoning_start_char=reasoning_start_idx if reasoning_start_idx != -1 else full_text.find(reasoning_text),
        reasoning_text=reasoning_text,
        chunks=chunks,
        tokenizer=tokenizer,
    )
    if len(chunk_token_ranges_reasoning) != len(chunks):
        return {"error": f"Token range mismatch for pair {pair_idx}"}
    
    # Convert to token ranges in full_text
    tokens_before_reasoning = tokenizer.encode(full_text[:reasoning_start_idx], add_special_tokens=False)
    offset = len(tokens_before_reasoning)
    
    chunk_token_ranges = [(start + offset, end + offset) for start, end in chunk_token_ranges_reasoning]
    

    # ========================================
    #        STAGE 1: Anchor Detection  
    # ========================================

    print(f"  Pair {pair_idx}: Computing suppression KL matrix...")

    if os.getenv("SKIP_SUPPRESSION", "0") in {"1", "true", "True"}:
        print("[fast] SKIP_SUPPRESSION=1 -> skipping KL computation.")
        return {
            "pair_idx": pair_idx,
            "question": question,
            "reasoning_text": reasoning_text,
            "chunks": chunks,
            "chunk_token_ranges": chunk_token_ranges,
            "anchor_vector": [0.0 for _ in chunks],
            "kl_matrix_shape": (len(chunks), len(chunks))
        }

    kl_matrix = compute_suppression_kl_matrix2(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        text=full_text,
        image=image,
        chunks=chunks,
        chunk_token_ranges=chunk_token_ranges,
        device=device,
        suppression_strategy=os.getenv("SUPPRESSION_STRATEGY")
    )

    # KL Matrix[i, j] = 문장 j를 억제했을 때 문장 i에 미치는 영향 (KL divergence)
    # Anchor Vector[i] = 문장 i가 후속 문장들에 미치는 평균 영향 (높을수록 중요한 anchor)
    
    anchor_vector = compute_anchor_vector(kl_matrix, method=os.getenv("ANCHOR_METHOD", "outgoing"))
    
    # Print detailed anchor analysis
    print_anchor_summary(
        chunks=chunks,
        anchor_vector=anchor_vector,
        kl_matrix=kl_matrix,
        top_k=min(5, len(chunks))  # 상위 5개 또는 전체 문장 수
    )
    
    # Debug: show decoded text for each chunk token range (first pair only to limit spam)
    if pair_idx == 0:
        try:
            full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
            for ci,(s,e) in enumerate(chunk_token_ranges):
                s_eff = min(s, full_ids.shape[-1])
                e_eff = min(e, full_ids.shape[-1])
                snippet = tokenizer.decode(full_ids[0, s_eff:e_eff]) if s_eff < e_eff else "<empty>"
                print(f"[chunk {ci}] token_range=({s},{e}) snippet='{snippet[:100]}'")
        except Exception as e:
            print(f"[debug] snippet decode failed: {e}")

    return {
        "pair_idx": pair_idx,
        "question": question,
        "reasoning_text": reasoning_text,
    "raw_generation": raw_generations_local or None,
        "chunks": chunks,
        "chunk_token_ranges": chunk_token_ranges,
        "anchor_vector": anchor_vector.tolist() if hasattr(anchor_vector, "tolist") else anchor_vector,
        "kl_matrix_shape": getattr(kl_matrix, "shape", None)
    }

    # ========================================
    # Step 2: Positive/Negative Sampling (TODO)
    # ========================================
