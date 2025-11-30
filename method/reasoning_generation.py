import os
import torch


def image_placeholder_for_model(model_name: str) -> str:
    name = (model_name or "").lower()
    if "qwen3-vl" in name or "qwen" in name:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    return "<image>"


def generate_reasoning_with_model(args, question, image, model, processor, tokenizer, device, model_name, max_new_tokens=256):
    """
    Generate reasoning with controlled prompt (<think>/<final>) and sanitation.
    """
    first_pass_max = max_new_tokens

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
        {"role": "system", "content": [
            {"type": "text", "text": system_prompt}
        ]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text}
        ]}
    ]


    eos_id = None
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'eos_token_id'):
        eos_id = processor.tokenizer.eos_token_id


    def _run_generate(cur_max_new: int, attempt: int, *, allow_sampling: bool):
        with torch.no_grad():
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

    
    is_thinking = "thinking" in (model_name or "").lower() or os.getenv("DO_SAMPLE", "0") == "1"
    
    # Attempt loop with fallback prompts
    max_attempts = int(args.max_attempts)
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
        input_len = inputs.get("input_ids").shape[-1] if inputs.get("input_ids") is not None else 0
        decoder = tokenizer if hasattr(tokenizer, "batch_decode") else processor
        gen_new = decoder.batch_decode(gen_output[:, input_len:], skip_special_tokens=True)[0]
        raw_generations.append(gen_new)

        # Basic quality checks
        def _count_numbered_steps(s: str) -> int:
            return sum(1 for n in ["1.", "2.", "3."] if n in s)
        
        if ("<think>" in gen_new and _count_numbered_steps(gen_new) >= 3 and "..." not in gen_new) or attempt_idx == max_attempts-1:
            final_generated_text = gen_new
            break

    generated_text = final_generated_text or raw_generations[-1]

    def _looks_garbage(s: str) -> bool:
        if not s:
            return True
        bad_mark = "\u001a"
        repl = "\ufffd"
        error_tokens = s.count("_error")
        non_printable = sum(ord(c) < 9 or (13 < ord(c) < 32) for c in s)
        return (error_tokens >= 3) or (repl in s) or (bad_mark in s) or (non_printable > 0)

    def _too_short(s: str) -> bool:
        return len(s.strip()) < 40 or s.count('.') < 2

    # Quality fallback: if no <think> present, wrap numbered extraction attempt
    if "<think>" not in generated_text and _count_numbered_steps(generated_text) >= 2:
        # Synthesize <think> wrapper for downstream extraction
        think_body = generated_text.strip()
        generated_text = f"<think>{think_body}</think>"
        print("[inject] Added synthetic <think> wrapper.")
    
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