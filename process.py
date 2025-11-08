from typing import Dict, Optional

import os
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "whitebox-analyses"))

from attention_analysis.attn_supp_funcs import compute_suppression_kl_matrix2
from calculate_anchor import compute_anchor_vector, print_anchor_summary

from utils import extract_qa_pairs, extract_reasoning_from_response, split_solution_into_chunks, \
                get_chunk_token_ranges

def process_dataset_sample(
    example: Dict,
    model,
    processor,
    tokenizer,
    device: str = "cuda",
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    cache_dir: Optional[str] = None
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
    
    IMAGE_DIR = os.getenv('IMAGE_DIR')
    image_file = example.get("image")
    image_path = os.path.join(IMAGE_DIR, image_file)
    image = Image.open(image_path).convert("RGB")
    print("image_path >>> ", image_path)
    

    # Process each QA pair
    results = []
    for pair_idx, (question, gpt_response) in enumerate(qa_pairs):
        result = process_qa_pair(
            question=question,
            gpt_response=gpt_response,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image=image,
            device=device,
            pair_idx=pair_idx
        )
        
    return {
        "total_pairs": len(qa_pairs),
        "successful_pairs": len(results),
        "image_path": image_path,
        "qa_pairs": results
    }


def process_qa_pair(
    question: str,
    gpt_response: str,
    model,
    processor,
    tokenizer,
    image,
    device: str = "cuda",
    pair_idx: int = 0
) -> Dict:

    # Reasoning만 활용 유무 
    print("Total Text >>> ", gpt_response, '\n')
    reasoning_text = extract_reasoning_from_response(gpt_response)
    #reasoning_text = gpt_response
    
    # Add <image> token 
    if "<image>" not in question:
        question = "<image>\n" + question
    
    full_text = question + "\n\n" + reasoning_text
    print("full text >>> ", full_text)
    chunks = split_solution_into_chunks(reasoning_text)
    print(f"  Pair {pair_idx}: Split into {len(chunks)} chunks")
    
    # find where reasoning_text starts in full_text
    reasoning_start_idx = full_text.find(reasoning_text)
    if reasoning_start_idx == -1:
        if chunks:
            first_chunk_idx = full_text.find(chunks[0])
            if first_chunk_idx != -1:
                reasoning_start_idx = first_chunk_idx

    # positions to token indices for chunks.
    chunk_token_ranges_reasoning = get_chunk_token_ranges(reasoning_text, chunks, tokenizer)
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

    kl_matrix = compute_suppression_kl_matrix2(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        text=full_text,
        image=image,
        chunks=chunks,
        chunk_token_ranges=chunk_token_ranges,
        device=device
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
    # Step 2: Positive/Negative Sampling
    # ========================================
