from typing import Dict, Optional

import os
import numpy as np
import torch
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "whitebox-analyses"))
sys.path.append(os.path.join(os.path.dirname(__file__), "method"))

from reasoning_generation import *

from attention_analysis.attn_supp_funcs import compute_suppression_kl_matrix2
from calculate_anchor import compute_anchor_vector, print_anchor_summary



from contrastive_generation import generate_from_anchor, extract_anchor_prefix
from context_vector import extract_hidden_states, compute_pca_context_vector, test_context_vector_effect



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
            pair_result = self.process_qa_pair(
                question=question,
                gpt_response=gpt_response,
                image=image,
                pair_idx=pair_idx
            )
            results.append(pair_result)
            
        print('\n')
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


        # ========================================
        #   STAGE 0: Generate Reasoning   
        # ========================================

        print(f"\n🌈 Stage 0: Generate Reasoning...")

        raw_generations_local = None
        if self.generate_reasoning:
            reasoning_source_text, raw_generations_local, template_text_used = generate_reasoning_with_model(
                args=self.args,
                question=question,
                image=image,
                model=self.model,
                processor=self.processor,
                tokenizer=self.tokenizer,
                device=self.device,
                model_name=self.model_name,
                max_new_tokens=self.args.max_new_tokens
            )
        else:
            reasoning_source_text = gpt_response

        reasoning_text = extract_reasoning_from_response(reasoning_source_text)

        # Qwen-VL : <|vision_start|><|image_pad|><|vision_end|>
        placeholder = image_placeholder_for_model(self.model_name)
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

        print("="*80)
        print(f"\n🌈 Stage 1: Anchor Detection ...")

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
        #   STAGE 2: Contrastive Generation
        # ========================================

        print("="*80)
        print(f"\n🌈 Stage 2: Contrastive Generation ...")

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
            contrastive_result = generate_from_anchor(
                model=self.model,
                processor=self.processor,
                tokenizer=self.tokenizer,
                prefix_text=prefix_text,
                image=image,
                device=self.device,
                correct_answer=correct_answer,
                num_samples=self.args.contrastive_samples,
                max_new_tokens=128,
                temperature=0.9,
                top_p=0.95
            )

            print(f"[contrastive] Generated {len(contrastive_result['all_samples'])} samples")

            # ========================================
            #   STAGE 3. PCA Context Vector Extraction
            # ========================================

            print("="*80)
            print("\n🌈 STAGE 3. PCA Context Vector Testing")
            
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

            # Extract hidden states from ALL layers
            print("Extracting hidden states from positive continuation...")
            positive_hidden = extract_hidden_states(
                model=self.model,
                tokenizer=self.tokenizer,
                text=positive_text,
                device=self.device
            )
            # positive_hidden shape: (num_layers, seq_len, hidden_dim)

            print("Extracting hidden states from negative continuation...")
            negative_hidden = extract_hidden_states(
                model=self.model,
                tokenizer=self.tokenizer,
                text=negative_text,
                device=self.device
            )
            # negative_hidden shape: (num_layers, seq_len, hidden_dim)

            # Compute difference for ALL layers at the last token position
            # hidden_diff shape: (num_layers, hidden_dim)
            hidden_diff = positive_hidden[:, -1, :] - negative_hidden[:, -1, :]
            
            print(f"[Hidden Diff] Computed difference across {hidden_diff.shape[0]} layers, "
                  f"hidden_dim={hidden_diff.shape[1]}")
            
            save_dir = os.path.join(os.path.dirname(__file__), "pca_data")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "vector_generation.npy")
            
            if os.path.exists(save_path):
                existing_data = np.load(save_path, allow_pickle=True)
                # Each sample has shape (num_layers, hidden_dim)
                if existing_data.ndim == 3:
                    # existing_data shape: (num_samples, num_layers, hidden_dim)
                    all_diffs = list(existing_data) + [hidden_diff]
                else:
                    all_diffs = list(existing_data) + [hidden_diff]
                print(f"[PCA] Loaded existing data. Total samples: {len(all_diffs)}")
            else:
                all_diffs = [hidden_diff]
            
            np.save(save_path, np.array(all_diffs, dtype=object))


        #     # Add PCA results to contrastive_result
        #     contrastive_result['pca_context'] = {
        #         "context_vector_shape": context_vector.shape,
        #         "results": {str(k): v for k, v in pca_results.items()}
        #     }

        #     print("/n")
        #     print(f">>>>>> Ground Truth: {correct_answer}")
        #     print(f">>>>>> Generate: {pca_results}")


        return {
            "pair_idx": pair_idx,
            "question": question,
            "reasoning_text": reasoning_text,
            "raw_generation": raw_generations_local,
            "chunks": chunks,
            "chunk_token_ranges": chunk_token_ranges,
            "anchor_vector": anchor_vector.tolist() if hasattr(anchor_vector, "tolist") else anchor_vector,
            "kl_matrix_shape": getattr(kl_matrix, "shape", None),
            "contrastive": contrastive_result,
        }


    def process_pca(self, pca_data):
        """
        Process PCA data to compute context vectors for all layers.
        
        Args:
            pca_data: numpy array with shape (num_samples, num_layers, hidden_dim)
                      or object array where each element is (num_layers, hidden_dim)
        
        Returns:
            context_vector: numpy array with shape (num_layers, hidden_dim)
        """
        if pca_data.dtype == object:
            # Object array - each element is (num_layers, hidden_dim)
            print(f"[PCA] Sample 0 shape: {pca_data[0].shape}")
            num_samples = len(pca_data)
            
            # Check if samples are layer-wise (2D: num_layers x hidden_dim)
            if pca_data[0].ndim == 2:
                # Stack into (num_samples, num_layers, hidden_dim)
                stacked_data = np.stack([sample for sample in pca_data], axis=0)
                print(f"[PCA] Stacked layer-wise data shape: {stacked_data.shape}")
            else:
                # Backward compatibility: 1D samples (hidden_dim,)
                stacked_data = np.vstack([sample.reshape(1, -1) for sample in pca_data])
                print(f"[PCA] Stacked single-layer data shape: {stacked_data.shape}")
        else:
            stacked_data = pca_data
            print(f"[PCA] Input data shape: {stacked_data.shape}")
        
        context_vector = compute_pca_context_vector(
            pca_data=stacked_data,
            n_components=1
        )
        
        print(f"[PCA] Final context vector shape: {context_vector.shape}")
        
        return context_vector


    def evaluate_with_context_vector(self, context_vector, example):

        # Extract QA pairs and image
        conversation = example.get("conversations", [])
        qa_pairs = extract_qa_pairs(conversation)
        
        image_file = example.get("image")
        image_path = os.path.join(self.images_root, image_file)
        image = Image.open(image_path).convert("RGB")
        
        eval_results = []
        for pair_idx, (question, gpt_response) in enumerate(qa_pairs):
            print(f"\n[Evaluate] Processing QA pair {pair_idx}...")
            
            correct_answer = None
            for tag_pair in [("<final>", "</final>"), ("<CONCLUSION>", "</CONCLUSION>")]:
                start_tag, end_tag = tag_pair
                if start_tag in gpt_response and end_tag in gpt_response:
                    start = gpt_response.find(start_tag) + len(start_tag)
                    end = gpt_response.find(end_tag)
                    correct_answer = gpt_response[start:end].strip()
                    break
            
            eval_result = test_context_vector_effect(
                model=self.model,
                processor=self.processor,
                tokenizer=self.tokenizer,
                question=question,
                image=image,
                context_vector=context_vector,
                correct_answer=correct_answer,
                device=self.device,
                num_trials=1,
                context_scales=[0.0, 1.0, 2.0]
            )
            
            eval_results.append({
                "pair_idx": pair_idx,
                "question": question,
                "correct_answer": correct_answer,
                "eval_result": eval_result
            })
        
        return {
            "image_path": image_path,
            "total_pairs": len(qa_pairs),
            "evaluated_pairs": len(eval_results),
            "results": eval_results
        }