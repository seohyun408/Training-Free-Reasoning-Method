import os
import numpy as np 
import torch 
import google.generativeai as genai

from typing import List, Dict, Tuple, Optional
from contrastive_generation import extract_final_answer
from method.context_vector import generate_reasoning_with_context_vector
from bert_score import score
from PIL import Image

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


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
        scale_llm_correct = 0
        scale_precisions = []
        scale_recalls = []
        scale_f1s = []
        scale_llm_confidences = []
        
        for trial_idx in range(num_trials):
            print(f"\n  Trial {trial_idx + 1}/{num_trials}:")
            
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
            
            precision, recall, f1 = compute_bertscore(generated_answer, correct_answer)
            print(f"  🌟 BERTScore - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
            
            llm_is_correct, llm_confidence, llm_reason = llm_judge(
                question=question,
                generated_answer=generated_answer,
                correct_answer=correct_answer,
                image=image
            )
            
            if llm_is_correct:
                scale_llm_correct += 1
            
            llm_status = "✅" if llm_is_correct else "❌"
            print(f"  🤖 LLM Judge: {llm_status} (conf: {llm_confidence:.2f}) - {llm_reason[:50]}...")
            
            scale_precisions.append(precision)
            scale_recalls.append(recall)
            scale_f1s.append(f1)
            scale_llm_confidences.append(llm_confidence)
            
        accuracy = scale_correct / num_trials if num_trials > 0 else 0.0
        llm_accuracy = scale_llm_correct / num_trials if num_trials > 0 else 0.0
        avg_precision = np.mean(scale_precisions) 
        avg_recall = np.mean(scale_recalls) 
        avg_f1 = np.mean(scale_f1s) 
        avg_llm_confidence = np.mean(scale_llm_confidences)
        
        print(f"\n  📈 Scale {scale} Results:")
        print(f"     Exact Match: {scale_correct}/{num_trials} = {accuracy:.2%}")
        print(f"     LLM Judge:   {scale_llm_correct}/{num_trials} = {llm_accuracy:.2%} (avg conf: {avg_llm_confidence:.2f})")
        print(f"     BERTScore - P: {avg_precision:.4f}, R: {avg_recall:.4f}, F1: {avg_f1:.4f}")
        
        all_results[float(scale)] = {
            "generated_answer": generated_answer,
            "accuracy": accuracy,
            "llm_accuracy": llm_accuracy,
            "correct_count": scale_correct,
            "llm_correct_count": scale_llm_correct,
            "total_trials": num_trials,
            "bertscore": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            },
            "llm_judge": {
                "avg_confidence": avg_llm_confidence
            },
            "trials": scale_results
        }
    
    print(f"\n{'='*80}")
    print(f"📊 All Scales Completed")
    print(f"{'='*80}\n")
    
    return all_results


def compute_bertscore(sentence1: str, sentence2: str):
    if sentence1 is None or sentence2 is None:
        return 0.0, 0.0, 0.0
    if not sentence1.strip() or not sentence2.strip():
        return 0.0, 0.0, 0.0

    cands = [sentence1]
    refs = [sentence2]

    P, R, F1 = score(cands, refs, lang="en", model_type="microsoft/deberta-large")

    return float(P[0]), float(R[0]), float(F1[0])


def llm_judge(
    question: str,
    generated_answer: str,
    correct_answer: str,
    image,
    model_name: str = "gemini-2.0-flash"
) -> Tuple[bool, float, str]:
    if generated_answer is None or correct_answer is None:
        return False, 0.0, "invalid input"
    
    judge_prompt = f"""You are an expert judge evaluating answer correctness.

Question: {question}

Generated Answer: {generated_answer}
Correct Answer: {correct_answer}

Evaluate if the generated answer is semantically equivalent to the correct answer.
Consider:
1. Same meaning even if different wording
2. Partial credit for partially correct answers
3. Ignore minor formatting differences

Respond in this exact format:
<verdict>CORRECT or INCORRECT</verdict>
<confidence>0.0-1.0</confidence>
<reason>brief explanation</reason>"""

    try:
        gemini_model = genai.GenerativeModel(model_name)
        
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.open(image).convert("RGB")
        
        response = gemini_model.generate_content(
            [pil_image, judge_prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=200
            )
        )
        
        response_text = response.text
        
    except Exception as e:
        print(f"  [Gemini Error] {str(e)[:50]}...")
        return False, 0.0, f"API error: {str(e)[:30]}"
    
    is_correct = False
    confidence = 0.0
    reason = ""
    
    if "<verdict>" in response_text and "</verdict>" in response_text:
        verdict_start = response_text.find("<verdict>") + len("<verdict>")
        verdict_end = response_text.find("</verdict>")
        verdict = response_text[verdict_start:verdict_end].strip().upper()
        is_correct = "CORRECT" in verdict and "INCORRECT" not in verdict
    
    if "<confidence>" in response_text and "</confidence>" in response_text:
        conf_start = response_text.find("<confidence>") + len("<confidence>")
        conf_end = response_text.find("</confidence>")
        try:
            confidence = float(response_text[conf_start:conf_end].strip())
        except:
            confidence = 1.0 if is_correct else 0.0
    
    if "<reason>" in response_text and "</reason>" in response_text:
        reason_start = response_text.find("<reason>") + len("<reason>")
        reason_end = response_text.find("</reason>")
        reason = response_text[reason_start:reason_end].strip()
    
    return is_correct, confidence, reason