import os
import json
import argparse
import torch
import math
import torch.multiprocessing as mp
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from collections import defaultdict

# 사용자의 process.py에서 함수 가져오기
from process import process_dataset_sample

def build_parser():
    p = argparse.ArgumentParser(description="Run reasoning + thought anchor extraction with Multi-GPU support.")
    # Dataset / Path configurations
    p.add_argument("--data-root", default="/home/win2dvp21/cmu/LSMA_proj/dataset/LLaVA-CoT-100k", 
                   help="Root directory containing train.jsonl and image subfolders")
    p.add_argument("--json-filename", default="train.jsonl", help="JSONL filename")
    
    # Model configurations
    p.add_argument("--model-cache", default=os.getenv("HF_MODEL_CACHE", "/mnt/hdd/huggingface-models"), help="HF model cache dir")
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct", help="Model name to load")
    
    # GPU / Execution configurations
    p.add_argument("--num-gpus", type=int, default=10, help="Number of GPUs to use")
    p.add_argument("--samples-per-folder", type=int, default=10, help="Number of samples to process per subfolder")
    
    # Generation configurations
    p.add_argument("--max-new-tokens", type=int, default=256, help="Generation max_new_tokens")
    p.add_argument("--do-sample", action="store_true", default=True, help="Enable sampling")
    p.add_argument("--skip-suppression", action="store_true", help="Skip suppression KL computation")
    p.add_argument("--suppression-strategy", choices=["attn", "embed"], default="attn", help="Counterfactual suppression strategy")
    p.add_argument("--anchor-method", choices=["outgoing", "incoming", "combined"], default="outgoing", help="Aggregate KL for anchor vector")
    p.add_argument("--sparse-top-p", type=float, default=0.0, help="If >0 apply nucleus sparsification before KL")
    p.add_argument("--output-dir", default="anchor_vectors_output", help="Directory to store JSON outputs")
    p.add_argument("--contrastive-samples", type=int, default=5, help="Number of samples for contrastive generation")
    p.add_argument("--test-pca-context", action="store_true", default=True, help="Test PCA context vector effect")
    p.add_argument("--pca-num-trials", type=int, default=5, help="Number of trials per context scale")
    
    return p

def inject_env_from_args(args):
    os.environ["MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["DO_SAMPLE"] = "1" if args.do_sample else "0"
    os.environ["SKIP_SUPPRESSION"] = "1" if args.skip_suppression else "0"
    os.environ["SUPPRESSION_STRATEGY"] = args.suppression_strategy
    os.environ["ANCHOR_METHOD"] = args.anchor_method
    if args.sparse_top_p > 0:
        os.environ["SPARSE_TOP_P"] = str(args.sparse_top_p)
    os.environ["ENABLE_CONTRASTIVE"] = "1"
    os.environ["CONTRASTIVE_SAMPLES"] = str(args.contrastive_samples)
    os.environ["TEST_PCA_CONTEXT"] = "1" if args.test_pca_context else "0"
    os.environ["PCA_NUM_TRIALS"] = str(args.pca_num_trials)

def load_local_dataset_filtered(data_root, json_filename, limit_per_folder=10):
    """
    로컬 train.jsonl을 읽어서 폴더별로 limit_per_folder 만큼만 데이터를 추출합니다.
    """
    json_path = os.path.join(data_root, json_filename)
    print(f"[Main] Loading and filtering dataset from {json_path}...")
    
    filtered_data = []
    folder_counts = defaultdict(int)
    
    # train.jsonl 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        # 전체 데이터를 순회하되, 이미 꽉 찬 폴더는 건너뛰는 최적화는 
        # 폴더 순서가 섞여있을 수 있으므로 전체 스캔을 권장
        all_lines = f.readlines()

    print(f"[Main] Total lines in jsonl: {len(all_lines)}")

    for idx, line in enumerate(all_lines):
        try:
            entry = json.loads(line)
            image_rel_path = entry.get('image', '')
            
            if not image_rel_path:
                continue
                
            # 이미지 경로에서 폴더명 추출 (예: ai2d/image_abc.png -> ai2d)
            folder_name = image_rel_path.split('/')[0]
            
            # 해당 폴더의 카운트가 제한보다 작을 때만 추가
            if folder_counts[folder_name] < limit_per_folder:
                # 원본 entry에 global index 보존
                entry['original_idx'] = idx
                entry['folder_name'] = folder_name
                # 이미지 절대 경로 수정 (필요시 process 함수 내부에서 쓰일 수 있음)
                # entry['image']는 process.py가 어떻게 처리하냐에 따라 다르지만 
                # 보통 상대경로를 유지하고 root를 따로 넘기는게 일반적임.
                filtered_data.append(entry)
                folder_counts[folder_name] += 1
        except Exception as e:
            print(f"Error parsing line {idx}: {e}")
            continue

    print(f"[Main] Filtered dataset size: {len(filtered_data)}")
    print(f"[Main] Counts per folder: {dict(folder_counts)}")
    return filtered_data

def worker_process(rank, args, subset_data, model_name, output_dir):
    """
    각 GPU에서 실행될 워커 함수
    """
    # 환경 변수 주입 (Process 별로 필요할 수 있음)
    inject_env_from_args(args)
    
    # GPU 설정
    device = f"cuda:{rank}"
    print(f"[Worker {rank}] Starting on {device}. Processing {len(subset_data)} items.")
    
    # 모델 로드
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=args.model_cache)
        
        load_kwargs = {
            "trust_remote_code": True,
            "cache_dir": args.model_cache,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16
        }
        
        # Device Map을 해당 GPU로 강제
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map={"": device},
            **load_kwargs
        )
        model.eval()
        tokenizer = getattr(processor, "tokenizer", None) or processor
        
    except Exception as e:
        print(f"[Worker {rank}] Failed to load model: {e}")
        return

    # 데이터 처리 루프
    local_processed = 0
    for item in tqdm(subset_data, desc=f"GPU {rank}", position=rank):
        original_idx = item['original_idx']
        folder_name = item.get('folder_name', 'unknown')
        
        # process_dataset_sample 호출
        # 주의: process_dataset_sample 내부에서 example['image']가 경로 문자열인지 PIL 이미지인지 확인 필요
        # 보통 jsonl 로드시 문자열이므로 process 함수가 로드하도록 images_root 전달
        try:
            result = process_dataset_sample(
                example=item,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                model_name=model_name,
                images_root=args.data_root, # 로컬 데이터 루트
                generate_reasoning=True,
            )
            
            result["example_idx"] = original_idx
            result["worker_rank"] = rank
            
            # 결과 저장
            out_path = output_dir / f"{folder_name}_sample_{original_idx}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            local_processed += 1
            
        except Exception as e:
            print(f"[Worker {rank}] Error processing index {original_idx}: {e}")
            # 에러 로그 저장
            with open(output_dir / f"error_{original_idx}.txt", "w") as f:
                f.write(str(e))

    print(f"[Worker {rank}] Finished. Processed {local_processed} items.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # 환경 설정
    os.environ["HF_HOME"] = args.model_cache
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. 데이터셋 로드 및 필터링 (Main Process에서 수행)
    full_dataset = load_local_dataset_filtered(
        data_root=args.data_root,
        json_filename=args.json_filename,
        limit_per_folder=args.samples_per_folder
    )
    
    if len(full_dataset) == 0:
        print("No data found matching criteria.")
        return

    # 2. 데이터 분배 (Chunking)
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"[Main] Distributing {len(full_dataset)} tasks across {num_gpus} GPUs.")
    
    chunk_size = math.ceil(len(full_dataset) / num_gpus)
    data_chunks = [full_dataset[i:i + chunk_size] for i in range(0, len(full_dataset), chunk_size)]
    
    # 3. 멀티프로세싱 실행 (Spawn context 사용 권장)
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for rank in range(num_gpus):
        if rank >= len(data_chunks): break # 데이터가 GPU 수보다 적을 경우 대비
        
        p = mp.Process(
            target=worker_process,
            args=(rank, args, data_chunks[rank], args.model_name, output_dir)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("[Main] All workers finished.")

if __name__ == "__main__":
    main()