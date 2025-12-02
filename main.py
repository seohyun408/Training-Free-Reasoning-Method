import os
import json
import argparse
import numpy as np
import shutil
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import glob

# process 모듈 임포트
from process import DatasetProcessor
import method.context_vector as cv 

# 환경 변수 로드
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
HF_HOME = os.getenv('HF_HOME')
DATASETS_CACHE = os.getenv('HF_DATASETS_CACHE')
MODEL_CACHE = os.getenv('HF_MODEL_CACHE')

USE_WANDB = False 

def build_parser():
    p = argparse.ArgumentParser(description="🔥Team Galaxy Multi-GPU Split Pipeline🔥")
    # Dataset & Model
    p.add_argument("--dataset_name", default="Xkev/LLaVA-CoT-100k")
    p.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--load_8bit", action="store_true")
    
    # Generation Config
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--do-sample", action="store_true", default=True)
    p.add_argument("--max_attempts", type=int, default=2)
    p.add_argument("--contrastive_samples", type=int, default=5)
    
    # Paths
    p.add_argument("--output_dir", default="outputs", help="Directory for Phase 1 results (vector generation)")
    p.add_argument("--output_dir2", default="outputs2", help="Directory for Phase 3 results (evaluation)")
    p.add_argument("--images_root", required=True)
    
    # Compute
    p.add_argument("--num_gpus", type=int, default=1)
    
    # [수정됨] 데이터셋 개수 조절 인자 추가
    p.add_argument("--num_vector_samples", type=int, default=30, 
                   help="Number of samples to use for Context Vector generation. Set -1 for ALL.")
    p.add_argument("--num_eval_samples", type=int, default=-1, 
                   help="Number of samples to evaluate on. Set -1 for ALL.")
    
    return p

# ==========================================
# Phase 1 Worker: 벡터 생성을 위한 데이터 처리
# ==========================================
def worker_processing(rank, args, subset_indices):
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{rank}")
    
    # 모델 로드
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, cache_dir=MODEL_CACHE)
    load_kwargs = {
        "trust_remote_code": True, "cache_dir": MODEL_CACHE, "low_cpu_mem_usage": True,
        "device_map": f"cuda:{rank}",
    }
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    elif args.load_8bit:
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["dtype"] = torch.float16

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name, **load_kwargs)
    model.eval()
    tokenizer = getattr(processor, "tokenizer", None)

    # [워커 내부] 데이터셋 로드 (로컬 우선)
    local_json_path = os.path.join(args.images_root, "train.jsonl")
    if os.path.exists(local_json_path):
        dataset = load_dataset("json", data_files=local_json_path, split="train", cache_dir=DATASETS_CACHE)
    else:
        dataset = load_dataset(args.dataset_name, split="train", cache_dir=DATASETS_CACHE)
        
    subset = dataset.select(subset_indices)

    # generate_reasoning=True (벡터 추출용)
    proc = DatasetProcessor(args, model, processor, tokenizer, device, args.images_root, True, args.model_name)
    output_dir = Path(args.output_dir)

    for i, example in enumerate(tqdm(subset, desc=f"VecGen-GPU{rank}", position=rank)):
        try:
            global_idx = subset_indices[i]
            result = proc.process_sample(example)
            result["example_idx"] = global_idx
            
            with open(output_dir / f"example_{global_idx}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[VecGen-GPU{rank}] Error idx {global_idx}: {e}")

# ==========================================
# Phase 3 Worker: 평가 (Evaluation)
# ==========================================
def worker_evaluation(rank, args, subset_indices, context_vector):
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{rank}")
    
    # 모델 로드 (메모리 정리를 위해 새로 로드)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, cache_dir=MODEL_CACHE)
    load_kwargs = {
        "trust_remote_code": True, "cache_dir": MODEL_CACHE, "low_cpu_mem_usage": True,
        "device_map": f"cuda:{rank}",
    }
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    elif args.load_8bit:
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["dtype"] = torch.float16

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name, **load_kwargs)
    model.eval()
    tokenizer = getattr(processor, "tokenizer", None)

    # [워커 내부] 데이터셋 로드 (로컬 우선)
    local_json_path = os.path.join(args.images_root, "train.jsonl")
    if os.path.exists(local_json_path):
        dataset = load_dataset("json", data_files=local_json_path, split="train", cache_dir=DATASETS_CACHE)
    else:
        dataset = load_dataset(args.dataset_name, split="train", cache_dir=DATASETS_CACHE)
        
    subset = dataset.select(subset_indices)

    # generate_reasoning=False (평가용이므로 불필요)
    proc = DatasetProcessor(args, model, processor, tokenizer, device, args.images_root, False, args.model_name)
    output_dir2 = Path(args.output_dir2)

    for i, example in enumerate(tqdm(subset, desc=f"Eval-GPU{rank}", position=rank)):
        try:
            global_idx = subset_indices[i]
            # 이미 계산된 context_vector를 사용하여 평가
            pca_results = proc.evaluate_with_context_vector(context_vector, example)
            
            with open(output_dir2 / f"example_{global_idx}.json", "w", encoding="utf-8") as f:
                json.dump(pca_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Eval-GPU{rank}] Error idx {global_idx}: {e}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    Path(args.output_dir2).mkdir(exist_ok=True, parents=True)
    
    # 임시 폴더 초기화
    pca_temp_dir = os.path.join(os.path.dirname(__file__), "pca_data_temp")
    if os.path.exists(pca_temp_dir): shutil.rmtree(pca_temp_dir)
    os.makedirs(pca_temp_dir, exist_ok=True)

    # 1. 전체 데이터셋 인덱스 로드 (메인 프로세스)
    print(">>>>>> Loading Dataset Metadata...")
    
    local_json_path = os.path.join(args.images_root, "train.jsonl")
    
    if os.path.exists(local_json_path):
        print(f"✅ Found local dataset file: {local_json_path}")
        dataset = load_dataset("json", data_files=local_json_path, split="train", cache_dir=DATASETS_CACHE)
    else:
        print(f"⚠️ Local file not found at {local_json_path}. Trying to download from HuggingFace...")
        dataset = load_dataset(args.dataset_name, split="train", cache_dir=DATASETS_CACHE)

    all_indices = [i for i, x in enumerate(dataset) if 'coco' in x.get('image', '').lower()]
    print(f"Total COCO examples found: {len(all_indices)}")
    
    # =========================================================
    # STEP 1: Context Vector 생성을 위한 샘플링 (Phase 1)
    # =========================================================
    
    # 벡터 생성용 인덱스 선택
    if args.num_vector_samples == -1:
        vector_indices = all_indices
        print(f"Using ALL {len(vector_indices)} samples for Vector Generation.")
    else:
        # 지정된 개수만큼만 사용 (예: 30개)
        count = min(args.num_vector_samples, len(all_indices))
        vector_indices = all_indices[:count]
        print(f"Using first {count} samples for Vector Generation.")

    print("\n" + "="*50)
    print(f"🚀 PHASE 1: Vector Generation Processing ({len(vector_indices)} samples)")
    print("="*50)

    # 멀티프로세싱으로 벡터 생성용 데이터 처리
    mp.set_start_method('spawn', force=True)
    
    num_gpus = args.num_gpus
    chunk_size = int(np.ceil(len(vector_indices) / num_gpus))
    # 빈 리스트가 생기지 않도록 처리
    if chunk_size == 0: chunk_size = 1 
    
    subset_indices_list = [vector_indices[i:i + chunk_size] for i in range(0, len(vector_indices), chunk_size)]
    
    processes = []
    for rank in range(min(num_gpus, len(subset_indices_list))):
        p = mp.Process(target=worker_processing, args=(rank, args, subset_indices_list[rank]))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print("\n✅ PHASE 1 Completed.")

    # =========================================================
    # STEP 2: Context Vector 계산 (Phase 2)
    # =========================================================
    print("\n" + "="*50)
    print("🚀 PHASE 2: Calculating Context Vector")
    print("="*50)

    pca_files = glob.glob(os.path.join(pca_temp_dir, "*.npy"))
    all_diffs = []
    
    if not pca_files:
        print("❌ Error: No PCA vectors found. Cannot proceed.")
        return

    for p_file in pca_files:
        try:
            vec = np.load(p_file, allow_pickle=True)
            if vec.ndim == 1: all_diffs.append(vec)
            else: 
                for v in vec: all_diffs.append(v)
        except: pass

    if len(all_diffs) == 0:
        print("❌ Error: Valid PCA vectors are empty.")
        return

    combined_pca_data = np.array(all_diffs, dtype=object)
    
    # 벡터 저장
    final_pca_dir = os.path.join(os.path.dirname(__file__), "pca_data")
    os.makedirs(final_pca_dir, exist_ok=True)
    np.save(os.path.join(final_pca_dir, "vector_generation.npy"), combined_pca_data)
    
    # PCA 계산
    context_vector = None
    try:
        context_vector = cv.compute_pca_context_vector(combined_pca_data, n_components=1)
        print(f"✅ Context Vector Calculated. Shape: {context_vector.shape}")
        print(f"   (Calculated from {len(all_diffs)} reasoning diffs)")
    except Exception as e:
        print(f"❌ Error computing context vector: {e}")
        return

    # 임시 폴더 청소
    shutil.rmtree(pca_temp_dir)

    # =========================================================
    # STEP 3: 전체(혹은 지정된) 데이터셋 평가 (Phase 3)
    # =========================================================
    
    # 평가용 인덱스 선택
    if args.num_eval_samples == -1:
        eval_indices = all_indices
        print(f"\nEvaluating on ALL {len(eval_indices)} samples.")
    else:
        count = min(args.num_eval_samples, len(all_indices))
        eval_indices = all_indices[:count]
        print(f"\nEvaluating on first {count} samples.")

    print("\n" + "="*50)
    print(f"🚀 PHASE 3: Evaluation Started ({len(eval_indices)} samples)")
    print("="*50)

    # 평가 데이터 분할
    chunk_size_eval = int(np.ceil(len(eval_indices) / num_gpus))
    if chunk_size_eval == 0: chunk_size_eval = 1
    
    subset_eval_list = [eval_indices[i:i + chunk_size_eval] for i in range(0, len(eval_indices), chunk_size_eval)]

    # GPU 메모리 리셋을 위해 새 프로세스 시작
    processes = []
    for rank in range(min(num_gpus, len(subset_eval_list))):
        p = mp.Process(target=worker_evaluation, args=(rank, args, subset_eval_list[rank], context_vector))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    print("\n🎉 All Finished Successfully!")

if __name__ == "__main__":
    main()
