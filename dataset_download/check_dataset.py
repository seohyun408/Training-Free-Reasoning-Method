import os
import json
from datasets import load_dataset
from pathlib import Path
from typing import List, Tuple, Dict
from utils import extract_qa_pairs

# Environment // API
HF_DATASETS_CACHE = os.getenv('HF_DATASETS_CACHE')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
WANDB_TOKEN = os.getenv('WANDB_TOKEN')


def print_example_structure(example: Dict, idx: int):
    """Print the structure of a single example"""
    print(f"\n{'='*80}")
    print(f"Example {idx + 1}")
    print(f"{'='*80}")
    
    # Print all top-level keys
    print(f"\nTop-level keys: {list(example.keys())}")
    
    # Print conversations structure
    conversations = example.get("conversations", [])
    print(f"\nNumber of conversation turns: {len(conversations)}")
    
    # Extract QA pairs
    qa_pairs = extract_qa_pairs(conversations)
    print(f"Number of QA pairs: {len(qa_pairs)}")
    
    # Print each conversation turn
    print(f"\n--- Conversation Turns ---")
    for i, turn in enumerate(conversations):
        role = turn.get("from", "unknown")
        value = turn.get("value", "")
        print(f"\nTurn {i + 1} ({role}):")
        print(f"{'-'*40}")
        # Print first 500 characters
        if len(value) > 500:
            print(f"{value[:500]}...")
            print(f"\n[Truncated: {len(value)} total characters]")
        else:
            print(value)
    
    # Print QA pairs
    if qa_pairs:
        print(f"\n--- Extracted QA Pairs ---")
        for pair_idx, (question, answer) in enumerate(qa_pairs):
            print(f"\nQA Pair {pair_idx + 1}:")
            print(f"Question ({len(question)} chars):")
            print(f"{' '*4}{question[:300]}{'...' if len(question) > 300 else ''}")
            print(f"\nAnswer ({len(answer)} chars):")
            print(f"{' '*4}{answer[:300]}{'...' if len(answer) > 300 else ''}")


def analyze_dataset_stats(sample_data: List[Dict]):
    """Analyze and print dataset statistics"""
    print(f"\n{'='*80}")
    print("Dataset Statistics")
    print(f"{'='*80}")
    
    total_examples = len(sample_data)
    total_conversations = sum(len(ex.get("conversations", [])) for ex in sample_data)
    total_qa_pairs = 0
    conversation_lengths = []
    qa_pair_counts = []
    
    for example in sample_data:
        conversations = example.get("conversations", [])
        conversation_lengths.append(len(conversations))
        qa_pairs = extract_qa_pairs(conversations)
        qa_pair_counts.append(len(qa_pairs))
        total_qa_pairs += len(qa_pairs)
    
    print(f"\nTotal examples analyzed: {total_examples}")
    print(f"Total conversation turns: {total_conversations}")
    print(f"Total QA pairs: {total_qa_pairs}")
    print(f"\nAverage conversation turns per example: {total_conversations / total_examples:.2f}")
    print(f"Average QA pairs per example: {total_qa_pairs / total_examples:.2f}")
    print(f"\nConversation turns - Min: {min(conversation_lengths)}, Max: {max(conversation_lengths)}")
    print(f"QA pairs - Min: {min(qa_pair_counts)}, Max: {max(qa_pair_counts)}")


def main():
    print("Loading LLaVA-CoT-100k dataset...")
    streamed_dataset = load_dataset("Xkev/LLaVA-CoT-100k", split="train", streaming=True)
    
    # Sampling examples
    sample_size = 5  # Check first 5 examples
    sample_data = []
    for i, example in enumerate(streamed_dataset):
        if len(sample_data) < sample_size:
            sample_data.append(example)
        else:
            break
    
    print(f"Loaded {len(sample_data)} examples for inspection")
    
    # Analyze statistics
    analyze_dataset_stats(sample_data)
    
    # Print detailed structure for each example
    print(f"\n\n{'='*80}")
    print("Detailed Example Inspection")
    print(f"{'='*80}")
    
    for idx, example in enumerate(sample_data):
        print_example_structure(example, idx)
    
    # Save sample to file for reference
    output_dir = Path("dataset_check_output")
    output_dir.mkdir(exist_ok=True)
    
    sample_file = output_dir / "sample_data.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Sample data saved to: {sample_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

