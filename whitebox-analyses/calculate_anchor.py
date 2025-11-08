"""
Anchor Vector 계산 함수들

KL divergence matrix로부터 각 문장의 중요도(anchor vector)를 추출합니다.
"""

import numpy as np
from typing import Literal


def compute_anchor_vector(
    kl_matrix: np.ndarray,
    method: Literal["outgoing", "incoming", "combined"] = "outgoing"
) -> np.ndarray:
    """
    KL divergence matrix로부터 anchor vector 추출
    
    Args:
        kl_matrix: [N, N] KL divergence matrix
                  kl_matrix[i, j] = 문장 j를 억제했을 때 문장 i에 미치는 영향
        method: anchor 벡터 추출 방법
            - "outgoing": 각 문장이 후속 문장에 미치는 평균 영향
            - "incoming": 각 문장이 이전 문장들로부터 받는 평균 영향
            - "combined": outgoing + incoming
    
    Returns:
        anchor_vector: [N] 각 문장의 중요도 점수
                      값이 클수록 중요한 anchor 문장
    """
    num_sentences = kl_matrix.shape[0]
    anchor_vector = np.zeros(num_sentences)
    
    if method == "outgoing":
        # 각 문장이 후속 문장에 미치는 평균 영향
        # kl_matrix[i, :] = i번째 문장을 억제했을 때의 영향
        # 따라서 kl_matrix[:, i]를 봐야 i번째 문장이 미치는 영향을 알 수 있음
        for i in range(num_sentences):
            # i번째 문장이 이후 문장들에 미치는 영향
            future_effects = kl_matrix[i+1:, i]
            anchor_vector[i] = np.nanmean(future_effects) if len(future_effects) > 0 else 0.0
        
    elif method == "incoming":
        # 각 문장이 이전 문장들로부터 받는 평균 영향
        for j in range(num_sentences):
            # j번째 문장을 억제했을 때 이전 문장들이 받는 영향
            previous_effects = kl_matrix[j, :j]
            anchor_vector[j] = np.nanmean(previous_effects) if len(previous_effects) > 0 else 0.0
        
    elif method == "combined":
        # Outgoing + Incoming
        outgoing = np.zeros(num_sentences)
        incoming = np.zeros(num_sentences)
        
        for i in range(num_sentences):
            future_effects = kl_matrix[i+1:, i]
            outgoing[i] = np.nanmean(future_effects) if len(future_effects) > 0 else 0.0
        
        for j in range(num_sentences):
            previous_effects = kl_matrix[j, :j]
            incoming[j] = np.nanmean(previous_effects) if len(previous_effects) > 0 else 0.0
        
        anchor_vector = outgoing + incoming
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'outgoing', 'incoming', or 'combined'")
    
    return anchor_vector


def get_top_anchors(anchor_vector: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    상위 k개의 anchor 문장 추출
    
    Args:
        anchor_vector: [N] 각 문장의 중요도 점수
        k: 반환할 상위 문장 개수
    
    Returns:
        top_indices: [k] 상위 k개 문장의 인덱스 (내림차순)
        top_scores: [k] 상위 k개 문장의 점수 (내림차순)
    """
    k = min(k, len(anchor_vector))
    top_indices = np.argsort(anchor_vector)[-k:][::-1]
    top_scores = anchor_vector[top_indices]
    return top_indices, top_scores


def print_anchor_summary(
    chunks: list[str],
    anchor_vector: np.ndarray,
    kl_matrix: np.ndarray,
    top_k: int = 5
):
    """
    Anchor 분석 결과를 출력
    
    Args:
        chunks: 문장 리스트
        anchor_vector: 각 문장의 중요도 점수
        kl_matrix: KL divergence matrix
        top_k: 출력할 상위 문장 개수
    """
    print("\n" + "=" * 80)
    
    print(f"\n✅ Statistics:")
    print(f"  - Number of sentences: {len(chunks)}")
    print(f"  - Anchor vector shape: {anchor_vector.shape}")
    print(f"  - Min importance: {np.nanmin(anchor_vector):.4f}")
    print(f"  - Max importance: {np.nanmax(anchor_vector):.4f}")
    print(f"  - Mean importance: {np.nanmean(anchor_vector):.4f}")
    print(f"  - Std importance: {np.nanstd(anchor_vector):.4f}")
    
    top_indices, top_scores = get_top_anchors(anchor_vector, k=top_k)
    
    print(f"\n✅ Top {top_k} Most Important Anchor Sentences:")
    print("-" * 80)
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
        sentence = chunks[idx]
        display_sentence = sentence if len(sentence) <= 100 else sentence[:97] + "..."
        print(f"\n  Rank {rank} (Sentence {idx}):")
        print(f"    Score: {score:.4f}")
        print(f"    Text: \"{display_sentence}\"")
    
    print(f"\n✅ KL Divergence Matrix Statistics:")
    print(f"  - Matrix shape: {kl_matrix.shape}")
    print(f"  - Non-NaN values: {np.sum(~np.isnan(kl_matrix))}/{kl_matrix.size}")
    print(f"  - Mean KL (non-NaN): {np.nanmean(kl_matrix):.4f}")
    print(f"  - Max KL: {np.nanmax(kl_matrix):.4f}")
    print(f"  - Min KL: {np.nanmin(kl_matrix):.4f}")
    
    print("\n" + "=" * 80)

