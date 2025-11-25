# #!/usr/bin/env python3
# """
# Comprehensive evaluation script for Thought Anchor Detection + PCA Context Vector experiments.

# Metrics:
# 1. Correlation: Spearman's ρ between anchor scores and output changes
# 2. Accuracy: EM (Exact Match) and F1 scores
# 3. Hallucination: Object/attribute hallucination detection
# 4. Efficiency: Latency, VRAM, tokens/sec
# 5. BERTScore: Alignment between reasoning and context
# """

# import json
# import glob
# import numpy as np
# from pathlib import Path
# from typing import Dict, List, Tuple
# from collections import defaultdict
# import re

# # Optional imports (will check if available)
# try:
#     from scipy.stats import spearmanr
#     HAS_SCIPY = True
# except ImportError:
#     HAS_SCIPY = False
#     print("⚠️  scipy not installed. Correlation metrics will be skipped.")

# try:
#     from bert_score import score as bert_score_fn
#     HAS_BERTSCORE = True
# except ImportError:
#     HAS_BERTSCORE = False
#     print("⚠️  bert-score not installed. BERTScore metrics will be skipped.")
#     print("   Install with: pip install bert-score")


# class MetricsEvaluator:
#     """Comprehensive metrics evaluator for VQA reasoning experiments."""

#     def __init__(self, results_dir: str = "anchor_vectors_output"):
#         self.results_dir = Path(results_dir)
#         self.results = []
#         self.metrics = {}

#     def load_results(self):
#         """Load all JSON results from output directory."""
#         result_files = sorted(self.results_dir.glob("example_*.json"))

#         if not result_files:
#             raise FileNotFoundError(f"No result files found in {self.results_dir}")

#         for file_path in result_files:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
#                 self.results.append(data)

#         print(f"✅ Loaded {len(self.results)} result files")
#         return self

#     # ========================================================================
#     # 1. ACCURACY METRICS
#     # ========================================================================

#     def normalize_answer(self, answer: str) -> str:
#         """Normalize answer for comparison."""
#         if not answer:
#             return ""
#         # Remove punctuation, convert to lowercase
#         answer = str(answer).strip().lower()
#         answer = re.sub(r'[^\w\s]', '', answer)
#         return answer

#     def compute_f1(self, prediction: str, ground_truth: str) -> float:
#         """Compute token-level F1 score."""
#         pred_tokens = self.normalize_answer(prediction).split()
#         gold_tokens = self.normalize_answer(ground_truth).split()

#         if len(pred_tokens) == 0 or len(gold_tokens) == 0:
#             return int(pred_tokens == gold_tokens)

#         common = set(pred_tokens) & set(gold_tokens)

#         if len(common) == 0:
#             return 0.0

#         precision = len(common) / len(pred_tokens)
#         recall = len(common) / len(gold_tokens)
#         f1 = 2 * (precision * recall) / (precision + recall)

#         return f1

#     def compute_accuracy_metrics(self) -> Dict:
#         """
#         Compute EM (Exact Match) and F1 scores.

#         Returns breakdown by:
#         - Baseline (scale 0.0)
#         - Best PCA scale
#         - Per-scale results
#         """
#         accuracy_data = {
#             'baseline': {'em': [], 'f1': []},
#             'best_scale': {'em': [], 'f1': []},
#             'per_scale': defaultdict(lambda: {'em': [], 'f1': []})
#         }

#         for result in self.results:
#             qa_pairs = result.get('qa_pairs', [])
#             if not qa_pairs:
#                 continue

#             qa_pair = qa_pairs[0]
#             contrastive = qa_pair.get('contrastive')

#             if not contrastive:
#                 continue

#             correct_answer = contrastive.get('correct_answer', '')
#             pca_context = contrastive.get('pca_context')

#             if not pca_context:
#                 continue

#             pca_results = pca_context.get('results', {})

#             # Find best scale
#             best_scale = None
#             best_accuracy = -1

#             for scale, scale_result in pca_results.items():
#                 scale_accuracy = scale_result.get('accuracy', 0.0)
#                 if scale_accuracy > best_accuracy:
#                     best_accuracy = scale_accuracy
#                     best_scale = scale

#                 # Compute EM and F1 for each trial
#                 generated_answers = scale_result.get('generated_answers', [])

#                 for gen_answer in generated_answers:
#                     # Exact Match
#                     em = int(self.normalize_answer(gen_answer) == self.normalize_answer(correct_answer))

#                     # F1 Score
#                     f1 = self.compute_f1(gen_answer, correct_answer)

#                     accuracy_data['per_scale'][scale]['em'].append(em)
#                     accuracy_data['per_scale'][scale]['f1'].append(f1)

#                     # Baseline (scale 0.0)
#                     if scale == '0.0':
#                         accuracy_data['baseline']['em'].append(em)
#                         accuracy_data['baseline']['f1'].append(f1)

#             # Best scale results
#             if best_scale and best_scale in pca_results:
#                 generated_answers = pca_results[best_scale].get('generated_answers', [])
#                 for gen_answer in generated_answers:
#                     em = int(self.normalize_answer(gen_answer) == self.normalize_answer(correct_answer))
#                     f1 = self.compute_f1(gen_answer, correct_answer)
#                     accuracy_data['best_scale']['em'].append(em)
#                     accuracy_data['best_scale']['f1'].append(f1)

#         # Compute averages
#         summary = {
#             'baseline': {
#                 'em': np.mean(accuracy_data['baseline']['em']) if accuracy_data['baseline']['em'] else 0.0,
#                 'f1': np.mean(accuracy_data['baseline']['f1']) if accuracy_data['baseline']['f1'] else 0.0
#             },
#             'best_scale': {
#                 'em': np.mean(accuracy_data['best_scale']['em']) if accuracy_data['best_scale']['em'] else 0.0,
#                 'f1': np.mean(accuracy_data['best_scale']['f1']) if accuracy_data['best_scale']['f1'] else 0.0
#             },
#             'per_scale': {}
#         }

#         for scale, data in accuracy_data['per_scale'].items():
#             summary['per_scale'][scale] = {
#                 'em': np.mean(data['em']) if data['em'] else 0.0,
#                 'f1': np.mean(data['f1']) if data['f1'] else 0.0,
#                 'count': len(data['em'])
#             }

#         self.metrics['accuracy'] = summary
#         return summary

#     # ========================================================================
#     # 2. CORRELATION METRICS
#     # ========================================================================

#     def compute_correlation_metrics(self) -> Dict:
#         """
#         Compute Spearman's ρ between:
#         - Anchor scores (KL divergence-based attention suppression)
#         - Output changes (probability difference between positive/negative samples)
#         """
#         if not HAS_SCIPY:
#             print("⚠️  Skipping correlation metrics (scipy not available)")
#             return {}

#         correlation_data = {
#             'anchor_vs_prob_delta': [],
#             'anchor_vs_pca_improvement': []
#         }

#         for result in self.results:
#             qa_pairs = result.get('qa_pairs', [])
#             if not qa_pairs:
#                 continue

#             qa_pair = qa_pairs[0]

#             # Anchor scores
#             anchor_vector = qa_pair.get('anchor_vector', [])
#             if not anchor_vector:
#                 continue

#             # Remove last element (usually 0.0 for <final> tag)
#             anchor_scores = anchor_vector[:-1] if anchor_vector[-1] == 0.0 else anchor_vector

#             contrastive = qa_pair.get('contrastive')
#             if not contrastive:
#                 continue

#             # Output change: positive_prob - negative_prob
#             pos_prob = contrastive.get('positive_probability', 0.0)
#             neg_prob = contrastive.get('negative_probability', 0.0)
#             prob_delta = pos_prob - neg_prob

#             # Correlation: anchor scores vs probability delta
#             # (We'll use the max anchor score as representative)
#             if anchor_scores:
#                 max_anchor_score = max(anchor_scores)
#                 correlation_data['anchor_vs_prob_delta'].append((max_anchor_score, prob_delta))

#             # PCA improvement: (best_scale_accuracy - baseline_accuracy)
#             pca_context = contrastive.get('pca_context')
#             if pca_context:
#                 pca_results = pca_context.get('results', {})
#                 baseline_acc = pca_results.get('0.0', {}).get('accuracy', 0.0)

#                 # Find best accuracy
#                 best_acc = max((r.get('accuracy', 0.0) for r in pca_results.values()), default=0.0)
#                 pca_improvement = best_acc - baseline_acc

#                 if anchor_scores:
#                     correlation_data['anchor_vs_pca_improvement'].append((max_anchor_score, pca_improvement))

#         # Compute Spearman correlations
#         summary = {}

#         if correlation_data['anchor_vs_prob_delta']:
#             x, y = zip(*correlation_data['anchor_vs_prob_delta'])
#             rho, pval = spearmanr(x, y)
#             summary['anchor_vs_prob_delta'] = {
#                 'spearman_rho': rho,
#                 'p_value': pval,
#                 'n_samples': len(x)
#             }

#         if correlation_data['anchor_vs_pca_improvement']:
#             x, y = zip(*correlation_data['anchor_vs_pca_improvement'])
#             rho, pval = spearmanr(x, y)
#             summary['anchor_vs_pca_improvement'] = {
#                 'spearman_rho': rho,
#                 'p_value': pval,
#                 'n_samples': len(x)
#             }

#         self.metrics['correlation'] = summary
#         return summary

#     # ========================================================================
#     # 3. HALLUCINATION METRICS
#     # ========================================================================

#     def detect_hallucination(self, reasoning: str, options: List[str]) -> Dict:
#         """
#         Detect hallucination: does the reasoning mention objects/options not in the question?

#         Simple heuristic:
#         - Extract mentioned options from reasoning
#         - Check if any are NOT in the provided options list
#         """
#         reasoning_lower = reasoning.lower()

#         # Extract letter mentions (A, B, C, D, etc.)
#         mentioned_letters = set(re.findall(r'\b([A-Z])\b', reasoning))

#         # Provided options (normalized)
#         valid_options = set(opt.strip().upper() for opt in options)

#         # Hallucinated options
#         hallucinated = mentioned_letters - valid_options

#         return {
#             'has_hallucination': len(hallucinated) > 0,
#             'hallucinated_options': list(hallucinated),
#             'hallucination_count': len(hallucinated)
#         }

#     def compute_hallucination_metrics(self) -> Dict:
#         """
#         Compute hallucination metrics:
#         - False positive rate (mentions non-existent options)
#         - CHAIR-style object hallucination
#         """
#         hallucination_data = {
#             'baseline': [],
#             'best_scale': [],
#             'per_scale': defaultdict(list)
#         }

#         for result in self.results:
#             qa_pairs = result.get('qa_pairs', [])
#             if not qa_pairs:
#                 continue

#             qa_pair = qa_pairs[0]
#             question = qa_pair.get('question', '')

#             # Extract options from question
#             options_match = re.findall(r'Options?:\s*\n((?:[-•]\s*[^\n]+\n?)+)', question, re.IGNORECASE)

#             if not options_match:
#                 continue

#             options_text = options_match[0]
#             options = re.findall(r'[-•]\s*([A-Z])\b', options_text)

#             contrastive = qa_pair.get('contrastive')
#             if not contrastive:
#                 continue

#             pca_context = contrastive.get('pca_context')
#             if not pca_context:
#                 continue

#             pca_results = pca_context.get('results', {})

#             # Find best scale
#             best_scale = max(pca_results.keys(),
#                            key=lambda s: pca_results[s].get('accuracy', 0.0),
#                            default=None)

#             for scale, scale_result in pca_results.items():
#                 generated_answers = scale_result.get('generated_answers', [])

#                 for gen_answer in generated_answers:
#                     halluc = self.detect_hallucination(str(gen_answer), options)

#                     hallucination_data['per_scale'][scale].append(halluc['has_hallucination'])

#                     if scale == '0.0':
#                         hallucination_data['baseline'].append(halluc['has_hallucination'])

#                     if scale == best_scale:
#                         hallucination_data['best_scale'].append(halluc['has_hallucination'])

#         # Compute rates
#         summary = {
#             'baseline': {
#                 'hallucination_rate': np.mean(hallucination_data['baseline']) if hallucination_data['baseline'] else 0.0,
#                 'count': len(hallucination_data['baseline'])
#             },
#             'best_scale': {
#                 'hallucination_rate': np.mean(hallucination_data['best_scale']) if hallucination_data['best_scale'] else 0.0,
#                 'count': len(hallucination_data['best_scale'])
#             },
#             'per_scale': {}
#         }

#         for scale, data in hallucination_data['per_scale'].items():
#             summary['per_scale'][scale] = {
#                 'hallucination_rate': np.mean(data) if data else 0.0,
#                 'count': len(data)
#             }

#         self.metrics['hallucination'] = summary
#         return summary

#     # ========================================================================
#     # 4. EFFICIENCY METRICS
#     # ========================================================================

#     def compute_efficiency_metrics(self) -> Dict:
#         """
#         Compute efficiency metrics from PCA context vector results.

#         Note: Actual latency/VRAM measurements require runtime profiling.
#         This estimates based on generation counts.
#         """
#         efficiency_data = {
#             'avg_trials_per_scale': {},
#             'total_generations': 0
#         }

#         for result in self.results:
#             qa_pairs = result.get('qa_pairs', [])
#             if not qa_pairs:
#                 continue

#             qa_pair = qa_pairs[0]
#             contrastive = qa_pair.get('contrastive')

#             if not contrastive:
#                 continue

#             pca_context = contrastive.get('pca_context')
#             if not pca_context:
#                 continue

#             pca_results = pca_context.get('results', {})

#             for scale, scale_result in pca_results.items():
#                 total_trials = scale_result.get('total_trials', 0)

#                 if scale not in efficiency_data['avg_trials_per_scale']:
#                     efficiency_data['avg_trials_per_scale'][scale] = []

#                 efficiency_data['avg_trials_per_scale'][scale].append(total_trials)
#                 efficiency_data['total_generations'] += total_trials

#         summary = {
#             'total_generations': efficiency_data['total_generations'],
#             'avg_trials_per_scale': {
#                 scale: np.mean(trials)
#                 for scale, trials in efficiency_data['avg_trials_per_scale'].items()
#             }
#         }

#         self.metrics['efficiency'] = summary
#         return summary

#     # ========================================================================
#     # 5. BERTSCORE METRICS
#     # ========================================================================

#     def compute_bertscore_metrics(self) -> Dict:
#         """
#         Compute BERTScore between reasoning sentences and their context.

#         Measures semantic alignment between:
#         - Generated reasoning
#         - Question + Options (context)
#         """
#         if not HAS_BERTSCORE:
#             print("⚠️  Skipping BERTScore metrics (bert-score not available)")
#             return {}

#         bertscore_data = {
#             'reasoning_vs_question': {'precision': [], 'recall': [], 'f1': []},
#             'positive_vs_negative': {'precision': [], 'recall': [], 'f1': []}
#         }

#         for result in self.results:
#             qa_pairs = result.get('qa_pairs', [])
#             if not qa_pairs:
#                 continue

#             qa_pair = qa_pairs[0]
#             question = qa_pair.get('question', '')
#             reasoning_text = qa_pair.get('reasoning_text', '')

#             if not question or not reasoning_text:
#                 continue

#             # Clean texts
#             question_clean = question.replace('<|vision_start|>', '').replace('<|vision_end|>', '').replace('<|image_pad|>', '').strip()
#             reasoning_clean = reasoning_text.split('<final>')[0].strip()  # Remove final answer

#             # BERTScore: reasoning vs question
#             try:
#                 P, R, F1 = bert_score_fn([reasoning_clean], [question_clean], lang='en', verbose=False)
#                 bertscore_data['reasoning_vs_question']['precision'].append(P.item())
#                 bertscore_data['reasoning_vs_question']['recall'].append(R.item())
#                 bertscore_data['reasoning_vs_question']['f1'].append(F1.item())
#             except Exception as e:
#                 print(f"⚠️  BERTScore error: {e}")

#             # BERTScore: positive vs negative sentences
#             contrastive = qa_pair.get('contrastive')
#             if contrastive:
#                 pos_sentence = contrastive.get('positive_sentence', '')
#                 neg_sentence = contrastive.get('negative_sentence', '')

#                 if pos_sentence and neg_sentence:
#                     try:
#                         P, R, F1 = bert_score_fn([pos_sentence], [neg_sentence], lang='en', verbose=False)
#                         bertscore_data['positive_vs_negative']['precision'].append(P.item())
#                         bertscore_data['positive_vs_negative']['recall'].append(R.item())
#                         bertscore_data['positive_vs_negative']['f1'].append(F1.item())
#                     except Exception as e:
#                         print(f"⚠️  BERTScore error: {e}")

#         # Compute averages
#         summary = {}

#         for key, data in bertscore_data.items():
#             if data['f1']:
#                 summary[key] = {
#                     'precision': np.mean(data['precision']),
#                     'recall': np.mean(data['recall']),
#                     'f1': np.mean(data['f1']),
#                     'count': len(data['f1'])
#                 }

#         self.metrics['bertscore'] = summary
#         return summary

#     # ========================================================================
#     # REPORT GENERATION
#     # ========================================================================

#     def generate_report(self, output_file: str = "evaluation_report.txt"):
#         """Generate comprehensive evaluation report."""

#         report_lines = []
#         report_lines.append("=" * 80)
#         report_lines.append("COMPREHENSIVE EVALUATION REPORT")
#         report_lines.append("Thought Anchor Detection + PCA Context Vector Steering")
#         report_lines.append("=" * 80)
#         report_lines.append("")

#         # 1. Accuracy Metrics
#         if 'accuracy' in self.metrics:
#             report_lines.append("1. ACCURACY METRICS (EM / F1)")
#             report_lines.append("-" * 80)

#             acc = self.metrics['accuracy']

#             report_lines.append(f"Baseline (Scale 0.0):")
#             report_lines.append(f"  EM:  {acc['baseline']['em']:.2%}")
#             report_lines.append(f"  F1:  {acc['baseline']['f1']:.4f}")
#             report_lines.append("")

#             report_lines.append(f"Best Scale:")
#             report_lines.append(f"  EM:  {acc['best_scale']['em']:.2%}")
#             report_lines.append(f"  F1:  {acc['best_scale']['f1']:.4f}")
#             report_lines.append(f"  Δ EM: {acc['best_scale']['em'] - acc['baseline']['em']:+.2%}")
#             report_lines.append("")

#             report_lines.append(f"Per-Scale Results:")
#             for scale in sorted(acc['per_scale'].keys(), key=float):
#                 s = acc['per_scale'][scale]
#                 report_lines.append(f"  Scale {scale:>4s}: EM={s['em']:.2%}, F1={s['f1']:.4f} (n={s['count']})")
#             report_lines.append("")

#         # 2. Correlation Metrics
#         if 'correlation' in self.metrics and self.metrics['correlation']:
#             report_lines.append("2. CORRELATION METRICS (Spearman's ρ)")
#             report_lines.append("-" * 80)

#             corr = self.metrics['correlation']

#             if 'anchor_vs_prob_delta' in corr:
#                 c = corr['anchor_vs_prob_delta']
#                 report_lines.append(f"Anchor Score vs Prob Delta:")
#                 report_lines.append(f"  ρ = {c['spearman_rho']:+.4f}  (p={c['p_value']:.4f}, n={c['n_samples']})")

#             if 'anchor_vs_pca_improvement' in corr:
#                 c = corr['anchor_vs_pca_improvement']
#                 report_lines.append(f"Anchor Score vs PCA Improvement:")
#                 report_lines.append(f"  ρ = {c['spearman_rho']:+.4f}  (p={c['p_value']:.4f}, n={c['n_samples']})")

#             report_lines.append("")

#         # 3. Hallucination Metrics
#         if 'hallucination' in self.metrics:
#             report_lines.append("3. HALLUCINATION METRICS")
#             report_lines.append("-" * 80)

#             hall = self.metrics['hallucination']

#             report_lines.append(f"Baseline (Scale 0.0):")
#             report_lines.append(f"  Hallucination Rate: {hall['baseline']['hallucination_rate']:.2%}  (n={hall['baseline']['count']})")
#             report_lines.append("")

#             report_lines.append(f"Best Scale:")
#             report_lines.append(f"  Hallucination Rate: {hall['best_scale']['hallucination_rate']:.2%}  (n={hall['best_scale']['count']})")
#             report_lines.append(f"  Δ Rate: {hall['best_scale']['hallucination_rate'] - hall['baseline']['hallucination_rate']:+.2%}")
#             report_lines.append("")

#             report_lines.append(f"Per-Scale Results:")
#             for scale in sorted(hall['per_scale'].keys(), key=float):
#                 s = hall['per_scale'][scale]
#                 report_lines.append(f"  Scale {scale:>4s}: {s['hallucination_rate']:.2%}  (n={s['count']})")
#             report_lines.append("")

#         # 4. Efficiency Metrics
#         if 'efficiency' in self.metrics:
#             report_lines.append("4. EFFICIENCY METRICS")
#             report_lines.append("-" * 80)

#             eff = self.metrics['efficiency']

#             report_lines.append(f"Total Generations: {eff['total_generations']}")
#             report_lines.append(f"Avg Trials per Scale:")
#             for scale in sorted(eff['avg_trials_per_scale'].keys(), key=float):
#                 avg_trials = eff['avg_trials_per_scale'][scale]
#                 report_lines.append(f"  Scale {scale:>4s}: {avg_trials:.1f} trials")
#             report_lines.append("")

#         # 5. BERTScore Metrics
#         if 'bertscore' in self.metrics and self.metrics['bertscore']:
#             report_lines.append("5. BERTSCORE METRICS (Semantic Alignment)")
#             report_lines.append("-" * 80)

#             bert = self.metrics['bertscore']

#             if 'reasoning_vs_question' in bert:
#                 b = bert['reasoning_vs_question']
#                 report_lines.append(f"Reasoning vs Question:")
#                 report_lines.append(f"  Precision: {b['precision']:.4f}")
#                 report_lines.append(f"  Recall:    {b['recall']:.4f}")
#                 report_lines.append(f"  F1:        {b['f1']:.4f}")
#                 report_lines.append(f"  (n={b['count']})")
#                 report_lines.append("")

#             if 'positive_vs_negative' in bert:
#                 b = bert['positive_vs_negative']
#                 report_lines.append(f"Positive vs Negative Sentences:")
#                 report_lines.append(f"  Precision: {b['precision']:.4f}")
#                 report_lines.append(f"  Recall:    {b['recall']:.4f}")
#                 report_lines.append(f"  F1:        {b['f1']:.4f}")
#                 report_lines.append(f"  (n={b['count']})")
#                 report_lines.append("")

#         report_lines.append("=" * 80)

#         # Print to console
#         report_text = "\n".join(report_lines)
#         print(report_text)

#         # Save to file
#         with open(output_file, 'w') as f:
#             f.write(report_text)

#         print(f"\n✅ Report saved to: {output_file}")

#         return report_text

#     def run_all_metrics(self):
#         """Run all evaluation metrics."""
#         print("\n" + "=" * 80)
#         print("Running Comprehensive Evaluation")
#         print("=" * 80 + "\n")

#         self.load_results()

#         print("\n[1/5] Computing Accuracy Metrics...")
#         self.compute_accuracy_metrics()

#         print("[2/5] Computing Correlation Metrics...")
#         self.compute_correlation_metrics()

#         print("[3/5] Computing Hallucination Metrics...")
#         self.compute_hallucination_metrics()

#         print("[4/5] Computing Efficiency Metrics...")
#         self.compute_efficiency_metrics()

#         print("[5/5] Computing BERTScore Metrics...")
#         self.compute_bertscore_metrics()

#         print("\n✅ All metrics computed!")

#         return self


# def main():
#     """Run comprehensive evaluation."""
#     evaluator = MetricsEvaluator("anchor_vectors_output")
#     evaluator.run_all_metrics()
#     evaluator.generate_report("evaluation_report.txt")

#     # Also save JSON for programmatic access
#     import json
#     with open("evaluation_metrics.json", 'w') as f:
#         json.dump(evaluator.metrics, f, indent=2)

#     print(f"✅ Metrics saved to: evaluation_metrics.json")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Comprehensive evaluation script for Thought Anchor Detection + PCA Context Vector experiments.
"""

import json
import glob
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import re

# Optional imports
try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not installed. Correlation metrics will be skipped.")

try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("⚠️  bert-score not installed. BERTScore metrics will be skipped.")

class MetricsEvaluator:
    """Comprehensive metrics evaluator for VQA reasoning experiments."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = []
        self.metrics = {}

    def load_results(self):
        """Load all JSON results from output directory."""
        # [수정됨] 우리가 파일명을 바꿨으므로 example_*.json 대신 *.json으로 모든 파일 검색
        result_files = sorted(self.results_dir.glob("*.json"))

        if not result_files:
            raise FileNotFoundError(f"No result files found in {self.results_dir}")

        print(f"📂 Loading results from: {self.results_dir}")
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 에러가 있는 파일은 제외
                    if 'error' not in data:
                        self.results.append(data)
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")

        print(f"✅ Loaded {len(self.results)} valid result files")
        return self

    # ========================================================================
    # 1. ACCURACY METRICS
    # ========================================================================

    def normalize_answer(self, answer: str) -> str:
        if not answer:
            return ""
        answer = str(answer).strip().lower()
        answer = re.sub(r'[^\w\s]', '', answer)
        return answer

    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self.normalize_answer(prediction).split()
        gold_tokens = self.normalize_answer(ground_truth).split()

        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return int(pred_tokens == gold_tokens)

        common = set(pred_tokens) & set(gold_tokens)
        if len(common) == 0:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def compute_accuracy_metrics(self) -> Dict:
        accuracy_data = {
            'baseline': {'em': [], 'f1': []},
            'best_scale': {'em': [], 'f1': []},
            'per_scale': defaultdict(lambda: {'em': [], 'f1': []})
        }

        for result in self.results:
            qa_pairs = result.get('qa_pairs', [])
            if not qa_pairs: continue

            contrastive = qa_pairs[0].get('contrastive')
            if not contrastive: continue

            correct_answer = contrastive.get('correct_answer', '')
            pca_context = contrastive.get('pca_context')
            if not pca_context: continue

            pca_results = pca_context.get('results', {})
            
            # Find best scale logic
            best_scale = None
            best_accuracy = -1
            for scale, scale_result in pca_results.items():
                if scale_result.get('accuracy', 0.0) > best_accuracy:
                    best_accuracy = scale_result.get('accuracy', 0.0)
                    best_scale = scale

                generated_answers = scale_result.get('generated_answers', [])
                for gen_answer in generated_answers:
                    em = int(self.normalize_answer(gen_answer) == self.normalize_answer(correct_answer))
                    f1 = self.compute_f1(gen_answer, correct_answer)
                    accuracy_data['per_scale'][scale]['em'].append(em)
                    accuracy_data['per_scale'][scale]['f1'].append(f1)

                    if scale == '0.0':
                        accuracy_data['baseline']['em'].append(em)
                        accuracy_data['baseline']['f1'].append(f1)

            if best_scale and best_scale in pca_results:
                generated_answers = pca_results[best_scale].get('generated_answers', [])
                for gen_answer in generated_answers:
                    em = int(self.normalize_answer(gen_answer) == self.normalize_answer(correct_answer))
                    f1 = self.compute_f1(gen_answer, correct_answer)
                    accuracy_data['best_scale']['em'].append(em)
                    accuracy_data['best_scale']['f1'].append(f1)

        summary = {
            'baseline': {
                'em': np.mean(accuracy_data['baseline']['em']) if accuracy_data['baseline']['em'] else 0.0,
                'f1': np.mean(accuracy_data['baseline']['f1']) if accuracy_data['baseline']['f1'] else 0.0
            },
            'best_scale': {
                'em': np.mean(accuracy_data['best_scale']['em']) if accuracy_data['best_scale']['em'] else 0.0,
                'f1': np.mean(accuracy_data['best_scale']['f1']) if accuracy_data['best_scale']['f1'] else 0.0
            },
            'per_scale': {}
        }

        for scale, data in accuracy_data['per_scale'].items():
            summary['per_scale'][scale] = {
                'em': np.mean(data['em']) if data['em'] else 0.0,
                'f1': np.mean(data['f1']) if data['f1'] else 0.0,
                'count': len(data['em'])
            }

        self.metrics['accuracy'] = summary
        return summary

    # ========================================================================
    # 2. CORRELATION METRICS
    # ========================================================================

    def compute_correlation_metrics(self) -> Dict:
        if not HAS_SCIPY: return {}

        correlation_data = {
            'anchor_vs_prob_delta': [],
            'anchor_vs_pca_improvement': []
        }

        for result in self.results:
            qa_pairs = result.get('qa_pairs', [])
            if not qa_pairs: continue

            qa_pair = qa_pairs[0]
            anchor_vector = qa_pair.get('anchor_vector', [])
            contrastive = qa_pair.get('contrastive')
            
            if not anchor_vector or not contrastive: continue

            # Remove last element (final token) if 0
            anchor_scores = anchor_vector[:-1] if (anchor_vector and anchor_vector[-1] == 0.0) else anchor_vector
            if not anchor_scores: continue
            
            max_anchor_score = max(anchor_scores)

            pos_prob = contrastive.get('positive_probability', 0.0)
            neg_prob = contrastive.get('negative_probability', 0.0)
            correlation_data['anchor_vs_prob_delta'].append((max_anchor_score, pos_prob - neg_prob))

            pca_context = contrastive.get('pca_context')
            if pca_context:
                pca_results = pca_context.get('results', {})
                baseline_acc = pca_results.get('0.0', {}).get('accuracy', 0.0)
                best_acc = max((r.get('accuracy', 0.0) for r in pca_results.values()), default=0.0)
                correlation_data['anchor_vs_pca_improvement'].append((max_anchor_score, best_acc - baseline_acc))

        summary = {}
        for key, data in correlation_data.items():
            if data:
                x, y = zip(*data)
                # 표준편차가 0이면 상관계수 계산 불가하므로 예외처리
                if np.std(x) == 0 or np.std(y) == 0:
                     summary[key] = {'spearman_rho': 0.0, 'p_value': 1.0, 'n_samples': len(x)}
                else:
                    rho, pval = spearmanr(x, y)
                    summary[key] = {'spearman_rho': rho, 'p_value': pval, 'n_samples': len(x)}

        self.metrics['correlation'] = summary
        return summary

    # ========================================================================
    # 3. HALLUCINATION METRICS
    # ========================================================================

    def detect_hallucination(self, reasoning: str, options: List[str]) -> Dict:
        mentioned_letters = set(re.findall(r'\b([A-Z])\b', reasoning))
        valid_options = set(opt.strip().upper() for opt in options)
        hallucinated = mentioned_letters - valid_options
        return {'has_hallucination': len(hallucinated) > 0}

    def compute_hallucination_metrics(self) -> Dict:
        hallucination_data = {
            'baseline': [],
            'best_scale': [],
            'per_scale': defaultdict(list)
        }

        for result in self.results:
            qa_pairs = result.get('qa_pairs', [])
            if not qa_pairs: continue

            question = qa_pairs[0].get('question', '')
            options_match = re.findall(r'Options?:\s*\n((?:[-•]\s*[^\n]+\n?)+)', question, re.IGNORECASE)
            if not options_match: continue
            
            options = re.findall(r'[-•]\s*([A-Z])\b', options_match[0])
            contrastive = qa_pairs[0].get('contrastive')
            if not contrastive: continue

            pca_results = contrastive.get('pca_context', {}).get('results', {})
            if not pca_results: continue

            best_scale = max(pca_results.keys(), key=lambda s: pca_results[s].get('accuracy', 0.0), default=None)

            for scale, scale_result in pca_results.items():
                for gen_answer in scale_result.get('generated_answers', []):
                    halluc = self.detect_hallucination(str(gen_answer), options)
                    hallucination_data['per_scale'][scale].append(halluc['has_hallucination'])
                    
                    if scale == '0.0': hallucination_data['baseline'].append(halluc['has_hallucination'])
                    if scale == best_scale: hallucination_data['best_scale'].append(halluc['has_hallucination'])

        summary = {}
        for key in ['baseline', 'best_scale']:
            data = hallucination_data[key]
            summary[key] = {'hallucination_rate': np.mean(data) if data else 0.0, 'count': len(data)}
        
        summary['per_scale'] = {}
        for scale, data in hallucination_data['per_scale'].items():
            summary['per_scale'][scale] = {'hallucination_rate': np.mean(data) if data else 0.0, 'count': len(data)}

        self.metrics['hallucination'] = summary
        return summary

    # ========================================================================
    # 4. EFFICIENCY METRICS
    # ========================================================================

    def compute_efficiency_metrics(self) -> Dict:
        efficiency_data = {'avg_trials_per_scale': defaultdict(list), 'total_generations': 0}

        for result in self.results:
            qa_pairs = result.get('qa_pairs', [])
            if not qa_pairs: continue
            
            pca_results = qa_pairs[0].get('contrastive', {}).get('pca_context', {}).get('results', {})
            for scale, scale_result in pca_results.items():
                total = scale_result.get('total_trials', 0)
                efficiency_data['avg_trials_per_scale'][scale].append(total)
                efficiency_data['total_generations'] += total

        summary = {
            'total_generations': efficiency_data['total_generations'],
            'avg_trials_per_scale': {k: np.mean(v) for k, v in efficiency_data['avg_trials_per_scale'].items()}
        }
        self.metrics['efficiency'] = summary
        return summary

    # ========================================================================
    # 5. BERTSCORE METRICS
    # ========================================================================

    def compute_bertscore_metrics(self) -> Dict:
        if not HAS_BERTSCORE: return {}
        bertscore_data = {'reasoning_vs_question': {'f1': []}, 'positive_vs_negative': {'f1': []}}

        # Batch processing for speed could be added here, but doing sequential for simplicity
        for result in self.results:
            qa_pairs = result.get('qa_pairs', [])
            if not qa_pairs: continue
            
            qa = qa_pairs[0]
            question = qa.get('question', '').replace('<|vision_start|>', '').split('\n')[-1] # Simple clean
            reasoning = qa.get('reasoning_text', '').split('<final>')[0].strip()

            contrastive = qa.get('contrastive', {})
            pos = contrastive.get('positive_sentence', '')
            neg = contrastive.get('negative_sentence', '')

            if reasoning and question:
                try:
                    _, _, F1 = bert_score_fn([reasoning], [question], lang='en', verbose=False)
                    bertscore_data['reasoning_vs_question']['f1'].append(F1.item())
                except: pass
            
            if pos and neg:
                try:
                    _, _, F1 = bert_score_fn([pos], [neg], lang='en', verbose=False)
                    bertscore_data['positive_vs_negative']['f1'].append(F1.item())
                except: pass

        summary = {}
        for k, v in bertscore_data.items():
            summary[k] = {'f1': np.mean(v['f1']) if v['f1'] else 0.0, 'count': len(v['f1'])}
        
        self.metrics['bertscore'] = summary
        return summary

    # ========================================================================
    # REPORT GENERATION
    # ========================================================================

    def generate_report(self, output_file: str):
        lines = ["="*80, "COMPREHENSIVE EVALUATION REPORT", "="*80, ""]
        
        # Helper to safely get nested dict
        def get(d, keys):
            for k in keys: d = d.get(k, {})
            return d

        # Accuracy
        acc = self.metrics.get('accuracy', {})
        if acc:
            lines += ["1. ACCURACY", "-"*40]
            lines.append(f"Baseline EM: {acc['baseline']['em']:.2%} | F1: {acc['baseline']['f1']:.4f}")
            lines.append(f"Best Scale EM: {acc['best_scale']['em']:.2%} | F1: {acc['best_scale']['f1']:.4f}")
            lines.append("")
        
        # Correlation
        corr = self.metrics.get('correlation', {})
        if corr:
            lines += ["2. CORRELATION", "-"*40]
            for k, v in corr.items():
                lines.append(f"{k}: rho={v.get('spearman_rho',0):.4f} (p={v.get('p_value',1):.4f})")
            lines.append("")

        # Save
        report_text = "\n".join(lines)
        print(report_text)
        with open(output_file, 'w') as f: f.write(report_text)
        print(f"\n✅ Report saved to: {output_file}")

    def run_all(self):
        self.load_results()
        self.compute_accuracy_metrics()
        self.compute_correlation_metrics()
        self.compute_hallucination_metrics()
        self.compute_efficiency_metrics()
        self.compute_bertscore_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="result_output_10ea", help="Folder containing JSON results")
    parser.add_argument("--output-report", default="evaluation_report.txt")
    args = parser.parse_args()

    evaluator = MetricsEvaluator(args.input_dir)
    try:
        evaluator.run_all()
        evaluator.generate_report(args.output_report)
        
        # Save JSON
        json_out = args.output_report.replace('.txt', '.json')
        with open(json_out, 'w') as f:
            json.dump(evaluator.metrics, f, indent=2)
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")