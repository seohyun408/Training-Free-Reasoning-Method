#!/usr/bin/env python3
"""
Thought Anchor 결과를 HTML 리포트로 생성
"""

import json
import base64
from pathlib import Path
from PIL import Image
import io

def encode_image(image_path):
    """이미지를 base64로 인코딩"""
    try:
        with Image.open(image_path) as img:
            # 크기 조정 (너무 크면 400px로 제한)
            if img.width > 400:
                ratio = 400 / img.width
                new_height = int(img.height * ratio)
                img = img.resize((400, new_height), Image.Resampling.LANCZOS)

            # Base64 인코딩
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"[warn] Failed to encode image {image_path}: {e}")
        return None


def generate_html_report(json_files, output_path="thought_anchor_report.html"):
    """HTML 리포트 생성"""

    html_parts = []
    html_parts.append("""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thought Anchor Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
                      
        .example {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 40px;
            overflow: hidden;
        }
        .example-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            font-size: 18px;
            font-weight: bold;
        }
        .qa-pair {
            padding: 20px;
            border-bottom: 1px solid #eee;
        }
        .qa-pair:last-child {
            border-bottom: none;
        }
        .content-grid {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 30px;
            margin-bottom: 20px;
        }
        .image-section img {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .text-section {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .question {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }
        .question-label {
            font-weight: bold;
            color: #1976d2;
            margin-bottom: 5px;
        }
        .reasoning {
            background: #f3e5f5;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #9c27b0;
        }
                      
        .reasoning-label {
            font-weight: bold;
            color: #7b1fa2;
            margin-bottom: 10px;
        }
        .reasoning-section {
            margin: 8px 0;
            line-height: 1.6;
        }
        .section-tag {
            font-weight: bold;
            color: #6a1b9a;
        }
        .anchor-section {
            margin-top: 20px;
        }
        .anchor-header {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .anchor-item {
            background: #fff;
            border-left: 5px solid #ddd;
            padding: 12px 15px;
            margin-bottom: 10px;
            border-radius: 4px;
            transition: all 0.3s;
        }
        .anchor-item:hover {
            transform: translateX(5px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .rank-1 { border-left-color: #f44336; background: #ffebee; }
        .rank-2 { border-left-color: #ff9800; background: #fff3e0; }
        .rank-3 { border-left-color: #ffc107; background: #fffde7; }
        .rank-4 { border-left-color: #4caf50; background: #e8f5e9; }
        .rank-5 { border-left-color: #2196f3; background: #e3f2fd; }
        .rank-title {
            font-weight: bold;
            margin-bottom: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .rank-label {
            font-size: 16px;
        }
        .rank-score {
            background: rgba(0,0,0,0.1);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 14px;
        }
        .rank-text {
            color: #555;
            font-size: 14px;
            line-height: 1.5;
        }
        .error-box {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f44336;
        }
        .no-anchors {
            color: #999;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }
        .contrastive-section {
            margin-top: 20px;
        }
        .contrastive-header {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .contrastive-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .positive-box {
            background: #e8f5e9;
            border-left: 5px solid #4caf50;
            padding: 15px;
            border-radius: 8px;
        }
        .negative-box {
            background: #ffebee;
            border-left: 5px solid #f44336;
            padding: 15px;
            border-radius: 8px;
        }
        .contrastive-label {
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .contrastive-prob {
            background: rgba(0,0,0,0.1);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 14px;
        }
        .contrastive-text {
            color: #555;
            line-height: 1.6;
            font-size: 14px;
        }
        .samples-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .samples-table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        .samples-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }
        .samples-table tr:last-child td {
            border-bottom: none;
        }
        .samples-table tr:hover {
            background: #f5f5f5;
        }
        .sample-rank {
            font-weight: bold;
            color: #667eea;
        }
        .pca-section {
            margin-top: 20px;
            background: #f0f4ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .pca-header {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .pca-description {
            color: #666;
            font-size: 14px;
            margin-bottom: 15px;
            line-height: 1.5;
        }
        .pca-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .pca-table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        .pca-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }
        .pca-table tr:last-child td {
            border-bottom: none;
        }
        .pca-table tr:hover {
            background: #f5f5f5;
        }
        .pca-table tr.best-scale {
            background: #e8f5e9;
            font-weight: bold;
        }
        .pca-table tr.best-scale:hover {
            background: #d4edda;
        }
        .trials-details {
            margin-top: 15px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .trials-header {
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #667eea;
        }
        .trial-item {
            padding: 8px 12px;
            margin: 6px 0;
            border-left: 3px solid #ddd;
            background: #fafafa;
            border-radius: 4px;
        }
        .trial-item.correct {
            border-left-color: #28a745;
            background: #f0f9f4;
        }
        .trial-item.incorrect {
            border-left-color: #dc3545;
            background: #fff5f5;
        }
        .trial-label {
            font-weight: bold;
            margin-right: 8px;
        }
        .trial-answer {
            font-family: 'Courier New', monospace;
            background: white;
            padding: 2px 6px;
            border-radius: 3px;
            border: 1px solid #ddd;
        }
        .trial-prob {
            color: #666;
            font-size: 12px;
            margin-left: 10px;
        }
        @media (max-width: 900px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
            .contrastive-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Thought Anchor Analysis Report</h1>
""")

    # 각 예제 처리
    for json_file in sorted(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        example_name = json_file.stem

        html_parts.append(f"""
        <div class="example">
            <div class="example-header">📊 {example_name.upper()}</div>
""")

        # 에러 체크
        if 'error' in data:
            html_parts.append(f"""
            <div class="qa-pair">
                <div class="error-box">❌ Error: {data['error']}</div>
            </div>
""")
            html_parts.append("        </div>\n")
            continue

        if data.get('successful_pairs', 0) == 0:
            html_parts.append("""
            <div class="qa-pair">
                <div class="no-anchors">⚠️ No successful QA pairs</div>
            </div>
""")
            html_parts.append("        </div>\n")
            continue

        # 이미지 인코딩
        image_path = data.get('image_path')
        image_data = None
        if image_path and Path(image_path).exists():
            image_data = encode_image(image_path)

        # QA pairs 처리
        for qa_idx, qa_pair in enumerate(data.get('qa_pairs', [])):
            html_parts.append(f"""
            <div class="qa-pair">
""")

            # Question과 Reasoning을 그리드로 배치
            html_parts.append("""
                <div class="content-grid">
""")

            # 왼쪽: 이미지
            html_parts.append("""
                    <div class="image-section">
""")
            if image_data:
                html_parts.append(f"""
                        <img src="{image_data}" alt="Question Image">
""")
            else:
                html_parts.append("""
                        <div style="background: #f0f0f0; padding: 20px; text-align: center; border-radius: 8px;">
                            No image available
                        </div>
""")
            html_parts.append("""
                    </div>
""")

            # 오른쪽: Question과 Reasoning
            html_parts.append("""
                    <div class="text-section">
""")

            # Question
            question = qa_pair.get('question', '')
            if '<|vision_start|>' in question:
                question = question.split('\n')[-1]

            html_parts.append(f"""
                        <div class="question">
                            <div class="question-label">❓ Question:</div>
                            <div>{question}</div>
                        </div>
""")

            # Reasoning
            reasoning = qa_pair.get('reasoning_text', '')

            # Reasoning을 섹션별로 파싱
            reasoning_html = ""
            if '<SUMMARY>' in reasoning:
                # LLaVA-CoT 형식
                sections = {
                    'SUMMARY': '',
                    'CAPTION': '',
                    'REASONING': '',
                    'CONCLUSION': ''
                }
                for tag in sections.keys():
                    if f'<{tag}>' in reasoning and f'</{tag}>' in reasoning:
                        start = reasoning.find(f'<{tag}>') + len(f'<{tag}>')
                        end = reasoning.find(f'</{tag}>')
                        sections[tag] = reasoning[start:end].strip()

                for tag, content in sections.items():
                    if content:
                        reasoning_html += f'<div class="reasoning-section"><span class="section-tag">&lt;{tag}&gt;</span> {content} <span class="section-tag">&lt;/{tag}&gt;</span></div>'
            elif '<think>' in reasoning or '<reasoning>' in reasoning:
                # <think> 형식
                reasoning_html = reasoning.replace('\n', '<br>')
            else:
                reasoning_html = reasoning.replace('\n', '<br>')

            html_parts.append(f"""
                        <div class="reasoning">
                            <div class="reasoning-label">💭 Reasoning:</div>
                            <div>{reasoning_html}</div>
                        </div>
""")

            html_parts.append("""
                    </div>
                </div>
""")  # content-grid 끝

            # Thought Anchor Rankings
            chunks = qa_pair.get('chunks', [])
            anchor_vector = qa_pair.get('anchor_vector', [])

            if len(chunks) > 0 and len(anchor_vector) > 0:
                # Anchor scores와 함께 정렬
                anchor_pairs = list(zip(range(len(chunks)), chunks, anchor_vector))
                anchor_pairs.sort(key=lambda x: x[2], reverse=True)

                # Top 5만 표시
                top_anchors = anchor_pairs[:5]

                html_parts.append("""
                <div class="anchor-section">
                    <div class="anchor-header">🏆 Thought Anchor Rankings</div>
""")

                if any(score > 0 for _, _, score in top_anchors):
                    for rank, (chunk_idx, chunk, score) in enumerate(top_anchors, 1):
                        # 텍스트 정리
                        text = chunk.replace('<reasoning>', '').replace('</reasoning>', '')
                        text = text.replace('<think>', '').replace('</think>', '')
                        text = text.replace('<final>', '').replace('</final>', '')
                        text = text.strip()

                        if len(text) > 150:
                            text = text[:150] + "..."

                        html_parts.append(f"""
                    <div class="anchor-item rank-{rank}">
                        <div class="rank-title">
                            <span class="rank-label">Rank {rank} (Sentence {chunk_idx}):</span>
                            <span class="rank-score">Score: {score:.4f}</span>
                        </div>
                        <div class="rank-text">Text: "{text}"</div>
                    </div>
""")
                else:
                    html_parts.append("""
                    <div class="no-anchors">⚠️ All anchor scores are zero (attention masking may have failed)</div>
""")

                html_parts.append("""
                </div>
""")  # anchor-section 끝
            else:
                html_parts.append("""
                <div class="anchor-section">
                    <div class="no-anchors">⚠️ No chunks or anchor scores available</div>
                </div>
""")

            # Contrastive Generation Results
            contrastive = qa_pair.get('contrastive')
            if contrastive and contrastive.get('positive_sentence'):
                html_parts.append("""
                <div class="contrastive-section">
                    <div class="contrastive-header">🔄 Contrastive Generation Results</div>
""")

                # Positive and Negative sentences side by side
                html_parts.append("""
                    <div class="contrastive-grid">
""")

                # Positive sentence
                pos_sentence = contrastive.get('positive_sentence', '')
                pos_prob = contrastive.get('positive_probability', 0.0)
                html_parts.append(f"""
                        <div class="positive-box">
                            <div class="contrastive-label">
                                <span>✅ Positive Sentence</span>
                                <span class="contrastive-prob">Prob: {pos_prob:.4f}</span>
                            </div>
                            <div class="contrastive-text">{pos_sentence}</div>
                        </div>
""")

                # Negative sentence
                neg_sentence = contrastive.get('negative_sentence', '')
                neg_prob = contrastive.get('negative_probability', 0.0)
                html_parts.append(f"""
                        <div class="negative-box">
                            <div class="contrastive-label">
                                <span>❌ Negative Sentence</span>
                                <span class="contrastive-prob">Prob: {neg_prob:.4f}</span>
                            </div>
                            <div class="contrastive-text">{neg_sentence}</div>
                        </div>
""")

                html_parts.append("""
                    </div>
""")  # contrastive-grid 끝

                # All samples table
                all_samples = contrastive.get('all_samples', [])
                if all_samples:
                    html_parts.append("""
                    <table class="samples-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Probability</th>
                                <th>First Sentence</th>
                                <th>Final Answer</th>
                            </tr>
                        </thead>
                        <tbody>
""")

                    for i, sample in enumerate(all_samples, 1):
                        first_sentence = sample.get('first_sentence', '')[:200]
                        if len(sample.get('first_sentence', '')) > 200:
                            first_sentence += "..."

                        final_answer = sample.get('final_answer', 'N/A')
                        prob = sample.get('answer_probability', 0.0)

                        html_parts.append(f"""
                            <tr>
                                <td class="sample-rank">#{i}</td>
                                <td>{prob:.4f}</td>
                                <td>{first_sentence}</td>
                                <td><strong>{final_answer}</strong></td>
                            </tr>
""")

                    html_parts.append("""
                        </tbody>
                    </table>
""")

                html_parts.append("""
                </div>
""")  # contrastive-section 끝

                # PCA Context Vector Results
                pca_context = contrastive.get('pca_context')
                if pca_context and pca_context.get('results'):
                    html_parts.append("""
                <div class="pca-section">
                    <div class="pca-header">🧪 PCA Context Vector Results</div>
                    <p class="pca-description">
                        Testing the effect of adding PCA-extracted context vector (from positive - negative hidden states)
                        to decoder hidden states during generation.
                    </p>
""")

                    results = pca_context['results']

                    # Create comparison table
                    html_parts.append("""
                    <table class="pca-table">
                        <thead>
                            <tr>
                                <th>Context Scale</th>
                                <th>Accuracy</th>
                                <th>Avg Probability</th>
                                <th>Correct / Total</th>
                            </tr>
                        </thead>
                        <tbody>
""")

                    # Sort by scale
                    for scale in sorted(results.keys(), key=lambda x: float(x)):
                        result = results[scale]
                        accuracy = result.get('accuracy', 0.0)
                        avg_prob = result.get('avg_probability', 0.0)
                        correct = result.get('correct_count', 0)
                        total = result.get('total_trials', 0)

                        # Highlight best accuracy
                        row_class = "best-scale" if accuracy == max(r.get('accuracy', 0) for r in results.values()) else ""

                        html_parts.append(f"""
                            <tr class="{row_class}">
                                <td><strong>{scale}</strong></td>
                                <td>{accuracy:.1%}</td>
                                <td>{avg_prob:.4f}</td>
                                <td>{correct} / {total}</td>
                            </tr>
""")

                    html_parts.append("""
                        </tbody>
                    </table>
""")

                    # Add detailed trial results for each scale
                    correct_answer = contrastive.get('correct_answer', '')

                    for scale in sorted(results.keys(), key=lambda x: float(x)):
                        result = results[scale]
                        generated_answers = result.get('generated_answers', [])
                        answer_probs = result.get('answer_probabilities', [])

                        if generated_answers:
                            html_parts.append(f"""
                    <div class="trials-details">
                        <div class="trials-header">📋 Scale {scale} - Individual Trial Results</div>
""")

                            for trial_idx, (answer, prob) in enumerate(zip(generated_answers, answer_probs), 1):
                                # Normalize answers for comparison
                                def normalize_answer(ans):
                                    if not ans:
                                        return ""
                                    return str(ans).strip().upper()

                                is_correct = normalize_answer(answer) == normalize_answer(correct_answer)
                                trial_class = "correct" if is_correct else "incorrect"
                                icon = "✅" if is_correct else "❌"

                                # Truncate long answers
                                display_answer = answer if len(str(answer)) <= 100 else str(answer)[:100] + "..."
                                if not display_answer:
                                    display_answer = "(empty answer)"

                                html_parts.append(f"""
                        <div class="trial-item {trial_class}">
                            <span class="trial-label">{icon} Trial {trial_idx}:</span>
                            <span class="trial-answer">"{display_answer}"</span>
                            <span class="trial-prob">(prob: {prob:.4f})</span>
                        </div>
""")

                            html_parts.append("""
                    </div>
""")

                    html_parts.append("""
                </div>
""")  # pca-section 끝

            html_parts.append("""
            </div>
""")  # qa-pair 끝

        html_parts.append("""
        </div>
""")  # example 끝

    # HTML 마무리
    html_parts.append("""
    </div>
</body>
</html>
""")

    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))

    print(f"✅ HTML report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    output_dir = Path('anchor_vectors_output')
    result_files = sorted(output_dir.glob('example_*.json'))

    if not result_files:
        print("❌ No result files found in anchor_vectors_output/")
    else:
        print(f"Found {len(result_files)} result files")
        output_path = generate_html_report(result_files)
        print(f"\n🎉 Open the report: {Path(output_path).absolute()}")
