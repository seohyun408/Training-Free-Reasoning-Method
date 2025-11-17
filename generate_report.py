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
        @media (max-width: 900px) {
            .content-grid {
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
