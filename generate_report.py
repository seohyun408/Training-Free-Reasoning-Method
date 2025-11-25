# #!/usr/bin/env python3
# """
# Thought Anchor 결과를 HTML 리포트로 생성
# """

# import json
# import base64
# from pathlib import Path
# from PIL import Image
# import io

# def encode_image(image_path):
#     """이미지를 base64로 인코딩"""
#     try:
#         with Image.open(image_path) as img:
#             # 크기 조정 (너무 크면 400px로 제한)
#             if img.width > 400:
#                 ratio = 400 / img.width
#                 new_height = int(img.height * ratio)
#                 img = img.resize((400, new_height), Image.Resampling.LANCZOS)

#             # Base64 인코딩
#             buffered = io.BytesIO()
#             img.save(buffered, format="PNG")
#             img_str = base64.b64encode(buffered.getvalue()).decode()
#             return f"data:image/png;base64,{img_str}"
#     except Exception as e:
#         print(f"[warn] Failed to encode image {image_path}: {e}")
#         return None


# def generate_html_report(json_files, output_path="thought_anchor_report.html"):
#     """HTML 리포트 생성"""

#     html_parts = []
#     html_parts.append("""
# <!DOCTYPE html>
# <html lang="ko">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Thought Anchor Analysis Report</title>
#     <style>
#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             margin: 0;
#             padding: 20px;
#             background-color: #f5f5f5;
#         }
#         .container {
#             max-width: 1400px;
#             margin: 0 auto;
#         }
#         h1 {
#             text-align: center;
#             color: #333;
#             margin-bottom: 30px;
#         }
                      
#         .example {
#             background: white;
#             border-radius: 10px;
#             box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#             margin-bottom: 40px;
#             overflow: hidden;
#         }
#         .example-header {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             padding: 15px 20px;
#             font-size: 18px;
#             font-weight: bold;
#         }
#         .qa-pair {
#             padding: 20px;
#             border-bottom: 1px solid #eee;
#         }
#         .qa-pair:last-child {
#             border-bottom: none;
#         }
#         .content-grid {
#             display: grid;
#             grid-template-columns: 400px 1fr;
#             gap: 30px;
#             margin-bottom: 20px;
#         }
#         .image-section img {
#             width: 100%;
#             border-radius: 8px;
#             border: 1px solid #ddd;
#         }
#         .text-section {
#             display: flex;
#             flex-direction: column;
#             gap: 15px;
#         }
#         .question {
#             background: #e3f2fd;
#             padding: 15px;
#             border-radius: 8px;
#             border-left: 4px solid #2196f3;
#         }
#         .question-label {
#             font-weight: bold;
#             color: #1976d2;
#             margin-bottom: 5px;
#         }
#         .reasoning {
#             background: #f3e5f5;
#             padding: 15px;
#             border-radius: 8px;
#             border-left: 4px solid #9c27b0;
#         }
                      
#         .reasoning-label {
#             font-weight: bold;
#             color: #7b1fa2;
#             margin-bottom: 10px;
#         }
#         .reasoning-section {
#             margin: 8px 0;
#             line-height: 1.6;
#         }
#         .section-tag {
#             font-weight: bold;
#             color: #6a1b9a;
#         }
#         .anchor-section {
#             margin-top: 20px;
#         }
#         .anchor-header {
#             font-size: 20px;
#             font-weight: bold;
#             color: #333;
#             margin-bottom: 15px;
#             border-bottom: 2px solid #667eea;
#             padding-bottom: 10px;
#         }
#         .anchor-item {
#             background: #fff;
#             border-left: 5px solid #ddd;
#             padding: 12px 15px;
#             margin-bottom: 10px;
#             border-radius: 4px;
#             transition: all 0.3s;
#         }
#         .anchor-item:hover {
#             transform: translateX(5px);
#             box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#         }
#         .rank-1 { border-left-color: #f44336; background: #ffebee; }
#         .rank-2 { border-left-color: #ff9800; background: #fff3e0; }
#         .rank-3 { border-left-color: #ffc107; background: #fffde7; }
#         .rank-4 { border-left-color: #4caf50; background: #e8f5e9; }
#         .rank-5 { border-left-color: #2196f3; background: #e3f2fd; }
#         .rank-title {
#             font-weight: bold;
#             margin-bottom: 5px;
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#         }
#         .rank-label {
#             font-size: 16px;
#         }
#         .rank-score {
#             background: rgba(0,0,0,0.1);
#             padding: 3px 10px;
#             border-radius: 12px;
#             font-size: 14px;
#         }
#         .rank-text {
#             color: #555;
#             font-size: 14px;
#             line-height: 1.5;
#         }
#         .error-box {
#             background: #ffebee;
#             color: #c62828;
#             padding: 15px;
#             border-radius: 8px;
#             border-left: 4px solid #f44336;
#         }
#         .no-anchors {
#             color: #999;
#             font-style: italic;
#             padding: 20px;
#             text-align: center;
#         }
#         .contrastive-section {
#             margin-top: 20px;
#         }
#         .contrastive-header {
#             font-size: 20px;
#             font-weight: bold;
#             color: #333;
#             margin-bottom: 15px;
#             border-bottom: 2px solid #667eea;
#             padding-bottom: 10px;
#         }
#         .contrastive-grid {
#             display: grid;
#             grid-template-columns: 1fr 1fr;
#             gap: 20px;
#             margin-bottom: 20px;
#         }
#         .positive-box {
#             background: #e8f5e9;
#             border-left: 5px solid #4caf50;
#             padding: 15px;
#             border-radius: 8px;
#         }
#         .negative-box {
#             background: #ffebee;
#             border-left: 5px solid #f44336;
#             padding: 15px;
#             border-radius: 8px;
#         }
#         .contrastive-label {
#             font-weight: bold;
#             font-size: 16px;
#             margin-bottom: 10px;
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#         }
#         .contrastive-prob {
#             background: rgba(0,0,0,0.1);
#             padding: 3px 10px;
#             border-radius: 12px;
#             font-size: 14px;
#         }
#         .contrastive-text {
#             color: #555;
#             line-height: 1.6;
#             font-size: 14px;
#         }
#         .samples-table {
#             width: 100%;
#             border-collapse: collapse;
#             margin-top: 15px;
#             background: white;
#             border-radius: 8px;
#             overflow: hidden;
#             box-shadow: 0 1px 3px rgba(0,0,0,0.1);
#         }
#         .samples-table th {
#             background: #667eea;
#             color: white;
#             padding: 12px;
#             text-align: left;
#             font-weight: bold;
#         }
#         .samples-table td {
#             padding: 10px 12px;
#             border-bottom: 1px solid #eee;
#         }
#         .samples-table tr:last-child td {
#             border-bottom: none;
#         }
#         .samples-table tr:hover {
#             background: #f5f5f5;
#         }
#         .sample-rank {
#             font-weight: bold;
#             color: #667eea;
#         }
#         .pca-section {
#             margin-top: 20px;
#             background: #f0f4ff;
#             padding: 20px;
#             border-radius: 8px;
#             border-left: 4px solid #667eea;
#         }
#         .pca-header {
#             font-size: 20px;
#             font-weight: bold;
#             color: #333;
#             margin-bottom: 10px;
#         }
#         .pca-description {
#             color: #666;
#             font-size: 14px;
#             margin-bottom: 15px;
#             line-height: 1.5;
#         }
#         .pca-table {
#             width: 100%;
#             border-collapse: collapse;
#             background: white;
#             border-radius: 8px;
#             overflow: hidden;
#             box-shadow: 0 1px 3px rgba(0,0,0,0.1);
#         }
#         .pca-table th {
#             background: #667eea;
#             color: white;
#             padding: 12px;
#             text-align: left;
#             font-weight: bold;
#         }
#         .pca-table td {
#             padding: 10px 12px;
#             border-bottom: 1px solid #eee;
#         }
#         .pca-table tr:last-child td {
#             border-bottom: none;
#         }
#         .pca-table tr:hover {
#             background: #f5f5f5;
#         }
#         .pca-table tr.best-scale {
#             background: #e8f5e9;
#             font-weight: bold;
#         }
#         .pca-table tr.best-scale:hover {
#             background: #d4edda;
#         }
#         .trials-details {
#             margin-top: 15px;
#             background: white;
#             padding: 15px;
#             border-radius: 8px;
#             box-shadow: 0 1px 3px rgba(0,0,0,0.1);
#         }
#         .trials-header {
#             font-size: 16px;
#             font-weight: bold;
#             color: #333;
#             margin-bottom: 12px;
#             padding-bottom: 8px;
#             border-bottom: 2px solid #667eea;
#         }
#         .correct-answer-box {
#             background: #e3f2fd;
#             border-left: 4px solid #2196f3;
#             padding: 10px 15px;
#             margin-bottom: 15px;
#             border-radius: 6px;
#             font-size: 14px;
#         }
#         .correct-answer-label {
#             font-weight: bold;
#             color: #1976d2;
#             margin-right: 8px;
#         }
#         .correct-answer-text {
#             font-family: 'Courier New', monospace;
#             color: #333;
#             background: white;
#             padding: 4px 8px;
#             border-radius: 4px;
#             display: inline-block;
#         }
#         .trial-item {
#             padding: 8px 12px;
#             margin: 6px 0;
#             border-left: 3px solid #ddd;
#             background: #fafafa;
#             border-radius: 4px;
#         }
#         .trial-item.correct {
#             border-left-color: #28a745;
#             background: #f0f9f4;
#         }
#         .trial-item.incorrect {
#             border-left-color: #dc3545;
#             background: #fff5f5;
#         }
#         .trial-label {
#             font-weight: bold;
#             margin-right: 8px;
#         }
#         .trial-answer {
#             font-family: 'Courier New', monospace;
#             background: white;
#             padding: 2px 6px;
#             border-radius: 3px;
#             border: 1px solid #ddd;
#         }
#         .trial-prob {
#             color: #666;
#             font-size: 12px;
#             margin-left: 10px;
#         }
#         @media (max-width: 900px) {
#             .content-grid {
#                 grid-template-columns: 1fr;
#             }
#             .contrastive-grid {
#                 grid-template-columns: 1fr;
#             }
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>🎯 Thought Anchor Analysis Report</h1>
# """)

#     # 각 예제 처리
#     for json_file in sorted(json_files):
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)

#         example_name = json_file.stem

#         html_parts.append(f"""
#         <div class="example">
#             <div class="example-header">📊 {example_name.upper()}</div>
# """)

#         # 에러 체크
#         if 'error' in data:
#             html_parts.append(f"""
#             <div class="qa-pair">
#                 <div class="error-box">❌ Error: {data['error']}</div>
#             </div>
# """)
#             html_parts.append("        </div>\n")
#             continue

#         if data.get('successful_pairs', 0) == 0:
#             html_parts.append("""
#             <div class="qa-pair">
#                 <div class="no-anchors">⚠️ No successful QA pairs</div>
#             </div>
# """)
#             html_parts.append("        </div>\n")
#             continue

#         # 이미지 인코딩
#         image_path = data.get('image_path')
#         image_data = None
#         if image_path and Path(image_path).exists():
#             image_data = encode_image(image_path)

#         # QA pairs 처리
#         for qa_idx, qa_pair in enumerate(data.get('qa_pairs', [])):
#             html_parts.append(f"""
#             <div class="qa-pair">
# """)

#             # Question과 Reasoning을 그리드로 배치
#             html_parts.append("""
#                 <div class="content-grid">
# """)

#             # 왼쪽: 이미지
#             html_parts.append("""
#                     <div class="image-section">
# """)
#             if image_data:
#                 html_parts.append(f"""
#                         <img src="{image_data}" alt="Question Image">
# """)
#             else:
#                 html_parts.append("""
#                         <div style="background: #f0f0f0; padding: 20px; text-align: center; border-radius: 8px;">
#                             No image available
#                         </div>
# """)
#             html_parts.append("""
#                     </div>
# """)

#             # 오른쪽: Question과 Reasoning
#             html_parts.append("""
#                     <div class="text-section">
# """)

#             # Question
#             question = qa_pair.get('question', '')
#             if '<|vision_start|>' in question:
#                 question = question.split('\n')[-1]

#             html_parts.append(f"""
#                         <div class="question">
#                             <div class="question-label">❓ Question:</div>
#                             <div>{question}</div>
#                         </div>
# """)

#             # Reasoning
#             reasoning = qa_pair.get('reasoning_text', '')

#             # Reasoning을 섹션별로 파싱
#             reasoning_html = ""
#             if '<SUMMARY>' in reasoning:
#                 # LLaVA-CoT 형식
#                 sections = {
#                     'SUMMARY': '',
#                     'CAPTION': '',
#                     'REASONING': '',
#                     'CONCLUSION': ''
#                 }
#                 for tag in sections.keys():
#                     if f'<{tag}>' in reasoning and f'</{tag}>' in reasoning:
#                         start = reasoning.find(f'<{tag}>') + len(f'<{tag}>')
#                         end = reasoning.find(f'</{tag}>')
#                         sections[tag] = reasoning[start:end].strip()

#                 for tag, content in sections.items():
#                     if content:
#                         reasoning_html += f'<div class="reasoning-section"><span class="section-tag">&lt;{tag}&gt;</span> {content} <span class="section-tag">&lt;/{tag}&gt;</span></div>'
#             elif '<think>' in reasoning or '<reasoning>' in reasoning:
#                 # <think> 형식
#                 reasoning_html = reasoning.replace('\n', '<br>')
#             else:
#                 reasoning_html = reasoning.replace('\n', '<br>')

#             html_parts.append(f"""
#                         <div class="reasoning">
#                             <div class="reasoning-label">💭 Reasoning:</div>
#                             <div>{reasoning_html}</div>
#                         </div>
# """)

#             html_parts.append("""
#                     </div>
#                 </div>
# """)  # content-grid 끝

#             # Thought Anchor Rankings
#             chunks = qa_pair.get('chunks', [])
#             anchor_vector = qa_pair.get('anchor_vector', [])

#             if len(chunks) > 0 and len(anchor_vector) > 0:
#                 # Anchor scores와 함께 정렬
#                 anchor_pairs = list(zip(range(len(chunks)), chunks, anchor_vector))
#                 anchor_pairs.sort(key=lambda x: x[2], reverse=True)

#                 # Top 5만 표시
#                 top_anchors = anchor_pairs[:5]

#                 html_parts.append("""
#                 <div class="anchor-section">
#                     <div class="anchor-header">🏆 Thought Anchor Rankings</div>
# """)

#                 if any(score > 0 for _, _, score in top_anchors):
#                     for rank, (chunk_idx, chunk, score) in enumerate(top_anchors, 1):
#                         # 텍스트 정리
#                         text = chunk.replace('<reasoning>', '').replace('</reasoning>', '')
#                         text = text.replace('<think>', '').replace('</think>', '')
#                         text = text.replace('<final>', '').replace('</final>', '')
#                         text = text.strip()

#                         if len(text) > 150:
#                             text = text[:150] + "..."

#                         html_parts.append(f"""
#                     <div class="anchor-item rank-{rank}">
#                         <div class="rank-title">
#                             <span class="rank-label">Rank {rank} (Sentence {chunk_idx}):</span>
#                             <span class="rank-score">Score: {score:.4f}</span>
#                         </div>
#                         <div class="rank-text">Text: "{text}"</div>
#                     </div>
# """)
#                 else:
#                     html_parts.append("""
#                     <div class="no-anchors">⚠️ All anchor scores are zero (attention masking may have failed)</div>
# """)

#                 html_parts.append("""
#                 </div>
# """)  # anchor-section 끝
#             else:
#                 html_parts.append("""
#                 <div class="anchor-section">
#                     <div class="no-anchors">⚠️ No chunks or anchor scores available</div>
#                 </div>
# """)

#             # Contrastive Generation Results
#             contrastive = qa_pair.get('contrastive')
#             if contrastive and contrastive.get('positive_sentence'):
#                 html_parts.append("""
#                 <div class="contrastive-section">
#                     <div class="contrastive-header">🔄 Contrastive Generation Results</div>
# """)

#                 # Positive and Negative sentences side by side
#                 html_parts.append("""
#                     <div class="contrastive-grid">
# """)

#                 # Positive sentence
#                 pos_sentence = contrastive.get('positive_sentence', '')
#                 pos_prob = contrastive.get('positive_probability', 0.0)
#                 html_parts.append(f"""
#                         <div class="positive-box">
#                             <div class="contrastive-label">
#                                 <span>✅ Positive Sentence</span>
#                                 <span class="contrastive-prob">Prob: {pos_prob:.4f}</span>
#                             </div>
#                             <div class="contrastive-text">{pos_sentence}</div>
#                         </div>
# """)

#                 # Negative sentence
#                 neg_sentence = contrastive.get('negative_sentence', '')
#                 neg_prob = contrastive.get('negative_probability', 0.0)
#                 html_parts.append(f"""
#                         <div class="negative-box">
#                             <div class="contrastive-label">
#                                 <span>❌ Negative Sentence</span>
#                                 <span class="contrastive-prob">Prob: {neg_prob:.4f}</span>
#                             </div>
#                             <div class="contrastive-text">{neg_sentence}</div>
#                         </div>
# """)

#                 html_parts.append("""
#                     </div>
# """)  # contrastive-grid 끝

#                 # All samples table
#                 all_samples = contrastive.get('all_samples', [])
#                 if all_samples:
#                     html_parts.append("""
#                     <table class="samples-table">
#                         <thead>
#                             <tr>
#                                 <th>Rank</th>
#                                 <th>Probability</th>
#                                 <th>First Sentence</th>
#                                 <th>Final Answer</th>
#                             </tr>
#                         </thead>
#                         <tbody>
# """)

#                     for i, sample in enumerate(all_samples, 1):
#                         first_sentence = sample.get('first_sentence', '')[:200]
#                         if len(sample.get('first_sentence', '')) > 200:
#                             first_sentence += "..."

#                         final_answer = sample.get('final_answer', 'N/A')
#                         prob = sample.get('answer_probability', 0.0)

#                         html_parts.append(f"""
#                             <tr>
#                                 <td class="sample-rank">#{i}</td>
#                                 <td>{prob:.4f}</td>
#                                 <td>{first_sentence}</td>
#                                 <td><strong>{final_answer}</strong></td>
#                             </tr>
# """)

#                     html_parts.append("""
#                         </tbody>
#                     </table>
# """)

#                 html_parts.append("""
#                 </div>
# """)  # contrastive-section 끝

#                 # PCA Context Vector Results
#                 pca_context = contrastive.get('pca_context')
#                 if pca_context and pca_context.get('results'):
#                     html_parts.append("""
#                 <div class="pca-section">
#                     <div class="pca-header">🧪 PCA Context Vector Results</div>
#                     <p class="pca-description">
#                         Testing the effect of adding PCA-extracted context vector (from positive - negative hidden states)
#                         to decoder hidden states during generation.
#                     </p>
# """)

#                     results = pca_context['results']

#                     # Create comparison table
#                     html_parts.append("""
#                     <table class="pca-table">
#                         <thead>
#                             <tr>
#                                 <th>Context Scale</th>
#                                 <th>Accuracy</th>
#                                 <th>Avg Probability</th>
#                                 <th>Correct / Total</th>
#                             </tr>
#                         </thead>
#                         <tbody>
# """)

#                     # Sort by scale
#                     for scale in sorted(results.keys(), key=lambda x: float(x)):
#                         result = results[scale]
#                         accuracy = result.get('accuracy', 0.0)
#                         avg_prob = result.get('avg_probability', 0.0)
#                         correct = result.get('correct_count', 0)
#                         total = result.get('total_trials', 0)

#                         # Highlight best accuracy
#                         row_class = "best-scale" if accuracy == max(r.get('accuracy', 0) for r in results.values()) else ""

#                         html_parts.append(f"""
#                             <tr class="{row_class}">
#                                 <td><strong>{scale}</strong></td>
#                                 <td>{accuracy:.1%}</td>
#                                 <td>{avg_prob:.4f}</td>
#                                 <td>{correct} / {total}</td>
#                             </tr>
# """)

#                     html_parts.append("""
#                         </tbody>
#                     </table>
# """)

#                     # Add detailed trial results for each scale
#                     correct_answer = contrastive.get('correct_answer', '')

#                     for scale in sorted(results.keys(), key=lambda x: float(x)):
#                         result = results[scale]
#                         generated_answers = result.get('generated_answers', [])
#                         answer_probs = result.get('answer_probabilities', [])

#                         if generated_answers:
#                             # Truncate long correct answers for display
#                             display_correct = correct_answer if len(str(correct_answer)) <= 150 else str(correct_answer)[:150] + "..."

#                             html_parts.append(f"""
#                     <div class="trials-details">
#                         <div class="trials-header">📋 Scale {scale} - Individual Trial Results</div>
#                         <div class="correct-answer-box">
#                             <span class="correct-answer-label">🎯 Correct Answer:</span>
#                             <span class="correct-answer-text">{display_correct}</span>
#                         </div>
# """)

#                             for trial_idx, (answer, prob) in enumerate(zip(generated_answers, answer_probs), 1):
#                                 # Normalize answers for comparison
#                                 def normalize_answer(ans):
#                                     if not ans:
#                                         return ""
#                                     return str(ans).strip().upper()

#                                 is_correct = normalize_answer(answer) == normalize_answer(correct_answer)
#                                 trial_class = "correct" if is_correct else "incorrect"
#                                 icon = "✅" if is_correct else "❌"

#                                 # Truncate long answers
#                                 display_answer = answer if len(str(answer)) <= 100 else str(answer)[:100] + "..."
#                                 if not display_answer:
#                                     display_answer = "(empty answer)"

#                                 html_parts.append(f"""
#                         <div class="trial-item {trial_class}">
#                             <span class="trial-label">{icon} Trial {trial_idx}:</span>
#                             <span class="trial-answer">"{display_answer}"</span>
#                             <span class="trial-prob">(prob: {prob:.4f})</span>
#                         </div>
# """)

#                             html_parts.append("""
#                     </div>
# """)

#                     html_parts.append("""
#                 </div>
# """)  # pca-section 끝

#             html_parts.append("""
#             </div>
# """)  # qa-pair 끝

#         html_parts.append("""
#         </div>
# """)  # example 끝

#     # HTML 마무리
#     html_parts.append("""
#     </div>
# </body>
# </html>
# """)

#     # 파일 저장
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(''.join(html_parts))

#     print(f"✅ HTML report generated: {output_path}")
#     return output_path


# if __name__ == "__main__":
#     output_dir = Path('result_output_10ea')
#     result_files = sorted(output_dir.glob('*.json'))

#     if not result_files:
#         print("❌ No result files found in anchor_vectors_output/")
#     else:
#         print(f"Found {len(result_files)} result files")
#         output_path = generate_html_report(result_files)
#         print(f"\n🎉 Open the report: {Path(output_path).absolute()}")


#!/usr/bin/env python3
"""
Comprehensive evaluation script for Thought Anchor Detection + PCA Context Vector experiments.
"""

import json
import glob
import numpy as np
import argparse
import traceback
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
            qa_pairs = result.get('qa_pairs')
            if not qa_pairs: continue

            contrastive = qa_pairs[0].get('contrastive')
            if not contrastive: continue

            correct_answer = contrastive.get('correct_answer', '')
            
            # [Fix] or {} 패턴으로 None 방지
            pca_context = contrastive.get('pca_context') or {}
            pca_results = pca_context.get('results') or {}

            # Find best scale logic
            best_scale = None
            best_accuracy = -1
            
            for scale, scale_result in pca_results.items():
                # [Fix] scale_result가 None일 경우 건너뜀
                if not scale_result: continue
                
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
                best_res = pca_results[best_scale]
                if best_res:
                    generated_answers = best_res.get('generated_answers', [])
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
            qa_pairs = result.get('qa_pairs')
            if not qa_pairs: continue

            qa_pair = qa_pairs[0]
            anchor_vector = qa_pair.get('anchor_vector')
            contrastive = qa_pair.get('contrastive')
            
            if not anchor_vector or not contrastive: continue

            # Remove last element (final token) if 0
            anchor_scores = anchor_vector[:-1] if (anchor_vector and anchor_vector[-1] == 0.0) else anchor_vector
            if not anchor_scores: continue
            
            max_anchor_score = max(anchor_scores)

            pos_prob = contrastive.get('positive_probability', 0.0)
            neg_prob = contrastive.get('negative_probability', 0.0)
            correlation_data['anchor_vs_prob_delta'].append((max_anchor_score, pos_prob - neg_prob))

            # [Fix] get().get() 체인 안전하게 분해
            pca_context = contrastive.get('pca_context') or {}
            pca_results = pca_context.get('results') or {}
            
            if pca_results:
                # baseline safe get
                baseline_res = pca_results.get('0.0') or {}
                baseline_acc = baseline_res.get('accuracy', 0.0)
                
                # best safe get
                best_acc = 0.0
                for r in pca_results.values():
                    if r and r.get('accuracy', 0.0) > best_acc:
                        best_acc = r.get('accuracy', 0.0)
                
                correlation_data['anchor_vs_pca_improvement'].append((max_anchor_score, best_acc - baseline_acc))

        summary = {}
        for key, data in correlation_data.items():
            if data:
                x, y = zip(*data)
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
            qa_pairs = result.get('qa_pairs')
            if not qa_pairs: continue

            question = qa_pairs[0].get('question', '')
            options_match = re.findall(r'Options?:\s*\n((?:[-•]\s*[^\n]+\n?)+)', question, re.IGNORECASE)
            if not options_match: continue
            
            options = re.findall(r'[-•]\s*([A-Z])\b', options_match[0])
            
            contrastive = qa_pairs[0].get('contrastive') or {}
            pca_context = contrastive.get('pca_context') or {}
            pca_results = pca_context.get('results') or {}
            
            if not pca_results: continue

            # Safe Find Best Scale
            best_scale = None
            max_acc = -1
            for s, r in pca_results.items():
                if r and r.get('accuracy', 0.0) > max_acc:
                    max_acc = r.get('accuracy', 0.0)
                    best_scale = s

            for scale, scale_result in pca_results.items():
                if not scale_result: continue
                
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
            qa_pairs = result.get('qa_pairs')
            if not qa_pairs: continue
            
            contrastive = qa_pairs[0].get('contrastive') or {}
            pca_context = contrastive.get('pca_context') or {}
            pca_results = pca_context.get('results') or {}
            
            for scale, scale_result in pca_results.items():
                if not scale_result: continue
                total = scale_result.get('total_trials', 0)
                efficiency_data['avg_trials_per_scale'][scale].append(total)
                efficiency_data['total_generations'] += total

        summary = {
            'total_generations': efficiency_data['total_generations'],
            'avg_trials_per_scale': {}
        }
        if efficiency_data['avg_trials_per_scale']:
             summary['avg_trials_per_scale'] = {k: np.mean(v) for k, v in efficiency_data['avg_trials_per_scale'].items()}
        
        self.metrics['efficiency'] = summary
        return summary

    # ========================================================================
    # 5. BERTSCORE METRICS
    # ========================================================================

    def compute_bertscore_metrics(self) -> Dict:
        if not HAS_BERTSCORE: return {}
        bertscore_data = {'reasoning_vs_question': {'f1': []}, 'positive_vs_negative': {'f1': []}}

        for result in self.results:
            qa_pairs = result.get('qa_pairs')
            if not qa_pairs: continue
            
            qa = qa_pairs[0]
            question = qa.get('question', '').replace('<|vision_start|>', '').split('\n')[-1]
            reasoning = qa.get('reasoning_text', '').split('<final>')[0].strip()

            contrastive = qa.get('contrastive') or {}
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
        
        acc = self.metrics.get('accuracy', {})
        if acc:
            lines += ["1. ACCURACY", "-"*40]
            lines.append(f"Baseline EM: {acc.get('baseline', {}).get('em', 0):.2%} | F1: {acc.get('baseline', {}).get('f1', 0):.4f}")
            lines.append(f"Best Scale EM: {acc.get('best_scale', {}).get('em', 0):.2%} | F1: {acc.get('best_scale', {}).get('f1', 0):.4f}")
            lines.append("")
        
        corr = self.metrics.get('correlation', {})
        if corr:
            lines += ["2. CORRELATION", "-"*40]
            for k, v in corr.items():
                lines.append(f"{k}: rho={v.get('spearman_rho',0):.4f} (p={v.get('p_value',1):.4f})")
            lines.append("")

        hall = self.metrics.get('hallucination', {})
        if hall:
            lines += ["3. HALLUCINATION", "-"*40]
            lines.append(f"Baseline Rate: {hall.get('baseline', {}).get('hallucination_rate', 0):.2%}")
            lines.append(f"Best Scale Rate: {hall.get('best_scale', {}).get('hallucination_rate', 0):.2%}")
            lines.append("")

        report_text = "\n".join(lines)
        print(report_text)
        with open(output_file, 'w') as f: f.write(report_text)
        print(f"\n✅ Report saved to: {output_file}")

    def run_all(self):
        self.load_results()
        print("Computing Accuracy...")
        self.compute_accuracy_metrics()
        print("Computing Correlation...")
        self.compute_correlation_metrics()
        print("Computing Hallucination...")
        self.compute_hallucination_metrics()
        print("Computing Efficiency...")
        self.compute_efficiency_metrics()
        print("Computing BERTScore...")
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
        # [Fix] 상세한 에러 로그 출력
        traceback.print_exc()
        print(f"\n❌ Evaluation failed: {e}")