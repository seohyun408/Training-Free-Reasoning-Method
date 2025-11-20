# Training-Free-Reasoning-Method

> srun  -p suma_A6000 --gres=gpu:1 --time=1:00:00  --pty bash -i

### ✅ Team Members
| Name               | GitHub / Contact         |
| ------------------ | ------------------------ |
| **Jiyoung Ko**     | [@rhwldud0913](#)        |
| **Seohyun Lee**    | [@seohyun408](#)         |
| **Seungyoun Lee**  | [@win2dvp21](#)          |
| **Soojeong Lee**   | [@LeeSooJeong1124](#)    |


### ✅ Process (Step-by-Step)

1. **Thought Anchor Detection**: Identify critical reasoning sentences using KL divergence with attention masking
2. **Contrastive Generation**: Extract positive (high-prob, correct) vs negative (low-prob) reasoning paths
3. **PCA Context Vector**: Compute steering direction from (positive - negative) hidden states
4. **Latent Space Steering**: Add scaled context vector to decoder hidden states during generation

---

## 🚀 How to Run

### Step 1: Run Main Analysis

```bash
python main.py \
  --data llava-cot-100k \
  --num-examples 50 \
  --test-pca-context \
  --pca-num-trials 5
```

**Outputs**: `anchor_vectors_output/example_*.json`

### Step 2: Generate HTML Reports

```bash
python generate_report.py
```




