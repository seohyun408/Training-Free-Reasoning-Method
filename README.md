# Training-Free-Reasoning-Method

> srun  -p suma_A6000 --gres=gpu:1 --time=1:00:00  --pty bash -i

### ✅ Team Members
| Name               | GitHub / Contact         |
| ------------------ | ------------------------ |
| **Jiyoung Ko**     | [@rhwldud0913](#)        |
| **Seohyun Lee**    | [@seohyun408](#)         |
| **Seungyoun Lee**  | [@win2dvp21](#)          |
| **Soojeong Lee**   | [@LeeSooJeong1124](#)    |


### ✅ Folder
```
.
├── README.md
├── main.py                         
├── process.py                      
├── whitebox-analyses/              
    ├── calculate_anchor.py         # Anchor Detection (Stage1)
    └── attention_analysis/
        └── attn_supp_funcs.py      # KL divergence 
├── contrastive_generation.py       # Contrastive Generation (Stage2)
├── utils.py                        

```


### ✅ Process (Step-by-Step)

1. **Thought Anchor Detection**: Identify critical reasoning sentences using KL divergence with attention masking
2. **Contrastive Generation**: Extract positive (high-prob, correct) vs negative (low-prob) reasoning paths
3. **PCA Context Vector**: Compute steering direction from (positive - negative) hidden states
4. **Latent Space Steering**: Add scaled context vector to decoder hidden states during generation

---

## 🚀 How to Run

### Step 0. Environment Setting

```bash
export HUGGINGFACE_TOKEN=       
export HF_HOME=       
export HF_MODEL_CACHE=      
export HF_DATASETS_CACHE=         
```

### Step 1: Run Main.py 

```bash
python main.py \
  --data llava-cot-100k \
  --num-examples 50
```

**This runs the full pipeline**: Thought Anchor Detection → Contrastive Generation → **PCA Context Vector Testing**

**Default settings**:
- ✅ Stochastic generation (temperature=0.8, different results each run)
- ✅ PCA context vector testing (5 trials per scale)

**Outputs**: `anchor_vectors_output/example_*.json`

#### Optional: Disable sampling (deterministic, reproducible results)

```bash
python main.py \
  --data llava-cot-100k \
  --num-examples 50 \
  --no-sampling
```

#### Optional: Disable PCA (faster, but no steering results)

```bash
python main.py \
  --data llava-cot-100k \
  --num-examples 50 \
  --no-pca
```

#### Optional: Add PCA to existing results

If you already have results without PCA:
```bash
python run_pca_on_existing.py
```

### Step 2: Generate HTML Reports

```bash
python generate_report.py
```

**Outputs**: `thought_anchor_report.html` (interactive results with trial details)

### Step 3: Generate Methodology Documentation

```bash
python generate_methodology.py
```

**Outputs**: `methodology_report.html` (complete method explanation)

---

## 📊 Results Summary

- **60% accuracy improvement** on diagram reasoning (40% → 100%)
- **Training-free**: No fine-tuning required
- **Non-linear steering**: Optimal at scale 0.5-1.0, over-steering at 2.0, force mode at 5.0