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
│   ├── calculate_anchor.py
│   └── attention_analysis/
│       └── attn_supp_funcs.py
├── method/
│   ├── context_vector.py
│   ├── contrastive_generation.py
│   └── reasoning_generation.py
├── utils.py
└── view_outputs_results.ipynb

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
- ✅ PCA context vector testing (scale=[0.0, 1.0])

**Outputs**: `outputs/example_*.json`


### Step 2: Generate HTML Reports (여긴 수정안함)

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
