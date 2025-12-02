# 1. Project Overview

This project evaluates SKT’s open-source LLM **A.X 4.0-Light** using **KMMLU Light (44 subjects)** to measure performance changes across:

- Zero-shot
- Zero-shot CoT
- 5-shot
- Strategic Supervised Fine-Tuning (SFT, PEFT-based)

During the experiments, Korean History and Mathematics were identified as the **two weakest subjects**, significantly lowering the overall average score.

Therefore, this project applies **strategic SFT on the bottom 30% subjects**, and additionally analyzes the Korean History dev/test patterns.  
The insights from this analysis were later extended to the design of a personal **Korean History RAG-based learning assistant**.

The main purpose of the strategic SFT experiment is to examine whether:

- **With extremely limited data**,  
- **Targeted fine-tuning on weak subjects**  
can lead to observable performance improvements.

---

# 2. Repository Structure
---
```
KMMLU_myproject/
│
├── README.md             # Korean version
├── README_en.md          # English version
│
├── data/
│   └── kmmlu_sft_strategic_500.jsonl      # Strategic SFT dataset (500 samples)
│
├── scripts/
│   ├── prepare_sft_data_strategic.py      # Strategic SFT dataset generation
│   ├── finetune_lora_peft.py              # PEFT-based fine-tuning (not LoRA)
│   ├── evaluate_sft_model.py              # KMMLU Light evaluation
│   ├── kmmlu_ax_4.0_light_zeroshot.py     # Zero-shot evaluation
│   ├── kmmlu_ax_4.0_light_zeroshot_cot.py # Zero-shot + CoT evaluation
│   └── qwen_zero_shot.py                  # Baseline comparison (Qwen2.5-7B-Instruct)
│
└── results/
    └── README_results.md                  # Text-based result summary
```
---

# 3. Objectives
The project aims to:

✔ 1) Analyze performance on KMMLU Light
- Evaluate SKT A.X 4.0-Light on 44 KMMLU subjects to measure performance across Zero-shot → CoT → 5-shot → SFT.

✔ 2) Verify whether strategic SFT improves weak subjects
Instead of fine-tuning on all subjects, only the bottom 30% subjects are selected to investigate whether low-resource targeted SFT yields meaningful gains.

✔ 3) Build a reproducible KMMLU experiment pipeline
- Generate evaluation files (JSON/CSV)
- Set up PEFT-based low-cost training
- Document a reusable pipeline

# 4. Dataset & Benchmark
📌 Benchmark Dataset
- KMMLU Light  44 subjects, ~45 questions each
- Korean multi-task benchmark

📌 Evaluated Model (LLM)
- SKT/A.X 4.0-Light
- Initial baseline comparisons:
  Llama-3.2-Korean-Bllossom-3B
  Qwen2.5-7B-Instruct

- A.X 4.0-Light achieved the strongest Zero-shot baseline → selected for full evaluation

📌 Target Subjects
- Focus on low-performing subjects, especially:  Korean History  Mathematics

📌 SFT Data Conditions
- dev split contains only 1–3 samples per subject

- Korean History dev contains only 1 sample
→ Ideal environment for testing SFT under extreme low-resource conditions

# 5. Experiment Pipeline
**Zero-shot / Zero-shot CoT**
- kmmlu_ax_4.0_light_zeroshot.py
- kmmlu_ax_4.0_light_zeroshot_cot.py
**Strategic SFT Data Preparation**
- prepare_sft_data_strategic.py
- Select bottom 30% subjects → convert to Alpaca-style JSONL (500 samples)

**PEFT-based SFT Training**
- finetune_lora_peft.py
- (LoRA was not applicable → replaced with PEFT fine-tuning) r=16 / 3 epochs / LR=2e-4
- Trainable parameters ≈ 40M (0.55%)

**Post-SFT Evaluation**
- evaluate_sft_model.py
- Evaluate SFT model vs. Zero-shot baselines

# 6. Main Results
**1) Zero-shot**
- Moderate performance across subjects
- Korean History extremely low (~35–40%)

**2) Zero-shot + CoT**
- Improvements in reasoning-heavy subjects
- Korean History shows degradation due to hallucination

**3) 5-shot**
More stable improvements
Limited by small number of available few-shot samples

**4) Strategic SFT (Low-Resource)**
With Korean History dev = 1 sample
→ Extremely high risk of overfitting
→ Minimal performance improvement on test

**Conclusion**
SFT with 1–3 samples per subject rarely produces generalizable gains.

**7. Conclusions & Limitations**
✔ Strategic SFT is reasonable
Targeting weak subjects is a valid research direction.

✔ But KMMLU Light has extremely limited data
dev Korean History: 1 sample

test Korean History: 106 samples
→ Strong mismatch and overfitting risk

✔ Improvement was minimal
Training becomes memorization rather than generalization.

# 8. Future Directions
1) Expand to KMMLU Full
Larger datasets enable more meaningful SFT.

2) Integrate with Korean History RAG
Concepts → RAG

Problem solving → SFT → Build a hybrid Korean History tutor

3) Data Augmentation
- For low-resource subjects:
- Paraphrasing
- Distractor generation
- Synthetic Q&A creation

# 9. Contribution
This repository is a personal research project intended to serve as a reference for beginners working with KMMLU-based evaluations.

10. License
KMMLU Light data: Follow SKT open license

Code: MIT License
