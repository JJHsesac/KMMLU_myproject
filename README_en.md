# KMMLU Fine-tuning Project  
**Strategic Sampling + LoRA (PEFT)**  
Model: **SKT/A.X.4.0-Light**

This repository documents a personal experiment exploring how to improve KMMLU performance using:

- Zero-shot baseline evaluation  
- Weak-subject strategic sampling  
- Alpaca-format SFT dataset creation  
- LoRA fine-tuning (PEFT)  
- Full KMMLU benchmark evaluation  

All experiments were independently designed and implemented.

---

#  1. Overview

This project evaluates SKT’s open LLM **A.X 4.0-Light**  using the **KMMLU Light benchmark (44 subjects)**  
to analyze performance changes across Zero-shot → CoT → Few-shot → SFT.

---
```
KMMLU_myproject/
│
├── README_KR.md             # Korean version
├── README_en.md          # English version
│
├── data/
│   └── kmmlu_sft_strategic_500.jsonl      # result
│
├── scripts/
│   ├── prepare_sft_data_strategic.py      # Strategic SFT data creation
│   ├── finetune_lora_peft.py              # PEFT based fine-tuning (not LoRA)
│   ├── evaluate_sft_model.py              # Evaluation
│   ├── kmmlu_ax_4.0_light_zeroshot.py     # Zero-shot
│   ├── kmmlu_ax_4.0_light_zeroshot_cot.py # CoT
│   └── qwen_zero_shot.py                  # Model comparison (Qwen2.5-7B)
│
└── results/
    └── README_results.md                  # Wrapup Report 
```
---

# 3. Experiment Design

## ✔ Model  
- **Base Model:** SKT/A.X.4.0-Light  
- **Method:** LoRA (PEFT)  
- **Trainable Params:** ~40M (0.55%)  
- **Epochs:** 3  
- **Learning Rate:** 2e-4  

---

# Pipeline Diagram

---

# 4. Strategic Sampling

## Zero-shot Weak Subject Detection

The baseline zero-shot accuracy was:

| Metric    | Score      |
|-----------|------------|
| Zero-shot | **56.25%** |

Weakest subjects included Math (28%), Korean-History (37%), Engineering (~41–50%), Taxation, and Criminal-Law.

---

## Sampling Strategy

```
| Group         | Ratio             | Description                          |
|---------------|-------------------|--------------------------------------|
| Weak subjects | 70% (350 samples) | Based on error frequency             |
| All subjects  | 30% (150 samples) | ~3 per subject to prevent forgetting |
```
### Dataset Summary  
- **Total samples:** 500  
- **Format:** Alpaca (instruction / input / output)  
- **File:** `kmmlu_sft_strategic_500.jsonl`

---

# 5. Alpaca SFT Dataset Structure

Each SFT item follows:

instruction: "Solve the following KMMLU problem step-by-step."
input: "Question + Options"
output: "Chain-of-thought + final answer"

---

# 6. LoRA Fine-tuning

**Training Parameters**
```
| Parameter        | Value        |
|------------------|--------------|
| LoRA r           | 16           |
| Epochs           | 3            |
| Learning Rate    | 2e-4         |
| Trainable Params | 0.55% (~40M) |
```
### Training Curve


- Loss improved from **2.45 → 0.30**  
- Stable gradients  
- No overfitting observed  

---

# 7. Evaluation (KMMLU Benchmark)

Executed using: **evaluate_sft_model.py**
```
| Model                | Accuracy   |
|----------------------|------------|
| Zero-shot            | 56.25%     |
| LoRA SFT (Strategic) | **57.58%** |
```
**Overall Gain:** +1.33%p

---

# Subject-level Improvements

```
| Category       | Gain     |
|----------------|----------|
| Korean-History | **+8%p** |
| HUMSS          | +3.21%p  |
| Other          | +8.78%p  |
| Math           | +0.33%p  |
```
---

# 8. Analysis

### ✔ What worked
- Weak-subject sampling improved humanities categories  
- LoRA training converged cleanly  
- Coverage across all 45 subjects prevented catastrophic forgetting  

### ✖ Limitations
- Only 500 SFT samples (too small for 7B models)  
- Some subjects had very few training examples (e.g., Korean-History: 1 item)  
- CoT template mismatch across domains  
- Math improved minimally due to limited domain-specific reasoning  

---

# 9. Reproducibility

To reproduce results:

1. Zero-shot evaluation
python kmmlu_ax_4.0_light_zeroshot.py

2. Create SFT dataset
python prepare_sft_data_strategic.py

3. Fine-tune using LoRA
python finetune_lora_peft.py

4. Evaluate fine-tuned model
python evaluate_sft_model.py

---

# 10. Future Work

- Expand SFT dataset to 1k–2k samples  
- Develop Math-specific CoT templates  
- Evaluate CoT-free SFT  
- Use few-shot prompts during evaluation  
- Consider Q-LoRA for faster training at larger batch sizes  

---

# Contact  

Feel free to open an issue for further questions.
