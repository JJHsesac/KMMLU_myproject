# KMMLU Fine-tuning Project  
**Strategic Sampling + LoRA (PEFT)**  
Model: **SKT/A.X.4.0-Light**

This repository contains a full fine-tuning workflow I personally designed and executed to improve the KMMLU benchmark performance using:

- Zero-shot baseline evaluation  
- Weak-subject strategic sampling  
- Alpaca-format SFT dataset creation  
- LoRA fine-tuning (PEFT)  
- Full benchmark evaluation & analysis  

All experiments were implemented from scratch.

---

# 1. Overview

![Project Overview](images/project_overview.png)

This project explores whether a small, strategically selected SFT dataset can improve model reasoning abilities across 45 KMMLU subjects.

---

# 2. Repository Structure
📁 KMMLU_myproject/
│
├── prepare_sft_data_strategic.py
├── finetune_lora_peft.py
├── evaluate_sft_model.py
├── kmmlu_ax_4.0_light_zeroshot.py
├── kmmlu_ax_4.0_light_zeroshot_cot.py
└── qwen_zero_shot.py

---

# 3. Experiment Design

## ✔ Model  
- **Base Model:** SKT/A.X.4.0-Light  
- **Method:** LoRA (PEFT)  
- **Trainable Params:** ~40M (0.55%)  
- **Epochs:** 3  
- **Learning Rate:** 2e-4  

## ✔ Pipeline Diagram  

![Pipeline Flow](images/pipeline_flow.png)

**End-to-end process:**

1. Zero-shot benchmark  
2. Weak subjects detected  
3. 70% weak-subject sampling  
4. 30% uniform sampling  
5. Alpaca-format SFT dataset  
6. LoRA training  
7. KMMLU evaluation & analysis  

---

# 4. Strategic Sampling

## Zero-shot Weak Subject Detection  

![Weak Subject Chart](images/weak_subject_chart.png)

The baseline accuracy was:

| Metric | Score |
|--------|--------|
| **Zero-shot** | **56.25%** |

Lowest performing subjects included:  
Math (28%), Korean-History (37%), Engineering (41%–50%), Taxation, Criminal-Law, etc.

---

## Sampling Strategy

![Sampling Strategy](images/sampling_strategy.png)

|**Group** | **Ratio** | **Description** |
|-------|--------|-------------|
| Weak subjects | **70% (350 samples)** | Based on error count |
| All subjects | **30% (150 samples)** | ~3 per subject to avoid catastrophic forgetting |

### Result  
**Total SFT dataset size:** 500 samples  
**Format:** Alpaca instruction-following  
**File:** `kmmlu_sft_strategic_500.jsonl`

---

# 5. Alpaca SFT Dataset Structure

![Alpaca Structure](images/alpaca_structure.png)

Each sample contains:

```json
{
  "instruction": "Solve the following KMMLU problem step-by-step.",
  "input": "Question: ...",
  "output": "Thought process + final answer"
}
---
# 6. LoRA Fine-tuning

| **Parameter**    | **Value**    |
| ---------------- | ------------ |
| LoRA r           | 16           |
| Epochs           | 3            |
| LR               | 2e-4         |
| Trainable Params | 0.55% (~40M) |

## Training Curve
Loss: 2.45 → 0.30
Healthy gradients
No overfitting observed
---
# 7. Evaluation (KMMLU Benchmark)

Script: evaluate_sft_model.py

**Performance**
Model	Accuracy
Zero-shot	            56.25%
LoRA SFT (Strategic)	57.58%

Gain: +1.33%p
---
Improvements by Subject Group

Category	      Gain
Korean-History	+8%p
HUMSS	          +3.2%p
Other         	+8.78%p
Math	          +0.33%p

---
# 8. Analysis of Results
✔ What worked
- **Strategic sampling is beneficial for humanities subjects**
- **LoRA training converged stably**
- **Coverage across all 45 subsets prevented forgetting**

✖ What limited performance
- **Small SFT dataset (500 samples)**
- **Korean-history train set contained only 1 training item**
- **Math requires domain-specific CoT templates**
- **CoT styles between SFT and KMMLU differed**
---
9. Reproducibility
# 1. Zero-shot evaluation
python kmmlu_ax_4.0_light_zeroshot.py

# 2. SFT dataset creation
python prepare_sft_data_strategic.py

# 3. LoRA training
python finetune_lora_peft.py

# 4. Evaluation
python evaluate_sft_model.py
---
# 10. Future Work
- Expand SFT dataset to 1,000–2,000
- Math-specific templates
- Evaluate CoT-free SFT
- Few-shot evaluation setting
- Q-LoRA for larger batch sizes
