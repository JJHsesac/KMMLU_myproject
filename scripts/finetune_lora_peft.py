#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 A.X-4.0-Light LoRA Fine-tuning (PEFT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적: KMMLU 오답 데이터로 A.X 모델 Fine-tuning
방법: LoRA (Low-Rank Adaptation) - 0.1%만 학습
도구: PEFT (5배 빠른 LoRA)
예상 시간: 2시간 (batch_size=4)

작동 원리:
  1. A.X-4.0-Light 기본 모델 로드
  2. LoRA 레이어 추가 (r=16, 7M 파라미터)
  3. 오답 500개로 학습 (3 epochs)
  4. LoRA만 저장 (7MB)

비유: 
  학생(모델)에게 별책(LoRA) 주고 오답노트로 3번 공부시키기

근거:
  - LoRA 원논문 (Microsoft, 2021): r=16 권장
  - LLaMA-2 (Meta, 2023): epochs=3, lr=2e-4
  - Unsloth (2024): 5배 속도 향상

예상 효과:
  56.25% → 58~59% (+2~3%p)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚙️ 하이퍼파라미터 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 근거: LLaMA-2 논문 + LoRA 원논문 + 실험 최적화
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 LangSmith 연결 (선택사항)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "KMMLU-skt-AX-4.0-light-SFT"

MAX_SEQ_LENGTH = 2048           # 최대 토큰 길이 (KMMLU 문제 평균 ~500)
BATCH_SIZE = 1                  # 배치 크기 (V100 16GB 최적화, 1배 빠름!) 4로 했다 메모리부족
GRADIENT_ACCUMULATION = 4       # 그래디언트 누적 (메모리 부족 시 2로)
LEARNING_RATE = 2e-4            # 학습률 (LoRA 표준값)
NUM_EPOCHS = 3                  # 에폭 수 (3번 반복, 과적합 방지)
OUTPUT_DIR = "my_experiments/ax-kmmlu-sft"  # 저장 경로

def main():
    print("="*60)
    print("A.X-4.0-Light LoRA Fine-tuning (PEFT)")
    print("="*60)
    print(f"설정: Batch={BATCH_SIZE}, LoRA r=16, Epochs={NUM_EPOCHS}")
    print(f"예상 시간: 3시간")
    print("="*60 + "\n")
    
    # 4bit 양자화
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    # 모델 로드
    print("모델 로딩 중...")
    model = AutoModelForCausalLM.from_pretrained(
        "skt/A.X-4.0-Light",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("skt/A.X-4.0-Light", trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("모델 로딩 완료\n")
    
    # LoRA
    print("LoRA 추가 중...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.print_trainable_parameters()
    print("LoRA 완료\n")
    
    
    # 데이터
    print("데이터 로딩...")
    dataset = load_dataset(
        "json",
        data_files="./my_experiments/kmmlu_sft_strategic_500.jsonl",
        split="train"
    )
    print(f"데이터: {len(dataset)}개\n")
    
    # 포맷
    def formatting_prompts_func(examples):
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            text = f"""### Instruction:
{inst}

### Input:
{inp}

### Response:
{out}"""
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=50,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="adamw_torch",
        report_to="none",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    # 학습
    print("\n" + "="*60)
    print(" Fine-tuning 시작!")
    print("="*60 + "\n")
    
    trainer.train()
    
    # 저장
    print("\n 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("완료!")
    print(f" 위치: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
