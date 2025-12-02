#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 SFT 모델 평가 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적: Fine-tuned LoRA 모델 성능 평가
비교: Zero-shot vs SFT 결과
데이터: KMMLU 전체 45개 과목

출력:
1. JSON: 상세 결과 (my_experiments/kmmlu_ax_4.0_light_sft.json)
2. CSV: 과목별 비교 (my_experiments/kmmlu_sft_comparison.csv)

예상 시간: 30~40분
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datetime import datetime

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 모델 경로
BASE_MODEL = "skt/A.X-4.0-Light"  # 기본 모델
LORA_MODEL = "my_experiments/ax-kmmlu-sft"  # Fine-tuned LoRA

# 출력 파일
OUTPUT_JSON = "my_experiments/kmmlu_ax_4.0_light_sft.json"
OUTPUT_CSV = "my_experiments/kmmlu_sft_comparison.csv"
ZEROSHOT_JSON = "kmmlu_ax_4.0_light_zeroshot.json"  # 비교용

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 모델 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model():
    """
    LoRA Fine-tuned 모델 로드
    
    과정:
    1. 기본 모델 로드 (A.X-4.0-Light)
    2. LoRA 어댑터 추가
    3. GPU로 이동
    
    비유: 교과서 + 별책 조합
    """
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📦 모델 로딩 중...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # 기본 모델 (교과서)
    print(f"1. 기본 모델: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA 어댑터 (별책)
    print(f"2. LoRA 로드: {LORA_MODEL}")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL)
    model = model.merge_and_unload()  # LoRA를 기본 모델에 병합
    
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print("모델 로딩 완료\n")
    return model, tokenizer

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 평가 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_kmmlu(model, tokenizer):
    """
    KMMLU 전체 평가
    
    과정:
    1. 45개 과목 로드
    2. 각 과목별 문제 풀기
    3. 정확도 계산
    
    반환: dict (과목별 결과)
    """
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 KMMLU 평가 시작")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # KMMLU 데이터셋 
    # 45개 과목 리스트
    subjects = [
        'Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance',
        'Biology', 'Chemical-Engineering', 'Chemistry', 'Civil-Engineering',
        'Computer-Science', 'Construction', 'Criminal-Law', 'Ecology', 'Economics',
        'Education', 'Electrical-Engineering', 'Electronics-Engineering',
        'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing',
        'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer',
        'Information-Technology', 'Interior-Architecture-and-Design', 'Law',
        'Machine-Design-and-Manufacturing', 'Management', 'Maritime-Engineering',
        'Marketing', 'Materials-Engineering', 'Mechanical-Engineering',
        'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology',
        'Psychology', 'Public-Safety', 'Railway-and-Automotive-Engineering',
        'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare', 'Taxation',
        'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math'
    ]
    
    print(f"과목 수: {len(subjects)}개")
    print(f"예상 시간: 30~40분\n")
    
    results = {}
    
    # 과목별 평가
    for subject in tqdm(subjects, desc="📝 과목별 평가"):
        # 해당 과목 문제만 필터
        dataset = load_dataset("HAERAE-HUB/KMMLU", subject)
        subject_data = dataset['test']
        
        correct = 0
        total = len(subject_data)
        
        for example in subject_data:
            # 프롬프트 생성
            prompt = format_prompt(example)
            
            # 모델 예측
            prediction = get_prediction(model, tokenizer, prompt)
            
            # 정답 비교
            if prediction == example['answer']:
                correct += 1
        
        # 정확도 계산
        accuracy = correct / total if total > 0 else 0
        results[subject] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy * 100
        }
    
    return results

def format_prompt(example):
    """
    KMMLU 프롬프트 포맷
    
    데이터 구조:
    - question: 문제
    - A, B, C, D: 선택지
    - answer: 정답 (1, 2, 3, 4)
    """
    prompt = f"""다음 문제를 풀고 정답 번호(1, 2, 3, 4)만 답하세요.

문제: {example['question']}
1. {example['A']}
2. {example['B']}
3. {example['C']}
4. {example['D']}


정답:"""
    return prompt

def get_prediction(model, tokenizer, prompt):
    """
    모델 예측
    
    과정:
    1. 프롬프트 토큰화
    2. 모델 생성
    3. 답변 추출 (1, 2, 3, 4)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 생성된 텍스트
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # 답변 추출 (1, 2, 3, 4 중 첫 번째)
    for char in response:
        if char in ['1', '2', '3', '4']:
            return int(char)
    
    return 0  # 답변 없으면 0 (오답 처리)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 결과 비교
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compare_with_zeroshot(sft_results):
    """
    Zero-shot 결과와 비교
    
    출력: CSV (과목별 비교표)
    """
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 Zero-shot 비교")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Zero-shot 결과 로드
    if not os.path.exists(ZEROSHOT_JSON):
        print(f"⚠️  Zero-shot 결과 없음: {ZEROSHOT_JSON}")
        return None
    
    with open(ZEROSHOT_JSON, 'r', encoding='utf-8') as f:
        zeroshot_results = json.load(f)
    
    # 비교 데이터 생성
    comparison = []
    
    for subject in sft_results:
        zs_acc = zeroshot_results.get(subject, {}).get('accuracy', 0)
        sft_acc = sft_results[subject]['accuracy']
        improvement = sft_acc - zs_acc
        
        comparison.append({
            'Subject': subject,
            'Zero-shot (%)': f"{zs_acc:.2f}",
            'SFT (%)': f"{sft_acc:.2f}",
            'Improvement (%p)': f"{improvement:+.2f}",
            'Correct (ZS)': zeroshot_results.get(subject, {}).get('correct', 0),
            'Correct (SFT)': sft_results[subject]['correct'],
            'Total': sft_results[subject]['total']
        })
    
    # DataFrame 생성
    df = pd.DataFrame(comparison)
    
    # 평균 계산
    avg_zs = sum(float(row['Zero-shot (%)']) for row in comparison) / len(comparison)
    avg_sft = sum(float(row['SFT (%)']) for row in comparison) / len(comparison)
    avg_imp = avg_sft - avg_zs
    
    # 평균 추가
    df.loc[len(df)] = {
        'Subject': 'AVERAGE',
        'Zero-shot (%)': f"{avg_zs:.2f}",
        'SFT (%)': f"{avg_sft:.2f}",
        'Improvement (%p)': f"{avg_imp:+.2f}",
        'Correct (ZS)': '-',
        'Correct (SFT)': '-',
        'Total': '-'
    }
    
    # CSV 저장
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 비교표 저장: {OUTPUT_CSV}\n")
    
    return df

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 결과 출력
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_summary(results, df=None):
    """
    결과 요약 출력
    """
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎯 평가 결과 요약")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # 전체 평균
    avg_acc = sum(r['accuracy'] for r in results.values()) / len(results)
    print(f"\n전체 평균 정확도: {avg_acc:.2f}%")
    
    # TOP 5 / BOTTOM 5
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\n🔥 TOP 5 (가장 잘한 과목):")
    for i, (subject, data) in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {subject:30s} {data['accuracy']:6.2f}%")
    
    print("\n⚠️  BOTTOM 5 (취약 과목):")
    for i, (subject, data) in enumerate(sorted_results[-5:], 1):
        print(f"  {i}. {subject:30s} {data['accuracy']:6.2f}%")
    
    # Zero-shot 비교 (있으면)
    if df is not None:
        avg_row = df[df['Subject'] == 'AVERAGE'].iloc[0]
        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📈 Zero-shot 대비 향상")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Zero-shot: {avg_row['Zero-shot (%)']}")
        print(f"SFT:       {avg_row['SFT (%)']}")
        print(f"향상:      {avg_row['Improvement (%p)']}")
        
        # 가장 많이 개선된 과목
        df_sorted = df[df['Subject'] != 'AVERAGE'].sort_values(
            by='Improvement (%p)', 
            ascending=False
        )
        
        print(f"\n🚀 가장 많이 개선된 과목 TOP 5:")
        for i, row in enumerate(df_sorted.head(5).itertuples(), 1):
            print(f"  {i}. {row.Subject:30s} {row._4}")
    
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 메인 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("="*60)
    print("🔍 A.X-4.0-Light SFT 모델 평가")
    print("="*60)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"모델: {LORA_MODEL}")
    print(f"예상 시간: 30~40분")
    print("="*60 + "\n")
    
    # 1. 모델 로드
    model, tokenizer = load_model()
    
    # 2. 평가
    results = evaluate_kmmlu(model, tokenizer)
    
    # 3. JSON 저장
    print(f"\n💾 결과 저장: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 4. Zero-shot 비교
    df = compare_with_zeroshot(results)
    
    # 5. 결과 출력
    print_summary(results, df)
    
    print(f"\n완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
