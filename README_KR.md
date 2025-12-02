**1. 프로젝트 개요**

본 프로젝트는 SKT의 공개 LLM인 A.X 4.0-Light 모델을 대상으로
KMMLU Light(총 44개 과목) 평가셋을 활용하여
Zero-shot → CoT → Few-shot → 전략적 SFT 순으로 성능 변화를 측정한 실험입니다.

본 실험 과정에서 한국사(Korean History)와 수학(Math)이
전체 평균을 크게 떨어뜨리는 주요 취약 과목임을 확인하였고,
이에 따라 정답률 하위 30% 과목만 선별하여 SFT를 수행했습니다.

본 전략적 SFT는

데이터가 매우 적은 상황에서도,

취약 과목을 집중 재학습하면 성능 향상 효과가 있는지
를 검증하는 데 목적이 있습니다.

또한 한국사 dev/test 문항 분석 과정에서
문항 유형·핵심 개념 패턴을 파악할 수 있었고,
이 분석 결과는 후속 개인 프로젝트인
“한국사 RAG 기반 개인 학습 튜터” 설계로 자연스럽게 확장되었습니다.

**2. 저장소 구조**
---
```
KMMLU_myproject/
│
├── README.md             # 한국어
├── README_en.md          # 영어
│
├── data/
│   └── kmmlu_sft_strategic_500.jsonl      # 전략적 SFT 결과 (500 샘플)
│
├── scripts/
│   ├── prepare_sft_data_strategic.py      # 전략적 SFT 데이터 생성
│   ├── finetune_lora_peft.py              # PEFT 기반 파인튜닝 코드 (LoRA 아님)
│   ├── evaluate_sft_model.py              # KMMLU Light 평가
│   ├── kmmlu_ax_4.0_light_zeroshot.py     # Zero-shot 평가
│   ├── kmmlu_ax_4.0_light_zeroshot_cot.py # Zero-shot + CoT 평가
│   └── qwen_zero_shot.py                  # Qwen2.5-7B-Instruct 비교 실험
│
└── results/
    └── KMMLU_Wrapup_Report_KR_EN.md       # 최종 결과 요약 
```
---
**3. 실험 목표**

본 프로젝트의 핵심 목표는 다음과 같습니다:

✔ 1) KMMLU Light 기반 성능 분석

SKT A.X 4.0-Light 모델을 **KMMLU Light(44개 과목)**로 평가하여
Zero-shot → CoT → Few-shot → SFT 순으로 성능 변화를 측정.

✔ 2) 전략적 SFT가 실제 성능 향상으로 이어지는지 검증

정답률 하위 30% 과목만 선별하여 SFT를 수행하여
적은 데이터에서도 취약 과목 성능이 개선될 수 있는지 탐구.

✔ 3) 반복 실험 가능한 개인 연구 환경 구축

JSON/CSV 기반 평가 파일 자동 생성

PEFT 기반 Low-cost 파인튜닝 구조 구축

전체 파이프라인 정리로 재현성 확보

**4. 데이터 및 벤치마크**

📌 사용한 벤치마크 데이터셋
KMMLU Light (총 45개 문항 × 44개 과목)
한국어 멀티태스크 평가셋

📌 사용한 모델 (LLM)
SKT A.X 4.0-Light
초기 비교 모델:
Llama-3.2-Korean-Bllossom-3B
Qwen2.5-7B-Instruct
Zero-shot 기준 A.X 4.0-Light가 가장 우수하여 주 모델로 채택

📌 대상 과목
Zero-shot 기준 성능 하위 그룹 중심
특히 Korean History(한국사)와 Math(수학)에서 큰 취약성 확인

📌 SFT 데이터 구성
dev split 기준 각 과목 1~3개 수준(매우 희소)
한국사 dev는 단 1개 존재
→ 데이터 희소 상황에서의 SFT 효과 검증에 적합

**5. 실험 과정 요약**
**Zero-shot / Zero-shot CoT**
- kmmlu_ax_4.0_light_zeroshot.py
- kmmlu_ax_4.0_light_zeroshot_cot.py
**전략적 SFT 데이터 생성**
- prepare_sft_data_strategic.py
- 성능 낮은 과목만 선별 → Alpaca 형식으로 변환 → JSONL 생성
**PEFT 기반 SFT 파인튜닝**
- finetune_lora_peft.py
- (LoRA 적용 불가 상황 → PEFT로 대체)  r=16 / epoch=3 / LR=2e-4
- Trainable params ≈ 40M (0.55%)

**SFT 모델 평가**

- evaluate_sft_model.py
- SFT 적용 후 Zero-shot과 비교 평가

**6. 주요 실험 결과**
**1) Zero-shot 성능**

대부분 과목에서 중간 수준
한국사 약 35~40%로 매우 낮음

**2) Zero-shot + CoT**

일부 논리 기반 과목에서 개선
한국사는 오히려 CoT로 환각 증가 → 성능 저하 발생

**3) 5-shot**

대체로 안정적 개선
데이터 부족으로 개선 폭은 제한적

**4) 전략적 SFT (Low Resource)**

한국사 dev가 1개뿐 → 과적합 위험 매우 높음
실제 test 기준에서 성능 향상은 미미함

**결론**

데이터가 희소한 과목(SFT 샘플 1개 수준)은 성능 개선이 어렵다.

**7. 결론 및 한계점**]

✔ 전략적 SFT 방법론 자체는 타당
취약 과목 중심 집중 학습은 여러 연구에서도 검증된 접근.

✔ KMMLU Light의 데이터가 지나치게 작음
dev Korean History: 1문항
test Korean History: 106문항
→ train–test 분포 불일치 및 과적합 우려

✔ 이번 실험에서의 향상 폭은 제한적
SFT가 train 스플릿의 극소 데이터를 암기하는 수준
일반화 성능은 크게 증가하지 않음

**8. 앞으로의 개선 방향**
1) KMMLU Full / Hard / Plus로 확장

- 데이터가 많아지면 SFT 효과가 훨씬 명확해짐.

2) 한국사 RAG 시스템과 결합

- 개념 학습 → RAG
- 문제 풀이 → SFT
→ 하이브리드 한국사 튜터로 발전 가능

3) 데이터 증강(Augmentation)
- 패러프레이징
- synthetic QA generation
→ 한국사/수학 같은 희소 과목에 효과적

**9. 기여**

본 저장소는 개인 연구 프로젝트이며
KMMLU 실험을 처음 시도하는 사람에게 참고가 되는 것을 목표로 합니다.

**10. 라이선스**

KMMLU Light 데이터: SKT 공개 라이선스 준수
