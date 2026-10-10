# 2026 데이터·AI 크리에이터 캠프 (DCC) 대학부 예선 — 팀 200OK

**문제**: 전화대화 음성 및 전사 데이터를 이용한 응급상황 인식
(AI-Hub「위급상황 음성/음향(고도화) - 119 지능형 신고접수 음성 인식 데이터」, 서울/구급)

| 미션 | 입력 | 출력 | 평가 | 방법 |
|---|---|---|---|---|
| 1. 신고자 성별 | 음성 + startAt/endAt/speaker | gender | Accuracy | 신고자(speaker=1) 구간 최대 30초 → 고정 WavLM 특징 → MLP |
| 2. 신고자/119대원 | 발화 조각 음성 (startAt~endAt) | speaker (1/0) | Accuracy | 조각(최대 8초) → 고정 WavLM 특징 → MLP |
| 3. 환자 증상 | 대화 전사 텍스트 | symptom (9개 중 0~9개) | Macro F1 | KLUE-RoBERTa fine-tuning + window max-pooling |

- 사전학습 모델: `microsoft/wavlm-base-plus` (미션 1·2, 가중치 고정), `klue/roberta-base` (미션 3)
- Threshold: 모든 미션 0.5 고정 (2-class softmax argmax = p≥0.5, 증상별 sigmoid ≥ 0.5) — 규정 준수
- Validation 폴더는 학습·모델 선택에 사용하지 않음 (Training 내부 10% holdout 으로 epoch 선택)

## 실행 순서 (Colab Pro, GPU: L4 권장)

0. 공유받은 `대학부 데이터` 폴더를 **내 드라이브에 바로가기 추가** (용량 차지 안 함, 현재 `2026 공모전/데이터+AI 크리에이터 캠프/대학부 데이터`)
1. 각 미션 폴더의 노트북을 Colab 으로 열기 (GitHub → `BioCode67/DCC`, 브랜치 `ccr-1cdc76be-ibbw6s`)
   - `mission1/train_mission1.ipynb`, `mission2/train_mission2.ipynb`, `mission3/train_mission3.ipynb`
2. 런타임 → 런타임 유형 변경 → GPU(L4) + 높은 RAM
3. `QUICK_TEST = 200` 으로 한 번 끝까지 실행해서 경로/형식 확인 → `QUICK_TEST = 0` 으로 전체 실행
4. 체크포인트와 Validation 결과 CSV 는 `내 드라이브/DCC_outputs/` 에 저장됨
5. 실행 로그가 남은 노트북을 `파일 → 다운로드 → .ipynb` 로 받아 각 미션 폴더에 넣기 (제출용 학습 로그)

## 추론 (채점 명령어)

```bash
cd mission{N}
python inference.py --audio_dir {wav folder} --label_dir {json folder} \
                    --ckpt_path {checkpoint file} --output ./outputs/mission{N}.csv
```

- 출력 CSV: 미션1 `audio file name, gender` / 미션2 `audio file name, startAt, endAt, speaker` /
  미션3 `label file name, symptom` (예: `['두통', '복통']`)
- 체크포인트 하나에 사전학습 모델 설정·가중치까지 포함 → 채점 시 인터넷 불필요
  (미션3 은 체크포인트와 같은 폴더의 `tokenizer/` 도 함께 제출)
- 채점: `python tools/evaluate.py --mission N --pred outputs/missionN.csv --label_dir {json folder}`

## 제출 구조 (`200OK.zip`, 마감 10/14(수) 17:00)

```
200OK/
├── mission1/  inference.py, train.py, audio_model.py, requirements.txt, train_mission1.ipynb(로그 포함), mission1.pt
├── mission2/  inference.py, train.py, audio_model.py, requirements.txt, train_mission2.ipynb(로그 포함), mission2.pt
├── mission3/  inference.py, train.py, text_model.py, requirements.txt, train_mission3.ipynb(로그 포함), mission3.pt, tokenizer/
└── 200OK_결과정리.pptx  (미션 1·2·3 단일 파일)
```

## 규정 체크리스트

- [x] AI-Hub 제공 데이터만 사용, Training 은 서울 데이터만, Validation 은 학습 미사용
- [x] 미션별 지정 입력만 사용 (M1: 음성+startAt/endAt/speaker, M2: 조각 경계만·순서 미사용, M3: 전사 텍스트만)
- [x] label 별로 다른 전처리 없음, 평가 데이터 전용 전처리 없음
- [x] 상용 AI API 미사용 (공개 사전학습 모델만 사용)
- [x] Decision threshold 0.5 고정 (클래스 공통)
- [x] `inference.py` 1회 실행으로 결과 CSV 생성
