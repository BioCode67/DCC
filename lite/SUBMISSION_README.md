# 팀 200OK — 2026 DCC 대학부 예선 제출물

## 구성
- `mission1/` 신고자 성별 분류 / `mission2/` 신고자·119대원 발화 분류 / `mission3/` 증상 인식
  - `inference.py` (채점용), `train.py` (학습), `train_missionN.ipynb` (학습 로그), `missionN_model.pkl` (학습된 모델), `requirements.txt`
  - `outputs/missionN.csv` : Validation 추론 결과
- `200OK_결과정리.pptx` : 미션 1·2·3 결과 정리

## 실행 (각 미션 폴더에서)
```bash
pip install -r requirements.txt
python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path missionN_model.pkl --output ./outputs/missionN.csv
```
- CPU 만으로 동작, 인터넷 불필요 (모델은 scikit-learn, joblib 으로 저장된 .pkl)
- Mission 3 은 `--audio_dir` 을 사용하지 않음 (텍스트 전용)
- Decision threshold 0.5 고정
