# CPU 경량 제출 버전 (마감용)

GPU 없이 PC(CPU)에서 30분~1시간 안에 미션 1·2·3 학습 → Validation 채점 → PPT → `200OK.zip` 까지 한 번에 생성.

```powershell
pip install numpy scikit-learn librosa soundfile joblib python-pptx nbformat nbclient ipykernel
python run_all.py --data_root "C:\경로\대학부 데이터"
```

- 결과: `lite/submission/200OK.zip` (웹하드 [대학부] 폴더에 업로드), 로그: `lite/submission/run_all_log.txt`
- `--m1_limit 3000 --m2_limit 2000` : 미션 1·2 학습 통화 수 (시간 남으면 늘리기, 0=전체)
- 중간에 멈춰도 `--missions 1,2` 처럼 남은 미션만 다시 실행 가능 (결과는 results.json 에 누적)
