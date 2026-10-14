"""
Mission 3 추론: 대화 전사 텍스트 → 환자 증상

python inference.py --audio_dir {wav folder} --label_dir {json folder} \
                    --ckpt_path {checkpoint file} --output ./outputs/mission3.csv

출력 CSV: label file name, symptom   (예: ['두통', '복통'], 없으면 [])
라벨 json 에서는 전사 텍스트만 사용 (--audio_dir 은 공통 인터페이스용, 사용 안 함)
"""
import argparse
import csv
import time
from pathlib import Path

import joblib

from train import SYMPTOMS, dialogue_text, load_dir, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", default=None)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output", default="./outputs/mission3.csv")
    a = p.parse_args()

    ck = joblib.load(a.ckpt_path)
    t0 = time.time()
    paths, labs = load_dir(a.label_dir)
    P = predict(ck, [dialogue_text(l) for l in labs])
    elapsed = time.time() - t0

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label file name", "symptom"])
        for pth, row in zip(paths, P):
            w.writerow([pth.name, str([c for c, v in zip(SYMPTOMS, row) if v])])
    print(f"{len(paths)} files → {a.output}  total {elapsed:.1f}s, {elapsed / max(1, len(paths)) * 1000:.2f} ms/file")


if __name__ == "__main__":
    main()
