"""
Mission 3 추론: 대화 전사 텍스트 → 환자 증상

python inference.py --audio_dir {wav folder} --label_dir {json folder} \
                    --ckpt_path {checkpoint file} --output ./outputs/mission3.csv

출력 CSV 컬럼: label file name, symptom
symptom 은 9개 클래스 중 예측된 증상의 문자열 리스트 (예: ['두통', '복통'], 없으면 [])
라벨 json 에서는 전사 텍스트(utterances[*].text)만 사용한다. (--audio_dir 은 공통 인터페이스용, 사용 안 함)
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from text_model import THRESHOLD, dialogue_text, load_ckpt, make_windows, read_label
from train import predict_probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", default=None)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output", default="./outputs/mission3.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model, tokenizer = load_ckpt(args.ckpt_path, device)
    symptoms = ckpt["symptoms"]
    print(f"device {device} | params {sum(x.numel() for x in model.parameters()) / 1e6:.1f}M")

    paths = sorted(Path(args.label_dir).rglob("*.json"))
    t0 = time.time()
    docs = [(make_windows(tokenizer, dialogue_text(read_label(pth)), ckpt["max_len"], ckpt["stride"],
                          ckpt["max_windows"]), None) for pth in paths]
    P = predict_probs(model, docs, device, pad_id=tokenizer.pad_token_id) if docs else []
    elapsed = time.time() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label file name", "symptom"])
        for pth, prob in zip(paths, P):
            pred = [c for c, x in zip(symptoms, prob) if x >= THRESHOLD]   # Threshold 0.5 고정
            w.writerow([pth.name, str(pred)])
    print(f"{len(paths)} files → {out}  ({elapsed:.1f}s, {elapsed / max(1, len(paths)) * 1000:.1f} ms/file)")


if __name__ == "__main__":
    main()
