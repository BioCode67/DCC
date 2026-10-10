"""
Mission 1 추론: 신고자 성별 분류

python inference.py --audio_dir {wav folder} --label_dir {json folder} \
                    --ckpt_path {checkpoint file} --output ./outputs/mission1.csv

출력 CSV 컬럼: audio file name, gender
라벨 json 에서는 startAt / endAt / speaker 만 사용한다 (신고자 발화 구간 추출용).
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from audio_model import count_params, index_files, load_ckpt, predict_head
from train import extract_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--label_dir", default=None)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output", default="./outputs/mission1.csv")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, encoder, head = load_ckpt(args.ckpt_path, device)
    print(f"device {device} | params total {(count_params(encoder.model) + count_params(head)) / 1e6:.1f}M")

    wavs = index_files(args.audio_dir, "wav")
    jsons = index_files(args.label_dir, "json")
    pairs = [(wp, jsons.get(stem)) for stem, wp in wavs.items()]   # json 이 없으면 전체 음성 사용

    t0 = time.time()
    X = extract_features(pairs, encoder, args.num_workers)
    pred = predict_head(head, X, device)
    elapsed = time.time() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio file name", "gender"])
        for (wp, _), k in zip(pairs, pred):
            w.writerow([wp.name, ckpt["classes"][k]])
    print(f"{len(pairs)} files → {out}  ({elapsed:.1f}s, {elapsed / max(1, len(pairs)) * 1000:.1f} ms/file)")


if __name__ == "__main__":
    main()
