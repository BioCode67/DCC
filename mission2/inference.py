"""
Mission 2 추론: 신고자(1) / 119대원(0) 발화 분류

python inference.py --audio_dir {wav folder} --label_dir {json folder} \
                    --ckpt_path {checkpoint file} --output ./outputs/mission2.csv

출력 CSV 컬럼: audio file name, startAt, endAt, speaker
라벨 json 에서는 조각 경계(startAt / endAt)만 사용한다. 각 조각은 독립적으로 분류한다.
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from audio_model import count_params, index_files, load_ckpt, predict_head, read_label, utterance_spans
from train import extract_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output", default="./outputs/mission2.csv")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, encoder, head = load_ckpt(args.ckpt_path, device)
    print(f"device {device} | params total {(count_params(encoder.model) + count_params(head)) / 1e6:.1f}M")

    wavs = index_files(args.audio_dir, "wav")
    items = []
    for stem, jp in index_files(args.label_dir, "json").items():
        if stem not in wavs:
            print(f"[warn] wav 없음: {stem}")
            continue
        # speaker 값은 읽지 않도록 None 으로 덮어씀 (조각 경계만 사용)
        spans = [(s, e, None) for s, e, _ in utterance_spans(read_label(jp))]
        if spans:
            items.append((wavs[stem], jp.name, spans))

    t0 = time.time()
    X, metas = extract_features(items, encoder, args.num_workers)
    pred = predict_head(head, X, device)
    elapsed = time.time() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio file name", "startAt", "endAt", "speaker"])
        for (name, s, e, _), k in zip(metas, pred):
            w.writerow([name, s, e, int(k)])
    print(f"{len(metas)} segments / {len(items)} calls → {out}  "
          f"({elapsed:.1f}s, {elapsed / max(1, len(metas)) * 1000:.1f} ms/segment)")


if __name__ == "__main__":
    main()
