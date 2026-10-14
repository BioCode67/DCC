"""
Mission 2 추론: 신고자(1) / 119대원(0)

python inference.py --audio_dir {wav folder} --label_dir {json folder} \
                    --ckpt_path {checkpoint file} --output ./outputs/mission2.csv

출력 CSV: audio file name, startAt, endAt, speaker
라벨 json 에서는 조각 경계(startAt / endAt)만 사용하며, 각 조각을 독립적으로 분류한다.
"""
import argparse
import csv
import os
import time
from pathlib import Path

import joblib
import numpy as np

from audio_feats import index_files, read_label, utterance_spans
from train import extract


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output", default="./outputs/mission2.csv")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = p.parse_args()

    ck = joblib.load(a.ckpt_path)
    wavs, jsons = index_files(a.audio_dir, "wav"), index_files(a.label_dir, "json")
    jobs = []
    for s, jp in jsons.items():
        if s in wavs:
            spans = [(st, en, None) for st, en, _ in utterance_spans(read_label(jp))]  # speaker 값은 버림
            if spans:
                jobs.append((wavs[s], spans))
    t0 = time.time()
    res = extract(jobs, a.workers)
    rows, X = [], []
    dim = None
    for (wp, _), segs in zip(jobs, res):
        for f, st, en, _ in segs:
            rows.append((wp.name, st, en))
            X.append(f)
            dim = dim or (f.shape[0] if f is not None else None)
    X = np.stack([f if f is not None else np.zeros(dim, np.float32) for f in X])
    pred = ck["model"].predict(X)          # binary: 확률 0.5 기준 (argmax)
    elapsed = time.time() - t0

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio file name", "startAt", "endAt", "speaker"])
        for (n, st, en), k in zip(rows, pred):
            w.writerow([n, st, en, int(k)])
    print(f"{len(rows)} segments / {len(jobs)} calls → {a.output}  total {elapsed:.1f}s, "
          f"{elapsed / max(1, len(rows)) * 1000:.1f} ms/segment")


if __name__ == "__main__":
    main()
