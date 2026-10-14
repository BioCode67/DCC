"""
Mission 1 추론: 신고자 성별

python inference.py --audio_dir {wav folder} --label_dir {json folder} \
                    --ckpt_path {checkpoint file} --output ./outputs/mission1.csv

출력 CSV: audio file name, gender   (라벨 json 은 startAt / endAt / speaker 만 사용)
"""
import argparse
import csv
import os
import time
from pathlib import Path

import joblib
import numpy as np

from audio_feats import index_files, read_label
from train import extract


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--label_dir", default=None)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output", default="./outputs/mission1.csv")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = p.parse_args()

    ck = joblib.load(a.ckpt_path)
    wavs, jsons = index_files(a.audio_dir, "wav"), index_files(a.label_dir, "json")
    stems = list(wavs)
    t0 = time.time()
    # speaker 등 허용 정보만 쓰도록 utterances 의 startAt/endAt/speaker 만 남겨 전달
    jobs = []
    for s in stems:
        lab = read_label(jsons[s]) if s in jsons else None
        if lab:
            lab = {"utterances": [{k: u.get(k) for k in ("startAt", "endAt", "speaker")}
                                  for u in lab.get("utterances") or []]}
        jobs.append((wavs[s], lab))
    feats = extract(jobs, a.workers)
    dim = next(f.shape[0] for f in feats if f is not None)
    X = np.stack([f if f is not None else np.zeros(dim, np.float32) for f in feats])
    pred = ck["model"].predict(X)          # binary: 확률 0.5 기준 (argmax)
    elapsed = time.time() - t0

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio file name", "gender"])
        for s, k in zip(stems, pred):
            w.writerow([wavs[s].name, ck["classes"][k]])
    print(f"{len(stems)} files → {a.output}  total {elapsed:.1f}s, {elapsed / max(1, len(stems)) * 1000:.1f} ms/file")


if __name__ == "__main__":
    main()
