"""
Mission 1 학습: 음성으로부터 신고자 성별 분류 (CPU 경량 버전)

입력 : 음성 + 라벨의 startAt / endAt / speaker (규정상 허용 정보만 사용)
방법 : 신고자(speaker=1) 발화 구간 프레임만 모아 MFCC·F0·에너지 통계 → HistGradientBoosting
모델 선택 : Training set 내부 10% holdout (Validation 폴더는 학습에 사용하지 않음)
"""
import argparse
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from audio_feats import frame_features, index_files, ms_to_frames, pool, read_label, utterance_spans

CALLER = 1        # speaker=1: 신고자
MAX_SEC = 30      # 신고자 음성 최대 사용 길이(초)


def call_feature(args):
    """통화 1건 → 신고자 구간 특징 벡터. (wav 경로, 라벨 dict 또는 None)"""
    wav_path, label = args
    try:
        feats = frame_features(wav_path)
    except Exception as e:  # 손상 파일 대비
        print(f"[warn] {wav_path}: {e}", flush=True)
        return None
    T = feats[0].shape[1]
    spans = utterance_spans(label, speaker=CALLER)
    if spans:
        idx = np.concatenate([ms_to_frames(s, e, T) for s, e, _ in spans])[: MAX_SEC * 100]
    else:                                  # 신고자 구간이 없으면 전체 사용
        idx = np.arange(min(T, MAX_SEC * 100))
    return pool(feats, idx)


def extract(jobs, workers, log=print):
    out, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, f in enumerate(ex.map(call_feature, jobs, chunksize=8), 1):
            out.append(f)
            if i % 500 == 0 or i == len(jobs):
                log(f"  features {i}/{len(jobs)} ({time.time() - t0:.0f}s)")
    return out


def main(args, log=print):
    wavs, jsons = index_files(args.train_audio_dir, "wav"), index_files(args.train_label_dir, "json")
    stems = [s for s in wavs if s in jsons]
    rng = np.random.default_rng(42)
    if args.limit and args.limit < len(stems):          # 시간 절약용 무작위 샘플
        stems = [stems[i] for i in sorted(rng.choice(len(stems), args.limit, replace=False))]
    labels = {s: read_label(jsons[s]) for s in stems}
    stems = [s for s in stems if labels[s].get("gender") not in (None, "")]
    y_str = np.array([str(labels[s]["gender"]) for s in stems])
    log(f"학습 통화 {len(stems)}  성별 분포 {dict(Counter(y_str.tolist()))}")

    t0 = time.time()
    feats = extract([(wavs[s], labels[s]) for s in stems], args.workers, log)
    ok = [i for i, f in enumerate(feats) if f is not None]
    X, y_str = np.stack([feats[i] for i in ok]), y_str[ok]
    log(f"특징 추출 {time.time() - t0:.0f}s  X {X.shape}")

    classes = sorted(set(y_str))
    y = np.array([classes.index(v) for v in y_str])
    perm = rng.permutation(len(y))
    n_ho = max(1, len(y) // 10)
    ho, tr = perm[:n_ho], perm[n_ho:]
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True,
                                         validation_fraction=0.1, random_state=42)
    clf.fit(X[tr], y[tr])
    acc = (clf.predict(X[ho]) == y[ho]).mean()
    log(f"holdout accuracy {acc:.4f}  (trees {clf.n_iter_})")
    clf.fit(X, y)                                        # 전체 Training 으로 재학습
    joblib.dump({"model": clf, "classes": classes, "holdout_acc": float(acc)}, args.ckpt_path)
    log(f"saved → {args.ckpt_path}")
    return {"holdout": float(acc), "n_train": int(len(y)), "train_sec": time.time() - t0}


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_audio_dir", required=True)
    p.add_argument("--train_label_dir", required=True)
    p.add_argument("--ckpt_path", default="mission1_model.pkl")
    p.add_argument("--limit", type=int, default=3000, help="학습 통화 수 (0=전체)")
    p.add_argument("--workers", type=int, default=4)
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
