"""
Mission 2 학습: 발화 조각이 신고자(1)인지 119대원(0)인지 분류 (CPU 경량 버전)

입력 : 라벨의 startAt / endAt 으로 자른 발화 조각 음성 (text·발화 순서 미사용)
방법 : 통화 전체 프레임 특징을 한 번 계산 → 조각 구간 통계(MFCC·F0·에너지·길이) → HistGradientBoosting
       통화당 최대 PER_CALL 개 조각을 화자별 균형 있게 샘플링
"""
import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from audio_feats import frame_features, index_files, ms_to_frames, pool, read_label, utterance_spans

PER_CALL = 10


def call_segments(args):
    """통화 1건 → [(특징, start, end, speaker)] (speaker 는 학습용 정답, 추론 시 None)."""
    wav_path, spans = args
    try:
        feats = frame_features(wav_path)
    except Exception as e:  # 손상 파일 대비
        print(f"[warn] {wav_path}: {e}", flush=True)
        return [(None, s, e, spk) for s, e, spk in spans]
    T = feats[0].shape[1]
    return [(pool(feats, ms_to_frames(s, e, T)), s, e, spk) for s, e, spk in spans]


def extract(jobs, workers, log=print):
    out, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, segs in enumerate(ex.map(call_segments, jobs, chunksize=4), 1):
            out.append(segs)
            if i % 300 == 0 or i == len(jobs):
                log(f"  calls {i}/{len(jobs)} ({time.time() - t0:.0f}s)")
    return out


def main(args, log=print):
    wavs, jsons = index_files(args.train_audio_dir, "wav"), index_files(args.train_label_dir, "json")
    stems = [s for s in wavs if s in jsons]
    rng = np.random.default_rng(42)
    pyrng = random.Random(42)
    if args.limit and args.limit < len(stems):
        stems = [stems[i] for i in sorted(rng.choice(len(stems), args.limit, replace=False))]

    jobs, groups = [], []
    for i, s in enumerate(stems):
        by = {}
        for sp in utterance_spans(read_label(jsons[s])):
            if str(sp[2]) in ("0", "1"):
                by.setdefault(int(sp[2]), []).append(sp)
        spans = [x for v in by.values() for x in pyrng.sample(v, min(len(v), PER_CALL // 2))]
        if spans:
            jobs.append((wavs[s], spans))
    log(f"학습 통화 {len(jobs)} / 조각 {sum(len(j[1]) for j in jobs)}")

    t0 = time.time()
    res = extract(jobs, args.workers, log)
    X, y, g = [], [], []
    for ci, segs in enumerate(res):
        for f, _, _, spk in segs:
            if f is not None:
                X.append(f); y.append(int(spk)); g.append(ci)
    X, y, g = np.stack(X), np.array(y), np.array(g)
    log(f"특징 추출 {time.time() - t0:.0f}s  X {X.shape}  speaker 분포(0=대원,1=신고자) {np.bincount(y).tolist()}")

    hold = set(rng.choice(len(jobs), max(1, len(jobs) // 10), replace=False).tolist())
    is_ho = np.array([c in hold for c in g])          # 같은 통화 조각은 같은 쪽으로 (누수 방지)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True,
                                         validation_fraction=0.1, random_state=42)
    clf.fit(X[~is_ho], y[~is_ho])
    acc = (clf.predict(X[is_ho]) == y[is_ho]).mean()
    log(f"holdout accuracy {acc:.4f}  (trees {clf.n_iter_})")
    clf.fit(X, y)
    joblib.dump({"model": clf, "holdout_acc": float(acc)}, args.ckpt_path)
    log(f"saved → {args.ckpt_path}")
    return {"holdout": float(acc), "n_train": int(len(y)), "train_sec": time.time() - t0}


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_audio_dir", required=True)
    p.add_argument("--train_label_dir", required=True)
    p.add_argument("--ckpt_path", default="mission2_model.pkl")
    p.add_argument("--limit", type=int, default=2000, help="학습 통화 수 (0=전체)")
    p.add_argument("--workers", type=int, default=4)
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
