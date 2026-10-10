"""
Mission 2 학습: 발화 조각이 신고자(speaker=1)인지 119대원(speaker=0)인지 분류

입력 : 한 사람의 한 발화 조각 (라벨의 startAt / endAt 으로 자른 음성)
       ※ text 등 다른 annotation, 발화 순서(홀짝 등)는 사용하지 않는다.
출력 : speaker (1: 신고자, 0: 119대원)

방법 : 발화 조각(최대 MAX_SEC 초) → 고정 WavLM 특징 → MLP 헤드
       학습 시 통화당 최대 PER_CALL 개 조각을 화자별 균형 있게 샘플링 (특징 추출 시간 절약)
"""
import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from audio_model import (FrozenEncoder, count_params, fit_length, index_files, load_pretrained_wavlm,
                         load_wav, read_label, save_ckpt, slice_ms, train_head, utterance_spans)

MAX_SEC = 8      # 발화 조각 최대 길이(초)
PER_CALL = 8     # 학습 시 통화당 샘플링할 조각 수


def list_collate(batch):
    return batch


class SegmentDataset(Dataset):
    """통화 1건 → (클립 목록, 메타 목록). spans=None 이면 라벨의 모든 발화를 사용."""

    def __init__(self, items):
        self.items = items          # [(wav_path, json_name, [(start, end, speaker), ...]), ...]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        wav_path, json_name, spans = self.items[i]
        try:
            wav = load_wav(wav_path)
        except Exception as e:  # 손상 파일 대비
            print(f"[warn] {wav_path.name}: {e}")
            wav = np.zeros(16000, dtype=np.float32)
        clips = [fit_length(slice_ms(wav, s, e), MAX_SEC) for s, e, _ in spans]
        metas = [(wav_path.name, s, e, spk) for s, e, spk in spans]
        return clips, metas


def extract_features(items, encoder, num_workers=4, chunk=16, log=print):
    loader = DataLoader(SegmentDataset(items), batch_size=chunk, num_workers=num_workers,
                        collate_fn=list_collate)
    feats, metas, t0 = [], [], time.time()
    for i, batch in enumerate(loader):
        clips = [c for cs, _ in batch for c in cs]
        metas += [m for _, ms in batch for m in ms]
        feats.append(encoder.encode(clips))
        if i % 50 == 0:
            log(f"  calls {min((i + 1) * chunk, len(items))}/{len(items)}  segments {len(metas)}  ({time.time() - t0:.0f}s)")
    return torch.cat(feats).numpy(), metas


def sample_spans(spans, k, rng):
    """화자별로 균형 있게 최대 k 개 조각 샘플링."""
    by_spk = {}
    for sp in spans:
        if str(sp[2]) in ("0", "1"):
            by_spk.setdefault(int(sp[2]), []).append(sp)
    out = []
    for v in by_spk.values():
        out += rng.sample(v, min(len(v), k // max(1, len(by_spk))))
    return out


def build_train_items(audio_dir, label_dir, per_call, seed=42, log=print):
    wavs, jsons = index_files(audio_dir, "wav"), index_files(label_dir, "json")
    rng = random.Random(seed)
    stems = [s for s in wavs if s in jsons]
    with ThreadPoolExecutor(32) as ex:              # Drive 에서 json 병렬 읽기
        labels = dict(zip(stems, ex.map(lambda s: read_label(jsons[s]), stems)))
    items = []
    for stem, wp in wavs.items():
        jp = jsons.get(stem)
        if jp is None:
            continue
        spans = sample_spans(utterance_spans(labels[stem]), per_call, rng)
        if spans:
            items.append((wp, jp.name, spans))
    log(f"wav {len(wavs)} / json {len(jsons)} / 학습 통화 {len(items)} / 조각 {sum(len(s) for *_, s in items)}")
    return items


def main(args, log=print):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    items = build_train_items(args.train_audio_dir, args.train_label_dir, args.per_call, log=log)
    if args.limit and args.limit < len(items):      # 무작위 부분 샘플 (무료 Colab 등 자원 제한 시)
        keep = sorted(np.random.default_rng(42).choice(len(items), args.limit, replace=False))
        items = [items[i] for i in keep]

    wavlm = load_pretrained_wavlm(args.pretrained)
    encoder = FrozenEncoder(wavlm, device)
    log(f"WavLM 파라미터 {count_params(wavlm) / 1e6:.1f}M (frozen)")

    feat_path = cache / f"m2_train_feats_{len(items)}_{args.per_call}.npz"
    if feat_path.exists():
        d = np.load(feat_path)
        X, y, groups = d["X"], d["y"], d["groups"]
        log(f"cached features {X.shape}")
    else:
        t0 = time.time()
        X, metas = extract_features(items, encoder, args.num_workers, log=log)
        y = np.array([int(m[3]) for m in metas])
        groups = np.array([m[0] for m in metas])          # 같은 통화 조각은 같은 holdout 그룹
        np.savez(feat_path, X=X, y=y, groups=groups)
        log(f"feature extraction {time.time() - t0:.0f}s  shape {X.shape}")
    log(f"speaker 분포 (0=119대원, 1=신고자) {np.bincount(y).tolist()}")

    head, best = train_head(X, y, groups, epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                            device=device, log=log)
    log(f"헤드 파라미터 {count_params(head) / 1e3:.1f}K")
    save_ckpt(args.ckpt_path, wavlm, head, {"max_sec": MAX_SEC, "holdout_acc": float(best)})
    log(f"saved → {args.ckpt_path}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_audio_dir", required=True)
    p.add_argument("--train_label_dir", required=True)
    p.add_argument("--ckpt_path", default="./checkpoints/mission2.pt")
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--pretrained", default="microsoft/wavlm-base-plus")
    p.add_argument("--per_call", type=int, default=PER_CALL)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="N개 통화만 무작위 샘플링해 사용 (0=전체)")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
