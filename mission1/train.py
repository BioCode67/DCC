"""
Mission 1 학습: 음성으로부터 신고자 성별 분류

입력 : 음성(원천데이터) + 라벨의 startAt / endAt / speaker (규정상 허용된 정보만 사용)
출력 : gender (라벨링데이터의 gender 값 그대로, 예: 'F' / 'M')

방법 : 신고자(speaker=1) 발화 구간만 이어붙여 최대 MAX_SEC 초 → 고정 WavLM 특징 → MLP 헤드
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from audio_model import (FrozenEncoder, count_params, fit_length, index_files, load_pretrained_wavlm,
                         load_wav, read_label, save_ckpt, slice_ms, train_head, utterance_spans)

CALLER = 1      # speaker=1: 신고자, speaker=0: 119대원
MAX_SEC = 30    # 성별 판단에 사용할 신고자 음성 최대 길이(초)


def caller_clip(wav, label):
    """신고자 발화 구간을 이어붙인 클립. 신고자 구간이 없으면 전체 음성 사용."""
    parts, total = [], 0
    for s, e, _ in utterance_spans(label, speaker=CALLER) if label else []:
        seg = slice_ms(wav, s, e)
        parts.append(seg)
        total += len(seg)
        if total >= MAX_SEC * 16000:
            break
    clip = np.concatenate(parts) if parts else wav
    return fit_length(clip, MAX_SEC)


def list_collate(batch):
    return batch


class CallDataset(Dataset):
    """(wav 경로, json 경로) → 신고자 클립. DataLoader worker 로 I/O 병렬화."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        wav_path, json_path = self.pairs[i]
        label = read_label(json_path) if json_path else None
        try:
            return caller_clip(load_wav(wav_path), label)
        except Exception as e:  # 손상 파일 대비
            print(f"[warn] {wav_path.name}: {e}")
            return np.zeros(16000, dtype=np.float32)


def extract_features(pairs, encoder, num_workers=4, chunk=64, log=print):
    loader = DataLoader(CallDataset(pairs), batch_size=chunk, num_workers=num_workers,
                        collate_fn=list_collate)
    feats, t0 = [], time.time()
    for i, clips in enumerate(loader):
        feats.append(encoder.encode(clips))
        if i % 20 == 0:
            log(f"  features {min((i + 1) * chunk, len(pairs))}/{len(pairs)}  ({time.time() - t0:.0f}s)")
    return torch.cat(feats).numpy()


def labeled_pairs(audio_dir, label_dir, log=print):
    wavs, jsons = index_files(audio_dir, "wav"), index_files(label_dir, "json")
    pairs, genders = [], []
    for stem, wp in wavs.items():
        jp = jsons.get(stem)
        if jp is None:
            continue
        g = read_label(jp).get("gender")
        if g in (None, "", "미상"):
            continue
        pairs.append((wp, jp))
        genders.append(str(g))
    log(f"wav {len(wavs)} / json {len(jsons)} / 학습가능 쌍 {len(pairs)}")
    return pairs, genders


def main(args, log=print):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    pairs, genders = labeled_pairs(args.train_audio_dir, args.train_label_dir, log)
    if args.limit and args.limit < len(pairs):      # 무작위 부분 샘플 (무료 Colab 등 자원 제한 시)
        keep = sorted(np.random.default_rng(42).choice(len(pairs), args.limit, replace=False))
        pairs, genders = [pairs[i] for i in keep], [genders[i] for i in keep]
    classes = sorted(set(genders))
    y = np.array([classes.index(g) for g in genders])
    log(f"classes {classes}  분포 {np.bincount(y).tolist()}")

    wavlm = load_pretrained_wavlm(args.pretrained)
    encoder = FrozenEncoder(wavlm, device)
    log(f"WavLM 파라미터 {count_params(wavlm) / 1e6:.1f}M (frozen)")

    feat_path = cache / f"m1_train_feats_{len(pairs)}.npy"
    if feat_path.exists():
        X = np.load(feat_path)
        log(f"cached features {X.shape}")
    else:
        t0 = time.time()
        X = extract_features(pairs, encoder, args.num_workers, log=log)
        np.save(feat_path, X)
        log(f"feature extraction {time.time() - t0:.0f}s  shape {X.shape}")

    head, best = train_head(X, y, groups=np.arange(len(y)), epochs=args.epochs, lr=args.lr,
                            hidden=args.hidden, device=device, log=log)
    log(f"헤드 파라미터 {count_params(head) / 1e3:.1f}K")
    save_ckpt(args.ckpt_path, wavlm, head, {"classes": classes, "max_sec": MAX_SEC,
                                            "holdout_acc": float(best)})
    log(f"saved → {args.ckpt_path}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_audio_dir", required=True)
    p.add_argument("--train_label_dir", required=True)
    p.add_argument("--ckpt_path", default="./checkpoints/mission1.pt")
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--pretrained", default="microsoft/wavlm-base-plus")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="N개 통화만 무작위 샘플링해 사용 (0=전체)")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
