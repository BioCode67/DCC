"""
Mission 3 학습: 대화 전사 텍스트 → 환자 증상(9개 클래스) multi-label 분류

모델 : klue/roberta-base fine-tuning + window max-pooling (text_model.py 참고)
손실 : BCEWithLogits (+ 선택: 희소 클래스 pos_weight — 손실 설계, Threshold 는 0.5 고정)
모델 선택 : Training set 내부 10% holdout 의 Macro F1 (Validation 폴더는 사용하지 않음)
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from text_model import (SYMPTOMS, THRESHOLD, build_model, collate_docs, dialogue_text, macro_f1,
                        make_windows, read_label, save_ckpt, target_vector)


class DocDataset(Dataset):
    def __init__(self, docs):
        self.docs = docs       # [(windows, target), ...]

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        return self.docs[i]


def load_docs(label_dir, tokenizer, args, log=print):
    paths = sorted(Path(label_dir).rglob("*.json"))
    docs, names = [], []
    for p in paths:
        lab = read_label(p)
        docs.append((make_windows(tokenizer, dialogue_text(lab), args.max_len, args.stride,
                                  args.max_windows), target_vector(lab)))
        names.append(p.name)
    log(f"{label_dir}: {len(docs)} docs, 평균 window {np.mean([len(w) for w, _ in docs]):.2f}")
    return docs, names


@torch.no_grad()
def predict_probs(model, docs, device, batch_docs=16, pad_id=1):
    model.eval()
    out = []
    for i in range(0, len(docs), batch_docs):
        b = collate_docs([(w, None) for w, _ in docs[i : i + batch_docs]], pad_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                           b["doc_index"], b["n_docs"])
        out.append(torch.sigmoid(logits).cpu())
    return torch.cat(out).numpy()


def main(args, log=print):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    docs, _ = load_docs(args.train_label_dir, tokenizer, args, log)
    if args.limit and args.limit < len(docs):       # 무작위 부분 샘플
        keep = sorted(np.random.default_rng(args.seed).choice(len(docs), args.limit, replace=False))
        docs = [docs[i] for i in keep]
    Y = np.array([t for _, t in docs])
    log("클래스별 양성 수: " + ", ".join(f"{c} {int(n)}" for c, n in zip(SYMPTOMS, Y.sum(0))))
    log(f"증상 0개 문서 비율 {(Y.sum(1) == 0).mean():.3f}")

    # Training set 내부 holdout (모델 선택 용)
    perm = np.random.permutation(len(docs))
    n_hold = max(1, int(len(docs) * args.holdout))
    hold_idx, tr_idx = perm[:n_hold], perm[n_hold:]
    tr_docs = [docs[i] for i in tr_idx]
    ho_docs = [docs[i] for i in hold_idx]

    model = build_model(args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"모델 파라미터 {n_params / 1e6:.1f}M (전체 학습)")

    pos_weight = None
    if args.pos_weight:
        pos = Y[tr_idx].sum(0)
        neg = len(tr_idx) - pos
        pos_weight = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1.0, args.pos_weight_cap),
                                  dtype=torch.float, device=device)
        log("pos_weight: " + ", ".join(f"{c} {w:.2f}" for c, w in zip(SYMPTOMS, pos_weight.tolist())))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    pad_id = tokenizer.pad_token_id
    loader = DataLoader(DocDataset(tr_docs), batch_size=args.batch_docs, shuffle=True,
                        collate_fn=lambda b: collate_docs(b, pad_id))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.06), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, t0 = -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for step, b in enumerate(loader, 1):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                               b["doc_index"], b["n_docs"])
            loss = loss_fn(logits.float(), b["targets"].to(device))
            scaler.scale(loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            tot += loss.item()
            n += 1
            if step % 200 == 0:
                log(f"  ep {ep} step {step}/{len(loader)} loss {tot / n:.4f} ({time.time() - t0:.0f}s)")

        P = predict_probs(model, ho_docs, device, pad_id=pad_id)
        f1, per = macro_f1(np.array([t for _, t in ho_docs]), (P >= THRESHOLD).astype(int))
        log(f"epoch {ep} train_loss {tot / n:.4f} holdout MacroF1 {f1:.4f} | "
            + " ".join(f"{c}:{x:.2f}" for c, x in zip(SYMPTOMS, per)))
        if f1 > best:
            best = f1
            save_ckpt(args.ckpt_path, model, tokenizer,
                      {"max_len": args.max_len, "stride": args.stride,
                       "max_windows": args.max_windows_infer, "holdout_f1": f1})
            log(f"  ↳ best 갱신, saved → {args.ckpt_path}")
    log(f"done. best holdout MacroF1 {best:.4f}  ({time.time() - t0:.0f}s)")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_label_dir", required=True)
    p.add_argument("--ckpt_path", default="./checkpoints/mission3.pt")
    p.add_argument("--pretrained", default="klue/roberta-base")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_docs", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--max_windows", type=int, default=4, help="학습 시 문서당 최대 window 수")
    p.add_argument("--max_windows_infer", type=int, default=8, help="추론 시 문서당 최대 window 수")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--pos_weight", type=int, default=1, help="1: 희소 클래스 pos_weight 사용")
    p.add_argument("--pos_weight_cap", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="N개 문서만 무작위 샘플링해 사용 (0=전체)")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
