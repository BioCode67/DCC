"""
Mission 3 공통 모듈: 대화 전사 텍스트 → 9개 증상 multi-label 분류

- 입력 텍스트: 라벨 json 의 utterances[*].text 를 발화 순서대로 이어붙임
               (규정상 전사 텍스트 외의 라벨 정보(speaker 등)는 사용하지 않음)
- 긴 대화 처리: 토큰을 길이 max_len 의 window 로 나누고(stride 겹침), window 별 logit 을
               문서 단위로 max-pooling (증상이 대화 어디에서 언급되든 잡아낼 수 있음)
- 판정: sigmoid 확률 >= 0.5 → 해당 증상 있음 (대회 규정: Threshold 0.5 고정, 클래스 공통)
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

SYMPTOMS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
THRESHOLD = 0.5


def read_label(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def dialogue_text(label):
    """발화 텍스트만 순서대로 연결."""
    texts = [str(u.get("text") or "").strip() for u in label.get("utterances") or []]
    return " ".join(t for t in texts if t)


def target_vector(label):
    """symptom 리스트 → 9차원 0/1 (9개 클래스 외 증상은 제외)."""
    sym = label.get("symptom") or []
    if isinstance(sym, str):
        sym = [sym]
    s = {str(x).strip() for x in sym}
    return [1.0 if c in s else 0.0 for c in SYMPTOMS]


def make_windows(tokenizer, text, max_len=512, stride=384, max_windows=4):
    """텍스트 → [CLS] window [SEP] 토큰 id 목록들. window 가 많으면 고르게 골라 max_windows 개 사용."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    body = max_len - 2
    starts = list(range(0, max(1, len(ids) - body + stride), stride)) or [0]
    if len(starts) > max_windows:
        step = (len(starts) - 1) / (max_windows - 1) if max_windows > 1 else 0
        starts = [starts[round(i * step)] for i in range(max_windows)]
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    return [[cls] + ids[s : s + body] + [sep] for s in starts]


def collate_docs(docs, pad_id):
    """docs: [(windows, target or None), ...] → 평탄화된 window 배치 + 문서 index."""
    wins, doc_idx, targets = [], [], []
    for d, (ws, t) in enumerate(docs):
        wins += ws
        doc_idx += [d] * len(ws)
        if t is not None:
            targets.append(t)
    L = max(len(w) for w in wins)
    ids = torch.full((len(wins), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(wins), L), dtype=torch.long)
    for i, w in enumerate(wins):
        ids[i, : len(w)] = torch.tensor(w)
        mask[i, : len(w)] = 1
    batch = {"input_ids": ids, "attention_mask": mask,
             "doc_index": torch.tensor(doc_idx), "n_docs": len(docs)}
    if targets:
        batch["targets"] = torch.tensor(targets)
    return batch


class DocClassifier(nn.Module):
    """사전학습 한국어 인코더(KLUE-RoBERTa) + window max-pooling multi-label 헤드."""

    def __init__(self, encoder, n_labels=len(SYMPTOMS), dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask, doc_index, n_docs):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        win_logits = self.classifier(self.dropout(h)).float()
        out = torch.full((n_docs, win_logits.shape[1]), float("-inf"), device=win_logits.device)
        idx = doc_index.to(win_logits.device)[:, None].expand_as(win_logits)
        return out.scatter_reduce(0, idx, win_logits, reduce="amax")


def build_model(name_or_config, pretrained=True):
    from transformers import AutoConfig, AutoModel
    if pretrained:
        enc = AutoModel.from_pretrained(name_or_config, add_pooling_layer=False)
    else:
        cfg = AutoConfig.for_model(**name_or_config)
        enc = AutoModel.from_config(cfg, add_pooling_layer=False)
    return DocClassifier(enc)


def save_ckpt(path, model, tokenizer, extra):
    """체크포인트 + 같은 폴더의 tokenizer/ 디렉터리 저장 (추론 시 인터넷 불필요)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_config": model.encoder.config.to_dict(),
        "state_dict": {k: v.half() for k, v in model.state_dict().items()},   # fp16 저장
        "symptoms": SYMPTOMS,
        **extra,
    }, path)
    tokenizer.save_pretrained(path.parent / "tokenizer")


def load_ckpt(path, device):
    from transformers import AutoTokenizer
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["encoder_config"], pretrained=False)
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    tok_dir = path.parent / "tokenizer"
    if not tok_dir.exists():                       # ckpt 와 다른 위치에 둔 경우 스크립트 폴더에서 찾음
        tok_dir = Path(__file__).resolve().parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    return ckpt, model.to(device).eval(), tokenizer


def macro_f1(y_true, y_pred):
    """대회 방식: 증상별 F1 을 계산한 뒤 9개 클래스 단순 평균."""
    import numpy as np
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in range(y_true.shape[1]):
        tp = ((y_true[:, c] == 1) & (y_pred[:, c] == 1)).sum()
        fp = ((y_true[:, c] == 0) & (y_pred[:, c] == 1)).sum()
        fn = ((y_true[:, c] == 1) & (y_pred[:, c] == 0)).sum()
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)), f1s
