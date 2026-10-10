"""
음성 공통 모듈 (Mission 1, 2 공용)

- 데이터 탐색: wav / json 파일을 하위 폴더까지 재귀 탐색하여 파일명(stem) 기준으로 짝지음
- 오디오 로딩: 16kHz mono float32 로 통일 (원본 샘플레이트가 달라도 resample)
- 인코더: 사전학습 WavLM(microsoft/wavlm-base-plus)을 '고정(frozen) 특징 추출기'로 사용
          지정된 layer들의 hidden state 를 시간축 mean/std pooling → (n_layers, 2*768) 특징
- 헤드: layer 가중합(학습 가능) → LayerNorm → MLP → 2-class logits
        (softmax 확률 >= 0.5 ↔ argmax 이므로 대회 Threshold 0.5 규정을 그대로 만족)

체크포인트(.pt) 하나에 WavLM 설정/가중치 + 헤드 가중치를 모두 저장하므로
채점 환경에서 인터넷 다운로드 없이 추론이 가능하다.
"""
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly

SR = 16000                 # 모델 입력 샘플레이트
LAYERS = (3, 6, 9, 12)     # pooling 할 WavLM hidden layer 번호 (0 = CNN 출력)
MIN_SAMPLES = SR // 2      # 너무 짧은 조각은 0.5초까지 zero-padding


# ---------------------------------------------------------------- 데이터 탐색 / 로딩
def index_files(root, ext):
    """root 하위의 *.ext 파일을 {파일명 stem: 경로} 로 반환 (하위 폴더 재귀 탐색)."""
    if root is None:
        return {}
    return {p.stem: p for p in sorted(Path(root).rglob(f"*.{ext}"))}


def read_label(path):
    """라벨 json 읽기 (BOM 유무 모두 처리)."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def utterance_spans(label, speaker=None):
    """라벨에서 (startAt, endAt, speaker) 목록 추출. 시간 단위는 ms.
    ※ 규정상 startAt / endAt / speaker 만 사용한다 (text 등은 사용하지 않음)."""
    spans = []
    for u in label.get("utterances") or []:
        s, e, spk = u.get("startAt"), u.get("endAt"), u.get("speaker")
        if s is None or e is None or e <= s:
            continue
        if speaker is not None and str(spk) != str(speaker):
            continue
        spans.append((int(s), int(e), spk))
    return spans


def load_wav(path, sr=SR):
    """wav → 16kHz mono float32 numpy."""
    wav, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if orig_sr != sr:
        g = gcd(int(orig_sr), sr)
        wav = resample_poly(wav, sr // g, int(orig_sr) // g).astype(np.float32)
    return wav


def slice_ms(wav, start_ms, end_ms, sr=SR):
    a = max(0, int(start_ms * sr / 1000))
    b = min(len(wav), int(end_ms * sr / 1000))
    return wav[a:b]


def fit_length(x, max_sec):
    """최대 길이 제한 + 최소 길이 padding."""
    x = x[: int(max_sec * SR)]
    if len(x) < MIN_SAMPLES:
        x = np.pad(x, (0, MIN_SAMPLES - len(x)))
    return x


# ---------------------------------------------------------------- 인코더
def load_pretrained_wavlm(name="microsoft/wavlm-base-plus"):
    """학습 단계에서만 호출 (인터넷 필요). 추론은 체크포인트에서 복원한다."""
    from transformers import WavLMModel
    return WavLMModel.from_pretrained(name)


def wavlm_from_ckpt(ckpt):
    """체크포인트에 저장된 config + state_dict 로 WavLM 복원 (인터넷 불필요)."""
    from transformers import WavLMConfig, WavLMModel
    model = WavLMModel(WavLMConfig.from_dict(ckpt["wavlm_config"]))
    model.load_state_dict({k: v.float() for k, v in ckpt["wavlm_state"].items()})
    return model


class FrozenEncoder:
    """WavLM 을 고정한 채 클립 목록 → pooled 특징 (N, len(LAYERS), 2*H) 을 계산."""

    def __init__(self, wavlm, device, layers=LAYERS, fp16=True):
        self.model = wavlm.to(device).eval()
        self.device = device
        self.layers = layers
        self.fp16 = fp16 and device.type == "cuda"

    @torch.no_grad()
    def _batch(self, clips):
        lens = [len(c) for c in clips]
        x = np.zeros((len(clips), max(lens)), dtype=np.float32)
        for i, c in enumerate(clips):
            c = (c - c.mean()) / (c.std() + 1e-7)        # 클립별 정규화 (Wav2Vec2FeatureExtractor 와 동일)
            x[i, : len(c)] = c
        x = torch.from_numpy(x).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            hs = self.model(x, output_hidden_states=True).hidden_states
        frames = self.model._get_feat_extract_output_lengths(torch.tensor(lens)).tolist()
        T = hs[0].shape[1]
        mask = torch.zeros(len(clips), T, 1, device=self.device)
        for i, f in enumerate(frames):
            mask[i, : max(1, min(int(f), T))] = 1
        n = mask.sum(1)
        out = []
        for l in self.layers:
            h = hs[l].float()
            mean = (h * mask).sum(1) / n
            std = (((h - mean[:, None]) ** 2 * mask).sum(1) / n).clamp_min(1e-6).sqrt()
            out.append(torch.cat([mean, std], -1))
        return torch.stack(out, 1).cpu().half()             # (B, L, 2H)

    def encode(self, clips, max_batch_sec=240, max_batch=64):
        """길이순 정렬 후 총 길이 budget 단위로 batch 처리 (padding 낭비 최소화)."""
        if not clips:
            return torch.zeros(0, len(self.layers), 2 * self.model.config.hidden_size, dtype=torch.float16)
        order = np.argsort([len(c) for c in clips])[::-1]
        feats = [None] * len(clips)
        i = 0
        while i < len(order):
            longest = len(clips[order[i]])
            bs = int(max(1, min(max_batch, max_batch_sec * SR // longest)))
            idx = order[i : i + bs]
            for j, f in zip(idx, self._batch([clips[k] for k in idx])):
                feats[j] = f
            i += bs
        return torch.stack(feats)


# ---------------------------------------------------------------- 분류 헤드
class LayerWeightedHead(nn.Module):
    """(B, L, D) 특징 → layer softmax 가중합 → LayerNorm → MLP → (B, n_cls)."""

    def __init__(self, n_layers=len(LAYERS), dim=1536, hidden=256, n_cls=2, dropout=0.2):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_cls),
        )

    def forward(self, feats):
        w = torch.softmax(self.layer_w, 0)
        x = (feats.float() * w[None, :, None]).sum(1)
        return self.mlp(self.norm(x))


def train_head(X, y, groups, epochs=30, lr=1e-3, batch=512, hidden=256, dropout=0.2,
               holdout=0.1, seed=42, device=None, log=print):
    """Training set 내부에서 call 단위(group) holdout 을 떼어 epoch 선택 → 최적 헤드 반환.
    (Validation 폴더는 학습/모델선택에 쓰지 않는다.)"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    hold_g = set(rng.choice(ug, size=max(1, int(len(ug) * holdout)), replace=False).tolist())
    is_hold = np.array([g in hold_g for g in groups])
    tr, ho = np.where(~is_hold)[0], np.where(is_hold)[0]
    log(f"train {len(tr)} / holdout {len(ho)} samples")

    torch.manual_seed(seed)
    head = LayerWeightedHead(X.shape[1], X.shape[2], hidden, int(y.max()) + 1, dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.long)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        head.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            loss = nn.functional.cross_entropy(head(Xt[b].to(device)), yt[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        acc = (predict_head(head, X[ho], device) == y[ho]).mean()
        log(f"epoch {ep:02d}  train_loss {tot / len(tr):.4f}  holdout_acc {acc:.4f}")
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    log(f"best holdout acc {best:.4f}")
    return head.cpu(), best


@torch.no_grad()
def predict_head(head, X, device=None, batch=2048, return_prob=False):
    device = device or next(head.parameters()).device
    head.eval()
    probs = []
    for i in range(0, len(X), batch):
        probs.append(torch.softmax(head(torch.as_tensor(X[i : i + batch]).to(device)), -1).cpu())
    p = torch.cat(probs).numpy() if probs else np.zeros((0, 2))
    return p if return_prob else p.argmax(1)        # argmax == (p1 >= 0.5) for 2 classes


def save_ckpt(path, wavlm, head, extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "wavlm_config": wavlm.config.to_dict(),
        "wavlm_state": {k: v.half() for k, v in wavlm.state_dict().items()},   # fp16 로 용량 절반
        "head_state": head.state_dict(),
        "head_args": dict(n_layers=head.layer_w.numel(), dim=head.norm.normalized_shape[0],
                          hidden=head.mlp[1].out_features, n_cls=head.mlp[4].out_features),
        "layers": LAYERS,
        **extra,
    }, path)


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder = FrozenEncoder(wavlm_from_ckpt(ckpt), device, tuple(ckpt["layers"]))
    head = LayerWeightedHead(**ckpt["head_args"])
    head.load_state_dict(ckpt["head_state"])
    return ckpt, encoder, head.to(device).eval()


def count_params(module):
    return sum(p.numel() for p in module.parameters())
