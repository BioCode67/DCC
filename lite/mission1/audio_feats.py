"""
음성 특징 추출 공통 모듈 (Mission 1, 2 공용 / CPU 전용 경량 버전)

- 16kHz mono 로 로딩
- 프레임(10ms) 단위로 MFCC 20차, 기본주파수(F0, 목소리 높낮이), RMS 에너지 계산
- 원하는 구간(프레임 index)에 대해 통계값(평균/표준편차/분위수)을 pooling → 고정 길이 벡터
"""
import json
from pathlib import Path

import librosa
import numpy as np

SR = 16000
HOP = 160          # 10ms
N_MFCC = 20


def index_files(root, ext):
    """root 하위 *.ext → {파일명 stem: 경로}."""
    if not root:
        return {}
    return {p.stem: p for p in sorted(Path(root).rglob(f"*.{ext}"))}


def read_label(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def utterance_spans(label, speaker=None):
    """(startAt, endAt, speaker) 목록 (ms). 규정상 startAt/endAt/speaker 만 사용."""
    out = []
    for u in (label or {}).get("utterances") or []:
        s, e, spk = u.get("startAt"), u.get("endAt"), u.get("speaker")
        if s is None or e is None or e <= s:
            continue
        if speaker is not None and str(spk) != str(speaker):
            continue
        out.append((int(s), int(e), spk))
    return out


def frame_features(wav_path):
    """통화 전체에 대해 프레임 단위 특징 계산 (한 번만 계산하고 구간별로 잘라 씀)."""
    wav, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    if len(wav) < SR // 2:
        wav = np.pad(wav, (0, SR // 2 - len(wav)))
    mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=HOP)
    f0 = librosa.yin(wav, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    rms = librosa.feature.rms(y=wav, frame_length=512, hop_length=HOP)[0]
    T = min(mfcc.shape[1], len(f0), len(rms))
    return mfcc[:, :T], f0[:T], rms[:T]


def ms_to_frames(start_ms, end_ms, T):
    a = max(0, int(start_ms / 1000 * SR / HOP))
    b = min(T, int(end_ms / 1000 * SR / HOP) + 1)
    return np.arange(a, max(a + 1, b))


def pool(feats, idx):
    """선택한 프레임들의 통계 특징 벡터 (크기 고정)."""
    mfcc, f0, rms = feats
    idx = idx[idx < mfcc.shape[1]]
    if len(idx) == 0:
        idx = np.arange(min(1, mfcc.shape[1]))
    m, f, r = mfcc[:, idx], f0[idx], rms[idx]
    d = np.diff(m, axis=1) if m.shape[1] > 1 else np.zeros((m.shape[0], 1))
    voiced = f[r >= np.percentile(r, 40)] if len(r) > 2 else f      # 에너지 큰(유성) 프레임의 F0
    fq = np.percentile(voiced, [10, 25, 50, 75, 90])
    return np.concatenate([
        m.mean(1), m.std(1), np.abs(d).mean(1),                       # MFCC 평균/표준편차/변화량
        fq, [voiced.std(), np.log(fq[2])],                            # F0 분위수 (성별·화자 단서)
        [r.mean(), r.std(), np.log1p(len(idx) * HOP / SR)],           # 에너지, 길이(초)
    ]).astype(np.float32)


FEAT_DIM = N_MFCC * 3 + 5 + 2 + 3
