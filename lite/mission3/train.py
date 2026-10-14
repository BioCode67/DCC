"""
Mission 3 학습: 대화 전사 텍스트 → 환자 증상 9개 multi-label 분류 (CPU 경량 버전)

입력 : 라벨 json 의 utterances[*].text (전사 텍스트만 사용)
방법 : 문자 n-gram(2~4) TF-IDF + 증상별 Logistic Regression (class_weight='balanced')
판정 : 증상별 확률 >= 0.5 (Threshold 0.5 고정, 모든 클래스 공통)
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SYMPTOMS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
THRESHOLD = 0.5


def read_label(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def dialogue_text(label):
    return " ".join(str(u.get("text") or "").strip() for u in label.get("utterances") or [])


def target_vector(label):
    sym = label.get("symptom") or []
    sym = {str(x).strip() for x in ([sym] if isinstance(sym, str) else sym)}
    return [int(c in sym) for c in SYMPTOMS]          # 9개 외 증상은 제외


def load_dir(label_dir):
    paths = sorted(Path(label_dir).rglob("*.json"))
    with ThreadPoolExecutor(16) as ex:
        labs = list(ex.map(read_label, paths))
    return paths, labs


def macro_f1(Y, P):
    f1s = []
    for c in range(Y.shape[1]):
        tp = int(((Y[:, c] == 1) & (P[:, c] == 1)).sum())
        fp = int(((Y[:, c] == 0) & (P[:, c] == 1)).sum())
        fn = int(((Y[:, c] == 1) & (P[:, c] == 0)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)), f1s


def predict(ck, texts):
    T = ck["vectorizer"].transform(texts)
    P = np.zeros((len(texts), len(SYMPTOMS)), dtype=int)
    for c, clf in enumerate(ck["models"]):
        if clf is not None:
            P[:, c] = clf.predict_proba(T)[:, 1] >= THRESHOLD
    return P


def fit(texts, Y):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=200000,
                          sublinear_tf=True)
    T = vec.fit_transform(texts)
    models = []
    for c in range(len(SYMPTOMS)):
        if Y[:, c].sum() == 0:
            models.append(None)
            continue
        clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
        clf.fit(T, Y[:, c])
        models.append(clf)
    return {"vectorizer": vec, "models": models, "symptoms": SYMPTOMS}


def main(args, log=print):
    t0 = time.time()
    _, labs = load_dir(args.train_label_dir)
    texts = [dialogue_text(l) for l in labs]
    Y = np.array([target_vector(l) for l in labs])
    log(f"학습 문서 {len(texts)} ({time.time() - t0:.0f}s 로딩)")
    log("증상별 양성 수: " + ", ".join(f"{c} {int(n)}" for c, n in zip(SYMPTOMS, Y.sum(0))))

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(texts))
    n_ho = max(1, len(texts) // 10)
    ho, tr = perm[:n_ho], perm[n_ho:]
    ck = fit([texts[i] for i in tr], Y[tr])
    f1, per = macro_f1(Y[ho], predict(ck, [texts[i] for i in ho]))
    log(f"holdout Macro F1 {f1:.4f} | " + " ".join(f"{c}:{x:.2f}" for c, x in zip(SYMPTOMS, per)))

    ck = fit(texts, Y)                               # 전체 Training 으로 재학습
    ck["holdout_f1"] = f1
    joblib.dump(ck, args.ckpt_path)
    log(f"saved → {args.ckpt_path}  ({time.time() - t0:.0f}s)")
    return {"holdout": f1, "n_train": len(texts), "train_sec": time.time() - t0,
            "class_counts": Y.sum(0).tolist()}


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_label_dir", required=True)
    p.add_argument("--ckpt_path", default="mission3_model.pkl")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
