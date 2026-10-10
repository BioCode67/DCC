"""
Mission 3 참고용 베이스라인: 문자 n-gram TF-IDF + 클래스별 Logistic Regression
(CPU 로 수 분 내 실행. 최종 모델(KLUE-RoBERTa)과 비교하기 위한 기준선, Threshold 0.5 고정)

python baseline_tfidf.py --train_label_dir {TL json folder} --val_label_dir {VL json folder}
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from text_model import SYMPTOMS, THRESHOLD, dialogue_text, macro_f1, read_label, target_vector


def load(label_dir):
    with ThreadPoolExecutor(32) as ex:             # Drive 에서 병렬 읽기
        labs = list(ex.map(read_label, sorted(Path(label_dir).rglob("*.json"))))
    return [dialogue_text(l) for l in labs], np.array([target_vector(l) for l in labs])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_label_dir", required=True)
    p.add_argument("--val_label_dir", required=True)
    a = p.parse_args()

    Xtr, Ytr = load(a.train_label_dir)
    Xva, Yva = load(a.val_label_dir)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=300000,
                          sublinear_tf=True)
    Ttr, Tva = vec.fit_transform(Xtr), vec.transform(Xva)

    pred = np.zeros_like(Yva)
    for c in range(len(SYMPTOMS)):
        if Ytr[:, c].sum() == 0:
            continue
        clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
        clf.fit(Ttr, Ytr[:, c])
        pred[:, c] = clf.predict_proba(Tva)[:, 1] >= THRESHOLD
    f1, per = macro_f1(Yva, pred)
    print(f"[TF-IDF baseline] Validation Macro F1 {f1:.4f}")
    for c, x in zip(SYMPTOMS, per):
        print(f"  {c:6s} {x:.4f}")


if __name__ == "__main__":
    main()
