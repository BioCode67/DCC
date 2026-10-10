"""
제출 CSV 채점 (Validation 정답 json 과 비교)

python tools/evaluate.py --mission 1 --pred outputs/mission1.csv --label_dir {json folder}
  Mission 1, 2: Accuracy / Mission 3: 증상별 F1 → Macro 평균
"""
import argparse
import ast
import csv
import json
from pathlib import Path

SYMPTOMS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]


def labels(label_dir):
    out = {}
    for p in Path(label_dir).rglob("*.json"):
        with open(p, encoding="utf-8-sig") as f:
            out[p.stem] = json.load(f)
    return out


def rows(path):
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        return list(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label_dir", required=True)
    a = ap.parse_args()
    gt = labels(a.label_dir)
    pred = rows(a.pred)

    if a.mission == 1:
        ok = [str(gt[Path(n).stem].get("gender")) == g for n, g in pred if Path(n).stem in gt]
        print(f"Mission 1 Accuracy {sum(ok) / len(ok):.4f}  (n={len(ok)})")
    elif a.mission == 2:
        ref = {}
        for stem, lab in gt.items():
            for u in lab.get("utterances") or []:
                ref[(stem, int(u["startAt"]), int(u["endAt"]))] = str(u.get("speaker"))
        ok = [ref[k] == spk for n, s, e, spk in pred
              if (k := (Path(n).stem, int(s), int(e))) in ref]
        print(f"Mission 2 Accuracy {sum(ok) / len(ok):.4f}  (n={len(ok)} / 정답 조각 {len(ref)})")
    else:
        stats = {c: [0, 0, 0] for c in SYMPTOMS}          # tp, fp, fn
        for n, s in pred:
            lab = gt.get(Path(n).stem)
            if lab is None:
                continue
            t = {x for x in (lab.get("symptom") or []) if x in SYMPTOMS}
            p = {x for x in ast.literal_eval(s) if x in SYMPTOMS}
            for c in SYMPTOMS:
                stats[c][0] += (c in t) and (c in p)
                stats[c][1] += (c not in t) and (c in p)
                stats[c][2] += (c in t) and (c not in p)
        f1 = {c: (2 * tp / (2 * tp + fp + fn) if tp else 0.0) for c, (tp, fp, fn) in stats.items()}
        print("Mission 3 Macro F1 {:.4f}".format(sum(f1.values()) / len(f1)))
        for c in SYMPTOMS:
            print(f"  {c:6s} F1 {f1[c]:.4f}  (tp/fp/fn {stats[c]})")


if __name__ == "__main__":
    main()
