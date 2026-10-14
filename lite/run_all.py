"""
DCC 대학부 예선 제출물 일괄 생성 (CPU 경량 버전)

python run_all.py --data_root "C:\\...\\대학부 데이터"

1) 데이터 폴더(TS_/TL_/VS_/VL_) 자동 탐색
2) 미션별 제출 폴더 구성 (코드 + requirements.txt)
3) 미션별 학습 노트북 실행 → 학습 로그가 담긴 train_missionN.ipynb + 가중치(.pkl)
4) Validation 에 대해 제출 명령어(inference.py)로 추론 → 채점 + 추론 시간 측정
5) 데이터 통계 + 결과로 PPT(200OK_결과정리.pptx) 생성
6) 200OK.zip 생성
"""
import argparse
import ast
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYMPTOMS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
REQ_AUDIO = "numpy>=1.24\nscikit-learn>=1.3\nlibrosa>=0.10\nsoundfile>=0.12\njoblib>=1.3\n"
REQ_TEXT = "numpy>=1.24\nscikit-learn>=1.3\njoblib>=1.3\n"


def find_dir(root, prefix, max_depth=4):
    root = Path(root)
    for dirpath, dirnames, _ in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        for d in dirnames:
            if d.startswith(prefix):
                return str(Path(dirpath) / d)
        if depth >= max_depth:
            dirnames[:] = []
    return None


def read_json(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


# ------------------------------------------------------------------ 데이터 통계
def data_stats(label_dir):
    paths = sorted(Path(label_dir).rglob("*.json"))
    g, spk, sym, dur, n_utt, n_sym = Counter(), Counter(), Counter(), [], [], Counter()
    for p in paths:
        d = read_json(p)
        g[str(d.get("gender"))] += 1
        us = d.get("utterances") or []
        n_utt.append(len(us))
        for u in us:
            spk[str(u.get("speaker"))] += 1
        if us:
            dur.append((max(u.get("endAt") or 0 for u in us)) / 1000)
        s = [x for x in (d.get("symptom") or []) if x in SYMPTOMS]
        n_sym[len(s)] += 1
        for x in s:
            sym[x] += 1
    avg = lambda v: sum(v) / max(1, len(v))
    return {"n_calls": len(paths), "gender": dict(g), "speaker": dict(spk),
            "symptom": {c: sym.get(c, 0) for c in SYMPTOMS}, "n_symptom": dict(sorted(n_sym.items())),
            "avg_dur_sec": avg(dur), "avg_utts": avg(n_utt)}


# ------------------------------------------------------------------ 채점
def score(mission, pred_csv, label_dir):
    gt = {p.stem: read_json(p) for p in Path(label_dir).rglob("*.json")}
    with open(pred_csv, encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    if mission == 1:
        ok = [str(gt[Path(n).stem].get("gender")) == g for n, g in rows if Path(n).stem in gt]
        return {"metric": "Accuracy", "score": sum(ok) / max(1, len(ok)), "n": len(ok)}
    if mission == 2:
        ref = {(s, int(u["startAt"]), int(u["endAt"])): str(u.get("speaker"))
               for s, d in gt.items() for u in d.get("utterances") or []}
        ok = [ref[k] == v for n, a, b, v in rows if (k := (Path(n).stem, int(a), int(b))) in ref]
        return {"metric": "Accuracy", "score": sum(ok) / max(1, len(ok)), "n": len(ok)}
    st = {c: [0, 0, 0] for c in SYMPTOMS}
    for n, s in rows:
        d = gt.get(Path(n).stem)
        if d is None:
            continue
        t = {x for x in d.get("symptom") or [] if x in SYMPTOMS}
        p = {x for x in ast.literal_eval(s) if x in SYMPTOMS}
        for c in SYMPTOMS:
            st[c][0] += (c in t) and (c in p)
            st[c][1] += (c not in t) and (c in p)
            st[c][2] += (c in t) and (c not in p)
    per = {c: (2 * a / (2 * a + b + e) if a else 0.0) for c, (a, b, e) in st.items()}
    return {"metric": "Macro F1", "score": sum(per.values()) / len(per), "per_class": per, "n": len(rows)}


# ------------------------------------------------------------------ 학습 노트북 실행
def make_and_run_notebook(mdir, n, train_args, log):
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(f"# Mission {n} 학습 로그\n\nCPU 경량 버전. `run_all.py` 가 실행하여 생성한 노트북입니다."),
        nbformat.v4.new_code_cell("import platform, os, sys, time\n"
                                  "print(platform.platform(), '|', platform.processor(), '| CPU cores', os.cpu_count())\n"
                                  "print('python', sys.version.split()[0])"),
        nbformat.v4.new_code_cell(f"import train\nargs = train.get_parser().parse_args({train_args!r})\n"
                                  "t0 = time.time()\nresult = train.main(args)\n"
                                  "print('결과', result)\nprint(f'총 학습 시간 {time.time()-t0:.0f}s')"),
    ]
    path = mdir / f"train_mission{n}.ipynb"
    try:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=None, kernel_name="python3",
                       resources={"metadata": {"path": str(mdir)}}).execute()
        nbformat.write(nb, path)
        log(f"  학습 노트북 실행 완료 → {path.name}")
        return True
    except Exception as e:
        log(f"  [notebook 실행 실패 → 스크립트로 대체] {e}")
        with open(mdir / "train_log.txt", "w", encoding="utf-8") as f:
            r = subprocess.run([sys.executable, "train.py", *train_args], cwd=mdir,
                               stdout=f, stderr=subprocess.STDOUT)
        nbformat.write(nb, path)              # 실행 안 된 노트북이라도 코드는 남김
        return r.returncode == 0


def run_inference(mdir, n, audio_dir, label_dir, ckpt, log):
    out = f"./outputs/mission{n}.csv"
    cmd = [sys.executable, "inference.py", "--audio_dir", audio_dir or "", "--label_dir", label_dir,
           "--ckpt_path", ckpt, "--output", out]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=mdir, capture_output=True, text=True, encoding="utf-8", errors="replace")
    sec = time.time() - t0
    log("  " + (r.stdout.strip().splitlines() or [""])[-1])
    if r.returncode != 0:
        log(r.stderr[-2000:])
        raise RuntimeError(f"mission{n} inference 실패")
    return mdir / out, sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Training / Validation 폴더가 들어있는 상위 폴더")
    ap.add_argument("--out", default=str(HERE / "submission"))
    ap.add_argument("--team", default="200OK")
    ap.add_argument("--m1_limit", type=int, default=3000)
    ap.add_argument("--m2_limit", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--missions", default="3,1,2", help="실행할 미션 (빠른 3번부터)")
    a = ap.parse_args()

    out_root = Path(a.out) / a.team
    out_root.mkdir(parents=True, exist_ok=True)
    logf = open(Path(a.out) / "run_all_log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    dirs = {k: find_dir(a.data_root, k) for k in ("TS_", "TL_", "VS_", "VL_")}
    for k, v in dirs.items():
        log(f"{k:4s} {v}")
    if not dirs["TL_"] or not dirs["VL_"]:
        sys.exit("라벨 폴더(TL_/VL_)를 찾지 못했습니다. --data_root 를 확인하세요.")

    results_path = Path(a.out) / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    results["env"] = {"platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
                      "cores": os.cpu_count(), "python": sys.version.split()[0]}
    if "stats" not in results:
        log("데이터 통계 계산 중 ...")
        results["stats"] = {"train": data_stats(dirs["TL_"]), "val": data_stats(dirs["VL_"])}
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    for n in [int(x) for x in a.missions.split(",")]:
        log(f"\n===== Mission {n} =====")
        mdir = out_root / f"mission{n}"
        mdir.mkdir(exist_ok=True)
        for f in (HERE / f"mission{n}").glob("*.py"):
            shutil.copy(f, mdir / f.name)
        (mdir / "requirements.txt").write_text(REQ_TEXT if n == 3 else REQ_AUDIO, encoding="utf-8")
        ckpt = f"mission{n}_model.pkl"
        if n == 3:
            targs = ["--train_label_dir", dirs["TL_"], "--ckpt_path", ckpt]
        else:
            targs = ["--train_audio_dir", dirs["TS_"], "--train_label_dir", dirs["TL_"], "--ckpt_path", ckpt,
                     "--limit", str(a.m1_limit if n == 1 else a.m2_limit), "--workers", str(a.workers)]
        t0 = time.time()
        if not make_and_run_notebook(mdir, n, targs, log) or not (mdir / ckpt).exists():
            log(f"Mission {n} 학습 실패 — 건너뜀")
            continue
        train_sec = time.time() - t0
        pred, inf_sec = run_inference(mdir, n, dirs["VS_"], dirs["VL_"], ckpt, log)
        sc = score(n, pred, dirs["VL_"])
        import joblib
        ck = joblib.load(mdir / ckpt)
        sc.update({"holdout": ck.get("holdout_acc", ck.get("holdout_f1")), "train_sec": train_sec,
                   "infer_sec": inf_sec, "ckpt_mb": (mdir / ckpt).stat().st_size / 1e6})
        results[f"mission{n}"] = sc
        log(f"Mission {n} Validation {sc['metric']} = {sc['score']:.4f}  (추론 {inf_sec:.0f}s)")
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    from make_ppt import build
    ppt = out_root / f"{a.team}_결과정리.pptx"
    build(results, ppt, a.team)
    log(f"\nPPT → {ppt}")
    shutil.copy(HERE / "SUBMISSION_README.md", out_root / "README.md")
    for d in out_root.glob("mission*/__pycache__"):
        shutil.rmtree(d, ignore_errors=True)
    z = shutil.make_archive(str(Path(a.out) / a.team), "zip", root_dir=Path(a.out), base_dir=a.team)
    log(f"ZIP → {z}  ({Path(z).stat().st_size / 1e6:.1f} MB)")
    log("완료! 업로드 전에 zip 을 열어 mission1~3 폴더와 pptx 가 들어있는지 확인하세요.")


if __name__ == "__main__":
    main()
