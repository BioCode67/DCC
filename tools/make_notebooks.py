"""미션별 Colab 학습 노트북(mission{N}/train_mission{N}.ipynb) 생성 스크립트.

python tools/make_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

REPO = "https://github.com/BioCode67/DCC.git"
BRANCH = "ccr-1cdc76be-ibbw6s"

SETUP = f'''# ① GPU 확인 + Google Drive 연결 + 코드 받기
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')

import os
if not os.path.exists('/content/DCC'):
    !git clone -q -b {BRANCH} {REPO} /content/DCC
else:
    !git -C /content/DCC pull -q
'''

FIND_DATA = r'''# ② 데이터 폴더 찾기
#   공유 폴더 '대학부 데이터' 를 내 드라이브에 '바로가기 추가' 해 둔 경로. 다르면 DATA_ROOT 만 고쳐주세요.
import glob, os, zipfile
DATA_ROOT = '/content/drive/MyDrive/2026 공모전/데이터+AI 크리에이터 캠프/대학부 데이터'
if not os.path.exists(DATA_ROOT):
    cands = glob.glob('/content/drive/MyDrive/*/*/대학부 데이터') + glob.glob('/content/drive/MyDrive/*/대학부 데이터')
    DATA_ROOT = cands[0] if cands else DATA_ROOT
print('DATA_ROOT =', DATA_ROOT, os.path.exists(DATA_ROOT))

def find_dir(prefix):
    """TS_/TL_/VS_/VL_ 로 시작하는 폴더(또는 zip) 경로 반환. (Drive 는 재귀 탐색이 매우 느려서 깊이 3까지만 확인)
    zip 이면 로컬 디스크에 압축 해제."""
    hits = []
    for depth in ('', '*/', '*/*/', '*/*/*/'):
        hits += sorted(glob.glob(f'{DATA_ROOT}/{depth}{prefix}*'))
    for h in hits:
        if os.path.isdir(h):
            return h
    for h in hits:
        if h.endswith('.zip'):
            dst = f'/content/data/{os.path.basename(h)[:-4]}'
            if not os.path.exists(dst):
                print('unzip', h); zipfile.ZipFile(h).extractall(dst)
            return dst
    return None

TRAIN_WAV, TRAIN_JSON = find_dir('TS_'), find_dir('TL_')
VAL_WAV, VAL_JSON = find_dir('VS_'), find_dir('VL_')
for k, v in dict(TRAIN_WAV=TRAIN_WAV, TRAIN_JSON=TRAIN_JSON, VAL_WAV=VAL_WAV, VAL_JSON=VAL_JSON).items():
    sample = sorted(os.listdir(v))[:3] if v else []
    print(f'{k:10s} {v}\n           예시: {sample}')
'''

SAVE = '''# ③ 설정
#  - SAVE_DIR: 체크포인트/결과 저장 위치. 기본은 Colab 로컬 디스크(드라이브 용량 사용 안 함).
#              런타임이 끊기면 사라지므로 마지막 셀에서 꼭 다운로드할 것.
#              드라이브 여유가 1GB 이상이면 '/content/drive/MyDrive/DCC_outputs' 로 바꿔도 됨.
#  - QUICK_TEST: 처음엔 200 으로 전체 흐름만 확인 → 문제 없으면 0 으로 바꿔 본 학습
#  - LIMIT: 학습에 쓸 통화 수 (0=전체). 무료 Colab(T4, 시간 제한) 기준 권장값이 기본으로 들어가 있음.
SAVE_DIR = '/content/ckpt'
os.makedirs(SAVE_DIR, exist_ok=True)
QUICK_TEST = 200
LIMIT = {limit}
N_LIMIT = QUICK_TEST or LIMIT
WORKERS = str(min(8, os.cpu_count()))
'''

DOWNLOAD = '''# [마지막] 결과 다운로드 (런타임 종료 전에 꼭 실행!) — 체크포인트 + 결과 CSV 를 zip 으로 내려받기
import shutil
from google.colab import files
zip_path = shutil.make_archive('/content/' + os.path.basename(os.getcwd()) + '_outputs', 'zip', SAVE_DIR)
files.download(zip_path)
'''

INSTALL = '!pip -q install -r /content/DCC/mission{n}/requirements.txt'


def audio_nb(n, title, extra_args, limit):
    return [
        nbf.v4.new_markdown_cell(f"# Mission {n}: {title}\n\n"
                                 "Colab(GPU) 에서 위에서부터 순서대로 실행하세요. 실행 로그가 그대로 학습 로그로 제출됩니다."),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell(INSTALL.format(n=n)),
        nbf.v4.new_code_cell(FIND_DATA),
        nbf.v4.new_code_cell(SAVE.format(limit=limit)),
        nbf.v4.new_code_cell(f'''# ④ 학습 (WavLM 특징 추출 → MLP 헤드 학습, Training 폴더만 사용)
import sys, time; sys.path.insert(0, '/content/DCC/mission{n}'); os.chdir('/content/DCC/mission{n}')
import train
args = train.get_parser().parse_args([
    '--train_audio_dir', TRAIN_WAV, '--train_label_dir', TRAIN_JSON,
    '--ckpt_path', f'{{SAVE_DIR}}/mission{n}.pt', '--cache_dir', '/content/cache',
    '--num_workers', WORKERS, '--limit', str(N_LIMIT){extra_args}])
t0 = time.time(); train.main(args); print(f'총 학습 시간 {{time.time()-t0:.0f}}s')
'''),
        nbf.v4.new_code_cell(f'''# ⑤ Validation 추론 (제출 명령어와 동일) + 채점 + 추론 시간 측정
!cd /content/DCC/mission{n} && time python inference.py --audio_dir "$VAL_WAV" --label_dir "$VAL_JSON" \\
    --ckpt_path "$SAVE_DIR/mission{n}.pt" --output ./outputs/mission{n}.csv --num_workers $WORKERS
!python /content/DCC/tools/evaluate.py --mission {n} --pred /content/DCC/mission{n}/outputs/mission{n}.csv --label_dir "$VAL_JSON"
!cp /content/DCC/mission{n}/outputs/mission{n}.csv "$SAVE_DIR/"
'''),
        nbf.v4.new_code_cell(DOWNLOAD),
    ]


def text_nb():
    n = 3
    return [
        nbf.v4.new_markdown_cell("# Mission 3: 환자의 증상 인식 (대화 텍스트 → 9개 증상 multi-label)\n\n"
                                 "Colab(GPU) 에서 위에서부터 순서대로 실행하세요. 실행 로그가 그대로 학습 로그로 제출됩니다."),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell(INSTALL.format(n=n)),
        nbf.v4.new_code_cell(FIND_DATA),
        nbf.v4.new_code_cell(SAVE.format(limit=0)),
        nbf.v4.new_code_cell('''# ④ (참고용 베이스라인) TF-IDF + Logistic Regression — CPU 로 수 분, PPT 비교용
import sys; sys.path.insert(0, '/content/DCC/mission3'); os.chdir('/content/DCC/mission3')
!python baseline_tfidf.py --train_label_dir "$TRAIN_JSON" --val_label_dir "$VAL_JSON"
'''),
        nbf.v4.new_code_cell('''# ⑤ 학습 (klue/roberta-base fine-tuning, Training 폴더만 사용)
import sys, time; sys.path.insert(0, '/content/DCC/mission3'); os.chdir('/content/DCC/mission3')
import train
args = train.get_parser().parse_args([
    '--train_label_dir', TRAIN_JSON, '--ckpt_path', f'{SAVE_DIR}/mission3/mission3.pt',
    '--limit', str(N_LIMIT),
    '--epochs', '3', '--max_windows', '3'])   # 무료 T4 기준. Colab Pro(A100/L4)면 epochs 4, max_windows 4 권장
t0 = time.time(); train.main(args); print(f'총 학습 시간 {time.time()-t0:.0f}s')
'''),
        nbf.v4.new_code_cell('''# ⑥ Validation 추론 (제출 명령어와 동일) + 채점 + 추론 시간 측정
!cd /content/DCC/mission3 && time python inference.py --audio_dir "$VAL_WAV" --label_dir "$VAL_JSON" \\
    --ckpt_path "$SAVE_DIR/mission3/mission3.pt" --output ./outputs/mission3.csv
!python /content/DCC/tools/evaluate.py --mission 3 --pred /content/DCC/mission3/outputs/mission3.csv --label_dir "$VAL_JSON"
!cp /content/DCC/mission3/outputs/mission3.csv "$SAVE_DIR/"
'''),
        nbf.v4.new_code_cell(DOWNLOAD),
    ]


def write(cells, path):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "L4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    nbf.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    write(audio_nb(1, "신고자의 성별 분류 (음성 → gender)", "", 8000), root / "mission1/train_mission1.ipynb")
    write(audio_nb(2, "신고자 / 119대원 발화 분류 (발화 조각 음성 → speaker)", "", 6000), root / "mission2/train_mission2.ipynb")
    write(text_nb(), root / "mission3/train_mission3.ipynb")
