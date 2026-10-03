# Step 00 — Project, environment, hardware

**Status:** complete. All Step 0 checks pass.

## What was done

1. Created the project tree under `adni-shortcut/`:
   `data/raw/`, `src/adni/`, `scripts/`, `tests/`, `configs/`, `results/`, `logs/`, `docs/`.
2. Copied the six ADNI CSVs into `data/raw/` with the exact expected filenames and made them
   **read-only** (`chmod a-w`). Copied `adni_stage1_build.py`, `adni_stage3_profile.py` → `scripts/`;
   `PIPELINE_v2.md`, `episodes.csv`, `joint_profile.csv`, `feature_spec.json` → `docs/`.
3. Created a Python **3.12.10** virtual environment at `.venv/` (`py -3.12 -m venv .venv`).
4. Installed the full stack: numpy pandas scipy scikit-learn torch lightgbm statsmodels matplotlib
   pyyaml tqdm joblib pytest pyarrow. This machine has an NVIDIA GPU, so the **CUDA** build of torch
   was installed (`torch 2.11.0+cu128`) rather than the CPU wheel that a plain `pip install torch`
   gives. `pip freeze > requirements.lock.txt`.
5. `git init`, wrote `.gitignore` (ignores `.venv/`, `__pycache__/`, `data/raw/`, `results/*/artifacts`,
   `results/**/blocks`, `*.pt`), first commit.
6. Wrote `scripts/env_report.py` (OS, Python, CPU, RAM, GPU/VRAM, torch version,
   `torch.cuda.is_available()`, SHA256 of every `data/raw/` file, read-only flag).
7. Ran `pytest -q` → `no tests ran in 0.02s` (expected, no tests yet) and `env_report.py`.

## Environment report

```
OS: Windows 11 (10.0.26300)
machine: AMD64
Python version: 3.12.10 (CPython)
Python executable: C:\D\5th Sem\ADNI NEW\adni-shortcut\.venv\Scripts\python.exe
CPU count (logical): 12
RAM: 15.3 GiB
torch version: 2.11.0+cu128
torch.cuda.is_available(): True
CUDA device count: 1
  GPU 0: NVIDIA GeForce RTX 3050 6GB Laptop GPU  VRAM=6.0 GiB  (capability 8.6)
------------------------------------------------------------------------
SHA256 of files in data\raw:
  DATADIC_03Oct2026.csv                         acf1b0c435731a7b39c33803b47334dba0b35a9d00bc460c32cdffef1b5c6a5d  readonly=True
  DXSUM.csv                                     967b0fa351c0ad26a5dbebd8e921d850f10c0225d47380a9ab753f3bcb933fd2  readonly=True
  UCBERKELEY_AMY_6MM_03Oct2026.csv              177e839ba3cdf41587af6e19fca001ee6eca93a554ec0ea29b9766119cf95499  readonly=True
  UCBERKELEY_TAU_6MM_03Oct2026.csv              5d5061df051e044796f9974e33274c57edf69d14aad40d8ca2a234ca01bda3c0  readonly=True
  UCBERKELEYFDG_8mm_02_17_23_03Oct2026.csv      1d1897f3f0a93163f9e1090ba7f0fba7c365b20d53d08b1be1c8c6e2d31992fa  readonly=True
  UCSFFSX7_03Oct2026.csv                        e00524a1799175188efcd0131a24f90d59bf819978666116f8819b6f717b8e16  readonly=True
```

## `pip freeze` (first 20 lines)

```
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.4.0
cycler==0.12.1
filelock==4.0.9
fonttools==4.66.1
formulaic==1.2.2
fsspec==2026.9.0
iniconfig==2.3.0
interface_meta==2.0.1
Jinja2==3.1.6
joblib==1.6.0
kiwisolver==1.5.1
lightgbm==4.7.0
MarkupSafe==3.0.4
matplotlib==3.11.2
mpmath==1.3.0
narwhals==2.26.0
networkx==3.7
numpy==2.5.3
```

## Key library versions

| Library | Version |
|---|---|
| Python | 3.12.10 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| torch | 2.11.0+cu128 |
| lightgbm | 4.7.0 |
| statsmodels | 0.15.0 |
| matplotlib | 3.11.2 |
| pyarrow | 25.0.1 |
| pytest | 9.1.1 |

## Checks

| Check | Result |
|---|---|
| venv active (`.venv`, Python 3.12.10) | PASS |
| torch imports, CUDA available | PASS (True, RTX 3050) |
| raw files read-only | PASS (all 6 `readonly=True`) |
| hashes printed | PASS (6 SHA256 values above) |
| `pytest -q` | `no tests ran in 0.02s` (expected) |

## Deviations / notes for the user

- **Raw data location.** The six ADNI CSVs were not in the playbook folder; I located them under
  `~/Downloads/` (`DXSUM.csv` in `~/Downloads`, `UCSFFSX7_03Oct2026.csv` in `~/Downloads/download (1)/`,
  the three Berkeley tables + `DATADIC` in `~/Downloads/download/`) and copied them into `data/raw/`.
  Filenames match the playbook exactly.
- **torch build.** A plain `pip install torch` on this machine yields the CPU wheel
  (`2.14.1+cpu`). Because an RTX 3050 is present, I reinstalled the CUDA build
  (`2.11.0+cu128`) from the PyTorch cu128 index. CUDA matmul verified.
- **GPU VRAM is small (6 GiB).** Fine for this pipeline (tabular encoders + 2-layer Transformer,
  batch 256), but worth keeping in mind for Step 7+.

STEP 00 REPORT
Command(s) run: `py -3.12 -m venv .venv`; `pip install <stack>`; `pip install torch --index-url .../cu128 --force-reinstall --no-deps`; `pip freeze > requirements.lock.txt`; `pytest -q`; `python scripts/env_report.py`
Git commit: step00 commit (see `git log`)
Tests: `no tests ran in 0.02s`
Expected vs actual: all Step 0 checks pass (see table above)
Deviations from the spec: raw data sourced from `~/Downloads/`; CUDA torch build chosen over the default CPU wheel
Things I am unsure about: none blocking
Files written: `adni-shortcut/` tree, `.gitignore`, `requirements.lock.txt`, `scripts/env_report.py`, `logs/step00_env_report.txt`, `results/step00/REPORT.md`
