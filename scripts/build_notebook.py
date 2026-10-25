"""Build notebooks/generate.ipynb: a self-contained Kaggle / Colab notebook.

The notebook embeds the vcascade package source (one %%writefile cell per
module), so the only thing to upload is the .ipynb. Rebuild after editing src/:

    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "vcascade"
# Router-side modules are not needed on the GPU box.
MODULES = ["__init__.py", "data.py", "prompts.py", "grading.py", "backends.py", "retrieval.py", "generate.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
# Generate the cache for context-free cascades

Runs the small model (SLM), the large model (LLM) and every verifier once over a mixed
stream of GSM8K and TriviaQA questions, then grades everything into `cache.jsonl`.
The router experiments run offline from that file.

**Kaggle:** Settings → Accelerator **GPU T4 x2** (vLLM does not support the P100) and
**Internet on** (needs a phone-verified account). Then *Save Version → Save & Run All* so it
runs unattended and the outputs persist.
**Colab:** Runtime → Change runtime type → **T4 GPU**.

Set `PILOT = True` first (200 queries, about 10 minutes) to check everything works, then
`PILOT = False` for the full run. Every stage resumes if the session drops.
Download `cache.zip` (Kaggle: Output tab; Colab: the file browser) when it finishes.
"""

CONFIG = """
PILOT = True                                   # 100 math + 100 factual; False = 1,319 + 1,500
SLM = "Qwen/Qwen2.5-1.5B-Instruct"
LLM = "Qwen/Qwen2.5-7B-Instruct-AWQ"           # 4-bit, fits one T4 alongside nothing else
BACKEND = "vllm"                               # "hf" = plain transformers fallback (much slower)
DTYPE = "half"                                 # T4 has no bfloat16
STAGES = ["prepare", "slm", "retrieve", "verify", "llm", "embed", "merge"]
WIKI_CONTACT = ""                              # your email or URL: Wikipedia allows 200 req/min with it, 10 without

import os, subprocess
cap = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()
print("GPU:", cap)
NEEDS_VLLM = BACKEND == "vllm" and any(s in STAGES for s in ("slm", "verify", "llm"))
if NEEDS_VLLM and (not cap or float(cap[0].split(",")[-1]) < 7.0):
    raise SystemExit("vLLM needs a T4 or newer GPU (compute capability >= 7.0). "
                     "Kaggle: Settings -> Accelerator -> GPU T4 x2. Colab: Runtime -> T4 GPU.")
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
OUT = f"{WORK}/cache_pilot" if PILOT else f"{WORK}/cache"
LIMITS = "--limit-math 100 --limit-factual 100" if PILOT else "--limit-factual 1500"
os.chdir(WORK)
os.environ["PYTHONPATH"] = f"{WORK}/src"
os.environ["WIKI_CONTACT"] = WIKI_CONTACT

# Outputs of earlier stages attached as a Kaggle dataset are copied in; those stages then skip.
import glob, shutil
os.makedirs(OUT, exist_ok=True)
STAGE_FILES = ["queries.jsonl", "slm.jsonl", "retrieval.jsonl", "verify.jsonl", "llm.jsonl",
               "embeddings.npy", "embeddings_qids.json", "cache.jsonl", "hidden.npz"]
for path in glob.glob("/kaggle/input/**/*", recursive=True):
    name = os.path.basename(path)
    if name in STAGE_FILES and not os.path.exists(f"{OUT}/{name}"):
        shutil.copy(path, f"{OUT}/{name}")
        print("using input", path)
print(OUT, STAGES)
"""

INSTALL = """
import subprocess, sys
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
pip = f"{sys.executable} -m pip"
if NEEDS_VLLM:
    subprocess.run(f"{pip} install -q vllm sentence-transformers datasets requests", shell=True, check=True)
    # vLLM brings its own torch build; the image's torchaudio no longer matches it and breaks
    # `import transformers`. Nothing here needs torchaudio.
    subprocess.run(f"{pip} uninstall -y -q torchaudio", shell=True)
else:  # hidden-state stage only: the image's torch + transformers suffice
    subprocess.run(f"{pip} install -q sentence-transformers datasets requests", shell=True, check=True)
"""

CHECK = """
import subprocess, sys
probe = ("import vllm; print('vllm', vllm.__version__); " if NEEDS_VLLM else "") +     "import torch, transformers, sentence_transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
print(r.stdout, r.stderr[-3000:])
assert r.returncode == 0, "environment check failed, see above"

os.environ["TQDM_MININTERVAL"] = "30"   # keep progress bars from flooding the log

def run(args):
    \"\"\"Run one pipeline stage, stream its output, and stop the notebook if it fails.\"\"\"
    cmd = f"{sys.executable} -m vcascade.generate --out {OUT} {args}"
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"stage failed: {args}")
"""

RUN = [
    ("Build the query stream", "prepare", 'run(f"prepare {LIMITS}")'),
    ("SLM: greedy answer with token log-probs (V1) + 8 samples (V2)", "slm",
     'run(f"slm --model {SLM} --backend {BACKEND} --dtype {DTYPE}")'),
    ("Retrieve Wikipedia passages for V3 (CPU, needs internet)", "retrieve", 'run("retrieve")'),
    ("SLM as verifier: retrieval-grounded (V3) and context-free (V4)", "verify",
     'run(f"verify --model {SLM} --backend {BACKEND} --dtype {DTYPE}")'),
    ("LLM answers", "llm", 'run(f"llm --model {LLM} --backend {BACKEND} --dtype {DTYPE}")'),
    ("Query embeddings for the query-only prior (V0)", "embed", 'run("embed")'),
    ("Grade and merge into cache.jsonl", "merge", 'run("merge")'),
    ("SLM hidden states for the probe signal", "hidden", 'run(f"hidden --model {SLM} --dtype {DTYPE}")'),
]

ZIP = """
os.makedirs(f"{OUT}_zip", exist_ok=True)
for f in STAGE_FILES:
    if os.path.exists(f"{OUT}/{f}"):
        shutil.copy(f"{OUT}/{f}", f"{OUT}_zip/{f}")
shutil.make_archive(f"{WORK}/{os.path.basename(OUT)}", "zip", f"{OUT}_zip")
print("wrote", f"{WORK}/{os.path.basename(OUT)}.zip")
"""


def build(pilot: bool = True, stages: list[str] | None = None, slm: str | None = None,
          llm: str | None = None) -> dict:
    config = CONFIG.replace("PILOT = True ", f"PILOT = {pilot} ", 1)
    if slm:
        config = re.sub(r'SLM = ".*?"', f'SLM = "{slm}"', config, count=1)
    if llm:
        config = re.sub(r'LLM = ".*?"', f'LLM = "{llm}"', config, count=1)
    if stages:
        config = re.sub(r"STAGES = \[.*?\]", f"STAGES = {stages!r}", config, count=1)
    cells = [md(INTRO), code(config), md("## Install"), code(INSTALL), code(CHECK),
             md("## Write the `vcascade` package\nGenerated from `src/vcascade/`; do not edit here.")]
    cells.append(code('os.makedirs("src/vcascade", exist_ok=True)  # relative to WORK, set above'))
    for name in MODULES:
        src = (PKG / name).read_text(encoding="utf-8")
        cells.append(code(f"%%writefile src/vcascade/{name}\n{src}"))
    cells.append(md("## Run the pipeline"))
    for title, stage, cmd in RUN:
        cells += [md(f"### {title}"), code(f'if "{stage}" in STAGES:\n    {cmd}')]
    cells += [md("## Package the outputs"), code(ZIP)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="PILOT = False (all 2,819 queries)")
    p.add_argument("--out", default=str(ROOT / "notebooks" / "generate.ipynb"))
    p.add_argument("--stages", default=None, help="comma-separated subset of stages to run")
    p.add_argument("--slm", default=None)
    p.add_argument("--llm", default=None)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(pilot=not args.full, stages=args.stages.split(",") if args.stages else None,
                                     slm=args.slm, llm=args.llm), indent=1), encoding="utf-8")
    print("wrote", out, "(full run)" if args.full else "(pilot)")
