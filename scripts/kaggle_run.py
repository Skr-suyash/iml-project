"""Push the generation notebook to Kaggle, wait for it, and download the cache.

Needs a Kaggle API token in ~/.kaggle/access_token (or KAGGLE_API_TOKEN).

    python scripts/kaggle_run.py push --pilot     # 200-query pilot on a T4
    python scripts/kaggle_run.py push --full      # full 2,819-query run
    python scripts/kaggle_run.py status
    python scripts/kaggle_run.py fetch            # download outputs into kaggle_output/

Two-phase run (retrieval done locally, where Wikipedia does not throttle):
    python scripts/kaggle_run.py push --full --stages prepare,slm,llm,embed
    python scripts/kaggle_run.py fetch
    WIKI_CONTACT=... python -m vcascade.generate --out kaggle_output/cache retrieve
    python scripts/kaggle_run.py upload --src kaggle_output/cache     # private dataset vcascade-inputs
    python scripts/kaggle_run.py push --full --stages verify,merge --dataset

SLM hidden states for the probe signal (about 5 minutes; no vLLM install):
    python scripts/kaggle_run.py push --full --stages hidden --dataset --slug vcascade-hidden
    python scripts/kaggle_run.py fetch --slug vcascade-hidden --out kaggle_output_hidden
    cp kaggle_output_hidden/cache/hidden.npz cache/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "kaggle_kernel"
SLUG = "vcascade-generate"
DATASET = "vcascade-inputs"
INPUT_FILES = ["queries.jsonl", "slm.jsonl", "retrieval.jsonl", "llm.jsonl", "embeddings.npy", "embeddings_qids.json"]
KAGGLE = str(Path(sys.executable).with_name("kaggle"))


def kaggle(*args, check=True) -> str:
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def username() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    name = api.get_config_value("username") or getattr(api, "_username", None)
    if not name:
        sys.exit("Could not determine your Kaggle username; pass --user.")
    return name


def upload(src: Path, user: str, dataset: str = DATASET, files: list[str] | None = None):
    """Create or update a private dataset holding earlier stages' outputs."""
    ddir = ROOT / "kaggle_dataset"
    if ddir.exists():
        shutil.rmtree(ddir)
    ddir.mkdir()
    for f in files or INPUT_FILES:
        if (src / f).exists():
            shutil.copy(src / f, ddir / f)
    (ddir / "dataset-metadata.json").write_text(json.dumps(
        {"title": dataset, "id": f"{user}/{dataset}", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    exists = f"{user}/{dataset}" in kaggle("datasets", "list", "--mine", "-s", dataset, check=False)
    if exists:
        print(kaggle("datasets", "version", "-p", str(ddir), "-m", "update inputs", "-r", "skip"))
    else:
        print(kaggle("datasets", "create", "-p", str(ddir)))  # private by default


def push(full: bool, user: str | None, stages: str | None = None, dataset: str | None = None,
         slug: str = SLUG, slm: str | None = None):
    user = user or username()
    if KDIR.exists():
        shutil.rmtree(KDIR)
    KDIR.mkdir()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_notebook.py"),
                    "--out", str(KDIR / "generate.ipynb"), *(["--full"] if full else []),
                    *(["--stages", stages] if stages else []), *(["--slm", slm] if slm else [])], check=True)
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "generate.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{dataset}"] if dataset else [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (KDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(kaggle("kernels", "push", "-p", str(KDIR), "--accelerator", "NvidiaTeslaT4"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["push", "status", "fetch", "upload"])
    p.add_argument("--full", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--stages", default=None, help="comma-separated stages to run")
    p.add_argument("--dataset", nargs="?", const=DATASET, default=None,
                   help="attach a private inputs dataset (default name vcascade-inputs)")
    p.add_argument("--src", default="kaggle_output/cache", help="folder to upload as the inputs dataset")
    p.add_argument("--files", default=None, help="comma-separated files to upload (default: all inputs)")
    p.add_argument("--slug", default=SLUG, help="kernel name")
    p.add_argument("--slm", default=None, help="override the small model")
    p.add_argument("--out", default="kaggle_output", help="where fetch downloads to")
    p.add_argument("--user", default=None)
    args = p.parse_args()
    user = args.user or username()
    ref = f"{user}/{args.slug}"
    if args.cmd == "push":
        push(full=args.full, user=user, stages=args.stages, dataset=args.dataset, slug=args.slug, slm=args.slm)
    elif args.cmd == "upload":
        upload(ROOT / args.src, user, dataset=args.dataset or DATASET,
               files=args.files.split(",") if args.files else None)
    elif args.cmd == "status":
        print(kaggle("kernels", "status", ref))
    else:
        out = ROOT / args.out
        out.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", ref, "-p", str(out)))


if __name__ == "__main__":
    main()
