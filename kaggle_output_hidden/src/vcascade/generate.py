"""Generation pipeline: produce every model output and verifier signal once.

Each stage writes one file in --out and is skipped if that file exists, so a
dropped Kaggle / Colab session resumes where it stopped. Run the GPU stages as
separate processes so each model's GPU memory is released on exit.

    python -m vcascade.generate --out cache prepare --limit-math 1319 --limit-factual 1500
    python -m vcascade.generate --out cache slm      --model Qwen/Qwen2.5-1.5B-Instruct
    python -m vcascade.generate --out cache retrieve
    python -m vcascade.generate --out cache verify   --model Qwen/Qwen2.5-1.5B-Instruct
    python -m vcascade.generate --out cache llm      --model Qwen/Qwen2.5-7B-Instruct-AWQ
    python -m vcascade.generate --out cache embed
    python -m vcascade.generate --out cache merge
    python -m vcascade.generate --out cache hidden   --model Qwen/Qwen2.5-1.5B-Instruct   # probe signal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import grading, prompts


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomic: a killed session never leaves a half-written stage file


def _backend(args):
    from .backends import make_backend

    if args.backend == "vllm":
        return make_backend({"kind": "vllm", "model": args.model, "dtype": args.dtype,
                             "tensor_parallel_size": args.tp, "max_model_len": args.max_model_len})
    if args.backend == "hf":
        return make_backend({"kind": "hf", "model": args.model, "dtype": args.dtype})
    return make_backend({"kind": "openai", "model": args.model, "base_url": args.base_url})


def stage_prepare(args, out: Path):
    from .data import build_stream

    queries = build_stream(limits={"math": args.limit_math, "factual": args.limit_factual}, seed=args.seed)
    write_jsonl(out / "queries.jsonl", queries)
    print(f"{len(queries)} queries written")


def stage_slm(args, out: Path):
    queries = read_jsonl(out / "queries.jsonl")
    be = _backend(args)
    convs = [prompts.answer_messages(q) for q in queries]
    greedy = be.generate(convs, n=1, temperature=0.0, max_tokens=prompts.MAX_ANSWER_TOKENS, logprobs=True)
    samples = be.generate(convs, n=args.n_samples, temperature=0.7,
                          max_tokens=prompts.MAX_ANSWER_TOKENS, seed=args.seed)
    write_jsonl(out / "slm.jsonl", [
        {"qid": q["qid"], "model": args.model, "greedy": g[0], "samples": s}
        for q, g, s in zip(queries, greedy, samples)])


def stage_retrieve(args, out: Path):
    from .retrieval import retrieve_all

    queries = read_jsonl(out / "queries.jsonl")
    slm = {r["qid"]: r for r in read_jsonl(out / "slm.jsonl")}
    search = []
    for q in queries:
        ans = grading.extract_answer(slm[q["qid"]]["greedy"]["text"]) or ""
        search.append(f"{q['question']} {ans}")
    passages = retrieve_all(search, k=args.k_passages, resume_file=out / "retrieval.partial.jsonl",
                            fallbacks=[q["question"] for q in queries])
    write_jsonl(out / "retrieval.jsonl", [
        {"qid": q["qid"], "search": s, "passages": p} for q, s, p in zip(queries, search, passages)])


def stage_verify(args, out: Path):
    from .retrieval import format_context

    queries = read_jsonl(out / "queries.jsonl")
    slm = {r["qid"]: r for r in read_jsonl(out / "slm.jsonl")}
    ret = {r["qid"]: r for r in read_jsonl(out / "retrieval.jsonl")}
    be = _backend(args)
    kw = dict(n=args.n_verify, temperature=0.7, max_tokens=prompts.MAX_VERIFY_TOKENS, seed=args.seed)
    v3 = be.generate([prompts.verify_messages(q, slm[q["qid"]]["greedy"]["text"],
                                              format_context(ret[q["qid"]]["passages"])) for q in queries], **kw)
    v4 = be.generate([prompts.verify_messages(q, slm[q["qid"]]["greedy"]["text"], None) for q in queries], **kw)
    write_jsonl(out / "verify.jsonl", [
        {"qid": q["qid"], "v3": a, "v4": b} for q, a, b in zip(queries, v3, v4)])


def stage_llm(args, out: Path):
    queries = read_jsonl(out / "queries.jsonl")
    be = _backend(args)
    greedy = be.generate([prompts.answer_messages(q) for q in queries], n=1, temperature=0.0,
                         max_tokens=prompts.MAX_ANSWER_TOKENS)
    write_jsonl(out / "llm.jsonl", [
        {"qid": q["qid"], "model": args.model, "greedy": g[0]} for q, g in zip(queries, greedy)])


def stage_embed(args, out: Path):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    queries = read_jsonl(out / "queries.jsonl")
    model = SentenceTransformer(args.embed_model)
    emb = model.encode([q["question"] for q in queries], batch_size=128, normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(out / "embeddings.npy", emb.astype("float32"))
    (out / "embeddings_qids.json").write_text(json.dumps([q["qid"] for q in queries]))


def stage_hidden(args, out: Path):
    """SLM hidden states for the probe signal: one forward pass over prompt + greedy answer.

    Saves, at four depths (1/4, 1/2, 3/4, last layer), float16 arrays of shape (n, 4, d):
        prompt  state at the last prompt token, before any answer exists (query-only)
        last    state at the last answer token
        mean    mean state over the answer tokens
    """
    import numpy as np
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    queries, seen = [], set()
    for q in read_jsonl(out / "queries.jsonl"):
        if q["qid"] not in seen:
            seen.add(q["qid"])
            queries.append(q)
    slm = {r["qid"]: r for r in read_jsonl(out / "slm.jsonl")}
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.dtype in ("half", "float16", "auto") else torch.bfloat16
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        dtype = torch.float32
    lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    n_layers = lm.config.num_hidden_layers
    layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]  # hidden_states[0] = embeddings

    seqs = []  # (prompt length, token ids), truncated to max_model_len
    for q in queries:
        prompt = tok.apply_chat_template(prompts.answer_messages(q), add_generation_prompt=True, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(slm[q["qid"]]["greedy"]["text"], add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
        seqs.append((len(p_ids), (p_ids + a_ids)[:args.max_model_len]))

    d = lm.config.hidden_size
    feats = {k: np.zeros((len(seqs), len(layers), d), dtype=np.float16) for k in ("prompt", "last", "mean")}
    order = np.argsort([len(s) for _, s in seqs])  # similar lengths per batch: less padding
    bs = args.batch_size
    for start in tqdm(range(0, len(order), bs)):
        idx = order[start:start + bs]
        width = max(len(seqs[i][1]) for i in idx)
        ids = torch.full((len(idx), width), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for j, i in enumerate(idx):  # right padding keeps positions simple
            s = seqs[i][1]
            ids[j, :len(s)], mask[j, :len(s)] = torch.tensor(s), 1
        with torch.no_grad():
            hs = lm(input_ids=ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True).hidden_states
        for li, layer in enumerate(layers):
            h = hs[layer].float()
            for j, i in enumerate(idx):
                p, length = seqs[i][0], len(seqs[i][1])
                p = min(p, length - 1)  # answer truncated away entirely: fall back to the prompt
                feats["prompt"][i, li] = h[j, p - 1].cpu().numpy()
                feats["last"][i, li] = h[j, length - 1].cpu().numpy()
                feats["mean"][i, li] = h[j, p:length].mean(0).cpu().numpy()
    np.savez_compressed(out / "hidden.npz", qids=np.array([q["qid"] for q in queries]), layers=np.array(layers),
                        **feats)
    print(f"hidden states for {len(queries)} queries, layers {layers}, dim {d}")


def stage_merge(args, out: Path):
    """Grade everything and write cache.jsonl: the only file the router side needs."""
    import math

    queries = read_jsonl(out / "queries.jsonl")
    slm = {r["qid"]: r for r in read_jsonl(out / "slm.jsonl")}
    llm = {r["qid"]: r for r in read_jsonl(out / "llm.jsonl")}
    ver = {r["qid"]: r for r in read_jsonl(out / "verify.jsonl")}
    rows, seen = [], set()
    for q in queries:
        if q["qid"] in seen:  # TriviaQA repeats some question ids; keep one copy
            continue
        seen.add(q["qid"])
        s, l, v = slm[q["qid"]], llm[q["qid"]], ver[q["qid"]]
        g = s["greedy"]
        lps = g["token_logprobs"] or [0.0]
        greedy_ans = grading.extract_answer(g["text"])
        rows.append({
            "qid": q["qid"],
            "task": q["task"],
            "question": q["question"],
            "slm_answer": grading.extract_answer(g["text"]),
            "llm_answer": grading.extract_answer(l["greedy"]["text"]),
            "slm_correct": int(grading.is_correct(q, g["text"])),
            "llm_correct": int(grading.is_correct(q, l["greedy"]["text"])),
            # V1: confidence of the greedy answer, free with generation
            "v1": {"mean_lp": sum(lps) / len(lps), "min_lp": min(lps),
                   "perplexity": math.exp(-sum(lps) / len(lps))},
            # V2: does each temperature sample agree with the greedy answer?
            "v2": [int(grading.answers_match(grading.extract_answer(c["text"]), greedy_ans))
                   for c in s["samples"]],
            # V3 / V4: verdict votes, 1 = Correct, 0 = Incorrect, None = unparseable
            "v3": [grading.parse_verdict(c["text"]) for c in v["v3"]],
            "v4": [grading.parse_verdict(c["text"]) for c in v["v4"]],
            "tokens": {
                "slm_in": g["prompt_tokens"], "slm_out": g["completion_tokens"],
                "sample_out": sum(c["completion_tokens"] for c in s["samples"]) / max(len(s["samples"]), 1),
                "v3_in": v["v3"][0]["prompt_tokens"], "v3_out": sum(c["completion_tokens"] for c in v["v3"]),
                "v4_in": v["v4"][0]["prompt_tokens"], "v4_out": sum(c["completion_tokens"] for c in v["v4"]),
                "llm_in": l["greedy"]["prompt_tokens"], "llm_out": l["greedy"]["completion_tokens"],
            },
        })
    write_jsonl(out / "cache.jsonl", rows)
    n = len(rows)
    for task in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == task]
        print(f"{task:8s} n={len(sub):5d}  SLM acc={sum(r['slm_correct'] for r in sub) / len(sub):.3f}"
              f"  LLM acc={sum(r['llm_correct'] for r in sub) / len(sub):.3f}")
    print(f"all      n={n:5d}  SLM acc={sum(r['slm_correct'] for r in rows) / n:.3f}"
          f"  LLM acc={sum(r['llm_correct'] for r in rows) / n:.3f}")


STAGES = {"prepare": (stage_prepare, "queries.jsonl"), "slm": (stage_slm, "slm.jsonl"),
          "retrieve": (stage_retrieve, "retrieval.jsonl"), "verify": (stage_verify, "verify.jsonl"),
          "llm": (stage_llm, "llm.jsonl"), "embed": (stage_embed, "embeddings.npy"),
          "merge": (stage_merge, "cache.jsonl"), "hidden": (stage_hidden, "hidden.npz")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--out", default="cache")
    p.add_argument("--force", action="store_true", help="rerun even if the stage's output exists")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--backend", choices=["vllm", "hf", "openai"], default="vllm")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, for --backend openai")
    p.add_argument("--dtype", default="auto", help="'half' on T4 / P100 GPUs, which lack bfloat16")
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size (number of GPUs)")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--limit-math", type=int, default=None)
    p.add_argument("--limit-factual", type=int, default=1500)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--n-verify", type=int, default=4)
    p.add_argument("--k-passages", type=int, default=3)
    p.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8, help="forward-pass batch for the hidden stage")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fn, product = STAGES[args.stage]
    if (out / product).exists() and not args.force:
        print(f"{args.stage}: {out / product} exists, skipping (use --force to rerun)")
        return
    fn(args, out)


if __name__ == "__main__":
    main()
