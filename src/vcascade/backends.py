"""Model backends with one interface.

    backend.generate(conversations, n=..., temperature=..., max_tokens=..., logprobs=...)
        -> one list per conversation, holding n completions of the form
           {"text": str, "token_logprobs": list[float] | None,
            "prompt_tokens": int, "completion_tokens": int}

VLLMBackend runs a local model on the GPU (Kaggle / Colab).
OpenAICompatBackend calls any OpenAI-compatible endpoint (Together, Groq,
OpenRouter, or a vLLM server).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor


class VLLMBackend:
    def __init__(self, model: str, dtype: str = "auto", tensor_parallel_size: int = 1,
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.90, seed: int = 0):
        from vllm import LLM

        self.model = model
        self.llm = LLM(model=model, dtype=dtype, tensor_parallel_size=tensor_parallel_size,
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       seed=seed, enforce_eager=False)

    def generate(self, conversations, n=1, temperature=0.0, max_tokens=512, logprobs=False, seed=0):
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0 if temperature == 0 else 0.95,
                            max_tokens=max_tokens, logprobs=1 if logprobs else None, seed=seed)
        outs = self.llm.chat(conversations, sp, use_tqdm=True)
        results = []
        for req in outs:
            comps = []
            for c in req.outputs:
                lps = None
                if logprobs and c.logprobs is not None:
                    lps = [step[tok].logprob for tok, step in zip(c.token_ids, c.logprobs)]
                comps.append({"text": c.text, "token_logprobs": lps,
                              "prompt_tokens": len(req.prompt_token_ids), "completion_tokens": len(c.token_ids)})
            results.append(comps)
        return results


class OpenAICompatBackend:
    def __init__(self, model: str, base_url: str, api_key_env: str = "LLM_API_KEY",
                 max_workers: int = 8, max_retries: int = 6):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=os.environ[api_key_env])
        self.max_workers = max_workers
        self.max_retries = max_retries

    def _one(self, messages, temperature, max_tokens, logprobs, seed):
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, logprobs=logprobs or None, seed=seed)
                ch = r.choices[0]
                lps = [t.logprob for t in ch.logprobs.content] if logprobs and ch.logprobs else None
                return {"text": ch.message.content or "", "token_logprobs": lps,
                        "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}
            except Exception:  # rate limits and transient errors: back off and retry
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def generate(self, conversations, n=1, temperature=0.0, max_tokens=512, logprobs=False, seed=0):
        # Many providers reject n > 1, so samples are separate requests.
        jobs = [(m, s) for m in conversations for s in range(n)]
        with ThreadPoolExecutor(self.max_workers) as pool:
            flat = list(pool.map(lambda j: self._one(j[0], temperature, max_tokens, logprobs, seed + j[1]), jobs))
        return [flat[i * n:(i + 1) * n] for i in range(len(conversations))]


class HFBackend:
    """Plain transformers fallback, for when vLLM will not install. Slower, works on any GPU."""

    def __init__(self, model: str, dtype: str = "auto", batch_size: int = 16, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        torch_dtype = {"auto": "auto", "half": torch.float16, "float16": torch.float16,
                       "bfloat16": torch.bfloat16}[dtype]
        self.lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype, device_map="auto").eval()
        self.batch_size = batch_size

    def generate(self, conversations, n=1, temperature=0.0, max_tokens=512, logprobs=False, seed=0):
        import torch
        from tqdm.auto import tqdm

        torch.manual_seed(seed)
        eos = {self.tok.eos_token_id, self.tok.pad_token_id}
        results = []
        bs = max(1, self.batch_size // n)
        for start in tqdm(range(0, len(conversations), bs)):
            texts = [self.tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                     for c in conversations[start:start + bs]]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.lm.device)
            sample = temperature > 0
            with torch.no_grad():
                out = self.lm.generate(**enc, max_new_tokens=max_tokens, do_sample=sample,
                                       temperature=temperature if sample else None, top_p=0.95 if sample else None,
                                       num_return_sequences=n, return_dict_in_generate=True,
                                       output_scores=logprobs, pad_token_id=self.tok.pad_token_id)
            gen = out.sequences[:, enc["input_ids"].shape[1]:]
            lps = self.lm.compute_transition_scores(out.sequences, out.scores, normalize_logits=True) \
                if logprobs else None
            for j in range(len(texts)):
                comps = []
                for r in range(n):
                    row = j * n + r
                    ids = gen[row].tolist()
                    length = next((t for t, x in enumerate(ids) if x in eos), len(ids))
                    comps.append({
                        "text": self.tok.decode(ids[:length], skip_special_tokens=True),
                        "token_logprobs": lps[row, :length].float().tolist() if logprobs else None,
                        "prompt_tokens": int(enc["attention_mask"][j].sum()), "completion_tokens": length})
                results.append(comps)
        return results


def make_backend(spec: dict):
    kind = spec.pop("kind")
    return {"vllm": VLLMBackend, "openai": OpenAICompatBackend, "hf": HFBackend}[kind](**spec)
