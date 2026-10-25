"""Answer extraction, answer comparison and correctness checks.

Extraction and comparison are task-agnostic (they feed router signals such as
self-consistency). Only `is_correct` reads the gold answer and the task.
"""

from __future__ import annotations

import re
import string

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_FINAL = re.compile(r"final answer\s*[:：]\s*(.+)", flags=re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    """Text after the last 'Final answer:'; falls back to the last non-empty line."""
    hits = _FINAL.findall(text)
    if hits:
        ans = hits[-1]
    else:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            return None
        ans = lines[-1]
    ans = ans.strip().strip("*").strip()
    return ans or None


def canon_number(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    if v != v or v in (float("inf"), float("-inf")):  # absurdly long digit strings overflow
        return None
    return str(int(v)) if v == int(v) else f"{v:g}"


def normalize_text(s: str) -> str:
    """TriviaQA / SQuAD normalisation: lowercase, drop punctuation and articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def canonical(answer: str | None) -> str | None:
    """Comparison key for two answers: a number if the answer is numeric, else normalised text."""
    if answer is None:
        return None
    nums = _NUM.findall(answer)
    # "18", "$18.00" and "18 eggs" all compare as 18; longer text compares as text.
    if len(nums) == 1 and len(answer.split()) <= 4:
        num = canon_number(nums[0])
        if num is not None:
            return num
    return normalize_text(answer)


MAX_ANSWER_WORDS = 15  # longer predictions are not credited by containment (they may hedge)


def _contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word containment of one normalised phrase in another."""
    return bool(phrase) and f" {phrase} " in f" {text} "


def answers_match(a: str | None, b: str | None) -> bool:
    """Do two extracted answers say the same thing? Used for self-consistency (V2).

    Numbers compare numerically; text compares after normalisation, also accepting
    one answer contained in the other ("Benedict Cumberbatch" vs "played by Benedict Cumberbatch").
    """
    ka, kb = canonical(a), canonical(b)
    if ka is None or kb is None:
        return False
    if ka == kb:
        return True
    if canon_number(ka) is not None or canon_number(kb) is not None:
        return False
    short, long_ = sorted((ka, kb), key=len)
    return len(long_.split()) <= MAX_ANSWER_WORDS and _contains_phrase(long_, short)


def is_correct(query: dict, text: str) -> bool:
    pred = extract_answer(text)
    if pred is None:
        return False
    gold = query["gold"]
    if query["task"] == "math":
        nums = _NUM.findall(pred)
        return bool(nums) and canon_number(nums[-1]) == canon_number(gold["answer"])
    # TriviaQA: exact match on normalised aliases, or an alias as whole words inside a short answer
    p = normalize_text(pred)
    if p in set(gold["aliases"]):
        return True
    return len(p.split()) <= MAX_ANSWER_WORDS and any(_contains_phrase(p, a) for a in gold["aliases"])


def parse_verdict(text: str) -> int | None:
    """1 = Correct, 0 = Incorrect, None = no verdict found. The last mention wins."""
    hits = re.findall(r"\b(incorrect|correct)\b", text, flags=re.IGNORECASE)
    if not hits:
        return None
    return 0 if hits[-1].lower() == "incorrect" else 1
