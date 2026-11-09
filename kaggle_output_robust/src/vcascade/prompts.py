"""Prompts for answering and for the two LLM-based verifiers (V3, V4).

One generic answer prompt serves every query: a deployed assistant does not
know the task type in advance, so nothing here depends on it.

The verification prompts follow AutoMix's entailment-style prompt
(Aggarwal et al., 2024, Listing 1). V3 adds retrieved Wikipedia passages as
the context; V4 is the same prompt with no context at all.
"""

from __future__ import annotations

ANSWER_PROMPT = (
    "Answer the following question. If it needs calculation or reasoning, think step by step first; "
    "if it is a factual question, answer directly in a few words. "
    "End your response with a final line of the form 'Final answer: <answer>', where <answer> is "
    "only the answer itself (a number, a name or a short phrase), not a sentence.\n\n"
    "Question: {question}"
)
MAX_ANSWER_TOKENS = 512

_VERIFY_INSTRUCTION = (
    "Instruction: Your task is to evaluate if the AI Generated Answer is correct{basis}. "
    "Think briefly, then finish with exactly one line: 'Verdict: Correct' or 'Verdict: Incorrect'."
)

VERIFY_WITH_CONTEXT = (
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "AI Generated Answer: {answer}\n\n"
    + _VERIFY_INSTRUCTION.format(basis=", based on the provided context and question")
    + "\n\nEvaluation:"
)

VERIFY_NO_CONTEXT = (
    "Question: {question}\n\n"
    "AI Generated Answer: {answer}\n\n"
    + _VERIFY_INSTRUCTION.format(basis="")
    + "\n\nEvaluation:"
)
MAX_VERIFY_TOKENS = 256


def answer_messages(query: dict) -> list[dict]:
    return [{"role": "user", "content": ANSWER_PROMPT.format(question=query["question"])}]


def verify_messages(query: dict, answer_text: str, context: str | None) -> list[dict]:
    answer_text = answer_text.strip()[-1500:]
    if context is None:
        content = VERIFY_NO_CONTEXT.format(question=query["question"], answer=answer_text)
    else:
        content = VERIFY_WITH_CONTEXT.format(context=context, question=query["question"], answer=answer_text)
    return [{"role": "user", "content": content}]
