from vcascade import grading


def test_extract_answer_prefers_last_final_answer():
    text = "Step 1: 3 + 4 = 7.\nFinal answer: 7\nWait, recheck.\nFinal answer: 8"
    assert grading.extract_answer(text) == "8"


def test_extract_answer_falls_back_to_last_line():
    assert grading.extract_answer("Paris is the capital.\n\nParis") == "Paris"


def test_canonical_numbers_and_text():
    assert grading.canonical("$1,200.00") == "1200"
    assert grading.canonical("18 eggs") == "18"
    assert grading.canonical("The Beatles") == "beatles"
    assert grading.canonical(None) is None


def test_math_correctness():
    q = {"task": "math", "gold": {"answer": "1200"}}
    assert grading.is_correct(q, "... so the total is 1,200 dollars.\nFinal answer: $1,200")
    assert not grading.is_correct(q, "Final answer: 1100")


def test_factual_correctness_uses_normalised_aliases():
    q = {"task": "factual", "gold": {"answer": "The Beatles", "aliases": ["beatles", "fab four"]}}
    assert grading.is_correct(q, "Final answer: The Beatles.")
    assert not grading.is_correct(q, "Final answer: The Rolling Stones")


def test_parse_verdict_last_mention_wins():
    assert grading.parse_verdict("The answer seems correct... Verdict: Incorrect") == 0
    assert grading.parse_verdict("Verdict: Correct") == 1
    assert grading.parse_verdict("no idea") is None


def test_factual_containment_in_short_sentence():
    q = {"task": "factual", "gold": {"answer": "Benedict Cumberbatch", "aliases": ["benedict cumberbatch"]}}
    assert grading.is_correct(q, "Final answer: The role is played by Benedict Cumberbatch.")
    long_hedge = "Final answer: " + "maybe " * 20 + "Benedict Cumberbatch"
    assert not grading.is_correct(q, long_hedge)


def test_answers_match():
    assert grading.answers_match("Benedict Cumberbatch", "played by Benedict Cumberbatch")
    assert grading.answers_match("$18", "18 eggs")
    assert not grading.answers_match("18", "180")
    assert not grading.answers_match("Paris", "London")
    assert not grading.answers_match(None, "x")
