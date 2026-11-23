"""Scoring helpers in evaluate.py. Offline."""

import evaluate


def question(**kw) -> dict:
    base = {"id": "q", "question": "?", "kind": "single", "expected_source": "",
            "expected_location": "", "reference_answer": ""}
    return {**base, **kw}


def test_expected_pairs_each_source_with_its_location():
    q = question(expected_source="Reg 2019/947;Reg 2019/945", expected_location="UAS.OPEN.030;PART 3")
    assert evaluate.expected(q) == [("Reg 2019/947", "UAS.OPEN.030"), ("Reg 2019/945", "PART 3")]


def test_expected_repeats_a_single_source_for_several_locations():
    q = question(expected_source="Reg 2019/947", expected_location="UAS.OPEN.020;UAS.OPEN.070")
    assert evaluate.expected(q) == [("Reg 2019/947", "UAS.OPEN.020"), ("Reg 2019/947", "UAS.OPEN.070")]


def test_is_hit_needs_the_same_source_and_a_location_prefix():
    exp = [("Reg 2019/945", "PART 3")]
    assert evaluate.is_hit({"source": "Reg 2019/945", "location": "PART 3 Requirements for a class C2"}, exp)
    assert not evaluate.is_hit({"source": "Reg 2019/945", "location": "PART 13 Requirements"}, exp)
    assert not evaluate.is_hit({"source": "Reg 2019/947", "location": "PART 3"}, exp)


def test_outcome_labels():
    assert evaluate.outcome({"kind": "unanswerable", "refused": True}) == "correct refusal"
    assert evaluate.outcome({"kind": "unanswerable", "refused": False}) == "answered unanswerable"
    assert evaluate.outcome({"kind": "single", "refused": True}) == "wrong refusal"
    assert evaluate.outcome({"kind": "single", "refused": False, "supported": 0, "correct": 1}) == "unsupported"
    assert evaluate.outcome({"kind": "multi", "refused": False, "supported": 1, "correct": 0}) == "wrong"
    assert evaluate.outcome({"kind": "single", "refused": False, "supported": 1, "correct": 1}) == "correct"


def test_kappa_is_undefined_without_variance_and_zero_at_chance():
    assert evaluate.kappa([(1, 1)] * 20) is None
    assert evaluate.kappa([(1, 1)] * 17 + [(1, 0)] * 3) == 0.0
    assert evaluate.kappa([(1, 1), (0, 0), (1, 1), (0, 0)]) == 1.0


def test_agreement_accepts_any_number_of_rows_and_skips_incomplete_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "EVAL", tmp_path)
    (tmp_path / "human_labels.csv").write_text(
        "id,correct,supported,notes\nq1,1,1,\nq2,0,1,why\nq3,,,\n", encoding="utf-8")
    results = [{"id": i, "correct": 1, "supported": 1} for i in ("q1", "q2", "q3")]
    a = evaluate.agreement(results)
    assert a["graded_by_me"] == 2
    assert a["agree_correct"] == 0.5 and a["agree_supported"] == 1.0
    assert a["disagreements"] == ["q2"]


def test_agreement_without_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "EVAL", tmp_path)
    assert evaluate.agreement([{"id": "q1", "correct": 1, "supported": 1}]) == {"graded_by_me": 0}
