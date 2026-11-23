"""
Evaluation (`main.py evaluate`). Writes everything to eval/.

1. Retrieval: hit rate at k = 1/3/5 for each chunk design   -> eval/retrieval.csv
   (hit@k: any expected passage; multi_all@k: multi questions with every expected passage)
2. Agent on every question (cached, so reruns are cheap)    -> eval/answers.jsonl
3. Code checks: citations valid, refusals right or wrong
4. LLM judge with a rubric, retrieved text as evidence      -> eval/judge.jsonl
5. Agreement with my grades in eval/human_labels.csv
6. Optional: add a document and re-ask one unanswerable question -> eval/add_document_demo.md
Totals go to eval/summary.json; failed questions to eval/failures.csv.
"""

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

from google.genai import errors, types

import agent
import index

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"
QUESTIONS = EVAL / "questions.csv"
QUERY_VECS = EVAL / "query_vectors.json"
ANSWERS = EVAL / "answers.jsonl"
JUDGE = EVAL / "judge.jsonl"
KS = (1, 3, 5)

# Before/after demo: an unanswerable question and a file that answers it.
# Leave as None until both are chosen.
DEMO_ID = "q31"  # U-space services: not in the corpus
DEMO_FILE = "eval/extra/reg_2021_664_u-space.html"  # Regulation (EU) 2021/664

RUBRIC = """You grade answers from an EU drone-rules assistant.

Question: {question}
Reference answer: {reference}
Retrieved passages (the only evidence the assistant had):
<<<
{evidence}
>>>
Assistant's answer:
<<<
{answer}
>>>

Grade two things, each 1 or 0:
- correct: 1 if the answer states the same rule as the reference (wording may differ;
  extra correct detail is fine). 0 if it contradicts the reference, misses its key point,
  or refuses although the reference gives an answer.
- supported: 1 if every factual claim in the answer appears in the retrieved passages.
  0 if any claim (a number, condition, exception) is not in them, even if it is true.

Reply with JSON only: {{"correct": 0 or 1, "supported": 0 or 1, "reason": "one sentence"}}"""


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if (r.get("id") or "").strip()]


def read_jsonl(path: Path) -> dict:
    if not path.is_file():
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r for r in rows}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def expected(q: dict) -> list[tuple[str, str]]:
    """(source, location) pairs; ';' separates several in both columns."""
    srcs = [s.strip() for s in q["expected_source"].split(";")]
    locs = [l.strip() for l in q["expected_location"].split(";")]
    if len(srcs) == 1 and len(locs) > 1:
        srcs *= len(locs)
    return list(zip(srcs, locs + [""] * (len(srcs) - len(locs))))


def is_hit(hit: dict, exp: list[tuple[str, str]]) -> bool:
    return any(hit["source"] == s and index.loc_match(l, hit["location"]) for s, l in exp)


# ---------- 1. retrieval ----------

def query_vectors(questions: list[dict]) -> dict:
    """One embedding per question text, saved to disk and shared by all designs."""
    vecs = json.loads(QUERY_VECS.read_text()) if QUERY_VECS.is_file() else {}
    todo = [q for q in questions if q["question"] not in vecs]
    for i in range(0, len(todo), index.BATCH):
        batch = todo[i:i + index.BATCH]
        for q, v in zip(batch, index.embed([q["question"] for q in batch], task="RETRIEVAL_QUERY")):
            vecs[q["question"]] = v
        QUERY_VECS.write_text(json.dumps(vecs))
    return vecs


def retrieval(questions: list[dict]) -> list[dict]:
    answerable = [q for q in questions if q["kind"] != "unanswerable"]
    multi = [q for q in answerable if q["kind"] == "multi"]
    vecs = query_vectors(answerable)
    rows = []
    for design in index.DESIGNS:
        hits_at = {k: 0 for k in KS}  # any expected passage in the top k
        all_at = {k: 0 for k in KS}   # multi questions: every expected passage in the top k
        words = 0
        for q in answerable:
            hits = index.retrieve(q["question"], design, k=max(KS), query_vec=vecs[q["question"]])
            words += sum(len(h["text"].split()) for h in hits)
            for k in KS:
                hits_at[k] += any(is_hit(h, expected(q)) for h in hits[:k])
                if q["kind"] == "multi":
                    all_at[k] += all(any(is_hit(h, [e]) for h in hits[:k]) for e in expected(q))
        n = len(answerable)
        rows.append({"design": design, **{f"hit@{k}": round(hits_at[k] / n, 3) for k in KS},
                     **{f"multi_all@{k}": round(all_at[k] / max(len(multi), 1), 3) for k in KS},
                     "avg_words_top5": round(words / n)})
    with (EVAL / "retrieval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------- 2-4. agent, code checks, judge ----------

def answer_all(questions: list[dict]) -> dict:
    done = read_jsonl(ANSWERS)
    for q in questions:
        if q["id"] in done:
            continue
        print(f"  asking {q['id']}", file=sys.stderr)
        try:
            r = asyncio.run(agent.run(q["question"], approve=lambda *_: False))
        except Exception as e:
            # MCP wraps errors in exception groups; dig out the real one
            while getattr(e, "exceptions", None):
                e = e.exceptions[0]
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                raise SystemExit(f"Gemini quota exhausted at {q['id']}. Answers so far are saved; rerun later to continue.")
            raise
        done[q["id"]] = {"id": q["id"], **r}
        append_jsonl(ANSWERS, done[q["id"]])  # saved one by one so a rerun resumes here
    return done


def code_checks(q: dict, a: dict) -> dict:
    refused = a["answer"].strip().startswith(agent.REFUSED_PREFIX)
    passages = [tuple(p) for p in a["passages"]]
    return {
        "refused": refused,
        "citations_valid": not agent.check_citations(a["answer"], passages),
        "retrieved_expected": q["kind"] == "unanswerable"
        or any(is_hit({"source": s, "location": l}, expected(q)) for s, l in passages),
    }


def judge(q: dict, a: dict) -> dict:
    prompt = RUBRIC.format(question=q["question"], reference=q["reference_answer"],
                           evidence=a["retrieved_text"][:20000] or "(nothing retrieved)", answer=a["answer"])
    for attempt in range(4):  # the free tier often answers 503 "high demand"
        try:
            res = index.client().models.generate_content(
                model=agent.MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
            break
        except (errors.ServerError, errors.ClientError) as e:
            if e.code not in (429, 503) or attempt == 3:
                raise
            time.sleep(30 * (attempt + 1))
    v = json.loads(res.text)
    return {"id": q["id"], "correct": int(v["correct"]), "supported": int(v["supported"]), "reason": v.get("reason", "")}


def grade_all(questions: list[dict], answers: dict) -> list[dict]:
    verdicts = read_jsonl(JUDGE)
    results = []
    for q in questions:
        a, c = answers[q["id"]], code_checks(q, answers[q["id"]])
        if q["kind"] == "unanswerable" or c["refused"]:
            # refusals are graded in code: right if unanswerable, wrong otherwise
            v = {"correct": int((q["kind"] == "unanswerable") == c["refused"]), "supported": 1, "reason": "graded in code"}
        else:
            if q["id"] not in verdicts:
                verdicts[q["id"]] = judge(q, a)
                append_jsonl(JUDGE, verdicts[q["id"]])
            v = verdicts[q["id"]]
        results.append({"id": q["id"], "kind": q["kind"], **c, "correct": v["correct"],
                        "supported": v["supported"], "reason": v["reason"], "guardrail": a["guardrail"],
                        "tokens": a["input_tokens"] + a["output_tokens"], "seconds": a["seconds"]})
    return results


def outcome(r: dict) -> str:
    if r["kind"] == "unanswerable":
        return "correct refusal" if r["refused"] else "answered unanswerable"
    if r["refused"]:
        return "wrong refusal"
    if not r["supported"]:
        return "unsupported"
    return "correct" if r["correct"] else "wrong"


# ---------- 5. agreement ----------

def agreement(results: list[dict]) -> dict:
    human = {h["id"]: h for h in read_csv(EVAL / "human_labels.csv")}
    pairs = [(r, human[r["id"]]) for r in results if r["id"] in human]
    if not pairs:
        return {"graded_by_me": 0}
    agree = lambda f: round(sum(int(h[f]) == r[f] for r, h in pairs) / len(pairs), 3)
    return {"graded_by_me": len(pairs), "agree_correct": agree("correct"), "agree_supported": agree("supported")}


# ---------- 6. add-document demo ----------

def add_document_demo(questions: list[dict], answers: dict) -> None:
    if not (DEMO_ID and DEMO_FILE):
        return
    q = next(q for q in questions if q["id"] == DEMO_ID)
    out = EVAL / "add_document_demo.md"
    if out.is_file():
        return  # done already; delete the file to redo it
    import server  # same code path the MCP tool uses
    added = server.add_document(str(ROOT / DEMO_FILE))
    after = asyncio.run(agent.run(q["question"], approve=lambda *_: False))
    out.write_text(
        f"# Add-document demo\n\n**Question ({q['id']}):** {q['question']}\n\n"
        f"## Before\n\n{answers[q['id']]['answer']}\n\n"
        f"## add_document\n\n{added}\n\n## After\n\n{after['answer']}\n", encoding="utf-8")


def main() -> None:
    questions = read_csv(QUESTIONS)
    if not questions:
        raise SystemExit("eval/questions.csv has no questions yet.")

    print("Retrieval (hit rate):")
    for row in retrieval(questions):
        print("  ", row)

    print("\nAnswering every question with the agent...")
    answers = answer_all(questions)
    results = grade_all(questions, answers)

    for r in results:
        r["outcome"] = outcome(r)
    counts = {o: sum(r["outcome"] == o for r in results) for o in
              ["correct", "wrong", "unsupported", "wrong refusal", "correct refusal", "answered unanswerable"]}
    n = len(results)
    summary = {
        "questions": n,
        "outcomes": counts,
        "citations_valid": round(sum(r["citations_valid"] for r in results) / n, 3),
        "guardrail_fired": sum(r["guardrail"] != "passed" for r in results),
        "avg_tokens": round(sum(r["tokens"] for r in results) / n),
        "avg_seconds": round(sum(r["seconds"] for r in results) / n, 1),
        "judge_vs_me": agreement(results),
    }
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failed = [r for r in results if r["outcome"] not in ("correct", "correct refusal")]
    # `cause` is filled in by hand; keep what is already there when this reruns
    causes = {r["id"]: r.get("cause", "") for r in read_csv(EVAL / "failures.csv")}
    with (EVAL / "failures.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "kind", "outcome", "retrieved_expected", "reason", "cause"])
        w.writeheader()
        for r in failed:
            w.writerow({**{k: r.get(k, "") for k in w.fieldnames}, "cause": causes.get(r["id"], "")})

    add_document_demo(questions, answers)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
