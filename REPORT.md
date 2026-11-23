# Assignment 3 Report

- **Name**: Javier López-Usero Fernández
- **Student ID**: <!-- TO WRITE (Javier) -->
- **Group**: <!-- TO WRITE (Javier): BBADBA 5A | BBADBA 5B | PPLEDBA 5A -->

> Draft note: every number below comes from the files in `eval/` and `index/`. Sections marked **TO WRITE (Javier)** need my own analysis or decisions. Delete these notes before submitting.

## Use case

Drone pilots and operators in the EU ask what the EU rules allow before they fly, buy or register a drone: height and distance limits, which subcategory (A1/A2/A3) and class (C0–C6) applies, registration, pilot competency and authorisations. They act on the answer directly, by flying, buying a drone or registering.

A wrong answer can mean an illegal or unsafe flight, a fine or an accident. "I don't know" only costs the pilot a call to the national aviation authority. So the assistant must refuse rather than guess, and every claim must be traceable to an article the pilot can check.

> **TO WRITE (Javier):** check this matches how I'd describe it; one sentence on why I picked this use case.

## Corpus

| Document | URL | Version / date | Licence | Words |
|---|---|---|---|---|
| Commission Implementing Regulation (EU) 2019/947 (operation of unmanned aircraft) | [EUR-Lex CELEX 02019R0947-20220404](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02019R0947-20220404) | consolidated 04/04/2022 | EU legal text; reuse free of charge under the Commission reuse policy (Decision 2011/833/EU), source acknowledged | ≈19,100 |
| Commission Delegated Regulation (EU) 2019/945 (UAS and third-country operators) | [EUR-Lex CELEX 02019R0945-20200809](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02019R0945-20200809) | consolidated 09/08/2020 | same as above | ≈20,400 |
| EASA AMC & GM to Part-UAS, Issue 1 (ED Decision 2019/021/R, Annex II) | [EASA PDF](https://www.easa.europa.eu/sites/default/files/dfu/AMC%20&%20GM%20to%20Part-UAS%20%E2%80%94%20Issue%201.pdf) | October 2019 | © EASA; reproduction authorised provided the source is acknowledged | 16,807 |
| **Total** | | | | **≈56,300** |

All three are public, contain no personal data, and are written for exactly this audience. Word counts are after extraction and cleaning; `main.py index` logs them per document. Added later for the add-document demo: Regulation (EU) 2021/664 on U-space (6,150 words), see [Adding a document](#adding-a-document).

## Ingestion

`index.py` downloads the three files into `corpus/` and splits each one along its own legal structure before chunking:

- **EUR-Lex HTML** (both regulations): text is split at each `Article N`, `ANNEX`, `PART`, `CHAPTER`, `SECTION`, `Appendix` heading and at each annex point such as `UAS.OPEN.020 UAS operations in subcategory A1`. Headings like `PART 2` take their title from the next line, so the location reads `PART 2 Requirements for a class C1 Unmanned aircraft system`.
- **EASA PDF**: text is split at each AMC/GM heading (for example `AMC1 UAS.OPEN.020(1) and (2) …`), and the page number is added to the location. Table-of-contents pages and the repeated header/footer lines (`Annex II to ED Decision 2019/021/R`, `Page x of y`) are dropped.
- **Cleaning**: EUR-Lex amendment markers (▼M1…) and scripts are removed, words hyphenated across PDF lines are rejoined, and whitespace is normalised.
- **Chunking**: word windows with 20% overlap, built *inside* each section, so a chunk never mixes two articles.

**Extraction problem fixed:** in the EUR-Lex HTML, list labels such as `(a)` or `(1)` sit alone on their own line. A plain line-by-line split left them as orphan fragments, separated from the rule they introduce. The extractor now glues a lone label to the line that follows, so `(a) weather conditions` stays one unit.
> **TO WRITE (Javier):** confirm this is the one I want to describe, or swap in another (TOC pages, amendment markers, hyphenation).

**Known leftovers:** a few EASA headings keep PDF spacing artefacts (for example `GM1 UAS.SPEC.1 00 The u se of certified equipment`). The A2 guidance headings come out of the PDF as `AMC1 UAS.OPEN.30(1)` rather than `030`. They are kept as extracted, so citations match the stored locations.

**Chunk metadata:** `source` (for example `Reg 2019/947`), `title`, `location` (article / UAS point / AMC-GM heading + page), `chunk` index and `file_hash` (SHA-256 of the file, used by `add_document`). The embedded text is `source | location` followed by the chunk, so headings help retrieval. Embeddings: `gemini-embedding-2`, 768 dimensions, cosine distance, ChromaDB in `index/chroma` (12 MB). Vectors are saved batch by batch, and only chunks without a saved vector are embedded.

| Design | Reg 2019/947 | Reg 2019/945 | EASA AMC/GM | Total |
|---|---|---|---|---|
| `w150` (150 words) | 184 | 192 | 168 | **544** (+58 from Reg 2021/664) |
| `w400` (400 words) | 94 | 94 | 95 | **283** |

## Test set

32 questions in `eval/questions.csv`: **20 single**, **6 multi** (two passages) and **6 unanswerable**.

Examples:
- *Single* (q05): "How far from residential areas must I stay when flying in subcategory A3?" Expected answer: `Reg 2019/947, UAS.OPEN.040`, at least 150 m horizontally from residential, commercial, industrial or recreational areas.
- *Multi* (q21): "I fly a C2 drone in A2: how close can I get to people with low speed mode, and how fast can it go in that mode?" Expected answer: `Reg 2019/947 UAS.OPEN.030` plus `Reg 2019/945 PART 3`, down to 5 m, with low-speed mode limited to 3 m/s.
- *Unanswerable* (q27): "How much does it cost to register as a UAS operator in Spain?" Fees are set nationally and are not in the corpus.

> **TO WRITE (Javier):** how I wrote the questions (from which documents, before tuning anything, how I chose the unanswerable ones so they stay close to the topic).

## Retrieval

Hit rate over the 26 answerable questions (a hit means an expected source and location appears among the top k):

| Design | hit@1 | hit@3 | hit@5 | Words added to the prompt (top 5) |
|---|---|---|---|---|
| **`w150`** | **0.846** | **1.000** | **1.000** | **633** |
| `w400` | 0.808 | 0.962 | 1.000 | 1,236 |

**Chosen design: `w150`.** It is better at k = 1 and k = 3, ties at k = 5, and adds about half as many words to each prompt.

For a multi question, the hit counts if *either* expected passage is retrieved. q23 (second passage `PART 16` never retrieved) and q26 (`UAS.OPEN.050` never retrieved) therefore count as hits.
> **TO WRITE (Javier):** one or two sentences on the choice and on this limitation of the metric.

## Tools and subagent

**`search(query)`**: "Search the EU drone (UAS) rules: Regulation 2019/947 (how drones may be flown: open/specific/certified categories, A1/A2/A3 subcategories, operator registration, remote pilot competency, operational authorisations, standard scenarios), Regulation 2019/945 (drone classes C0-C6, manufacturer duties, product requirements, remote ID) and EASA's AMC/GM guidance to Part-UAS. Use it for any question about operating, buying or building a drone in the EU. Returns the 5 most relevant passages, each numbered with its source and location (article, UAS point or AMC/GM heading). Cite them as [source, location]."

**`add_document(path)`**: "Add a local document (.pdf, .html or .txt) to the drone-rules index so search can find it. The file is extracted, chunked and embedded with the same pipeline as the corpus. The source name is the file name. If that file is already indexed with identical content nothing happens; if its content changed, the old chunks are replaced. Returns what was done and how many chunks were added."

One search result (q05, third of five passages):

```text
[3] Reg 2019/947, UAS.OPEN.040 UAS operations in subcategory A3 (score 0.781)
UAS operations in subcategory A3 shall comply with all of the following conditions: (1) be conducted
in an area where the remote pilot reasonably expects that no uninvolved person will be endangered …
(2) be conducted at a safe horizontal distance of at least 150 metres from residential, commercial,
industrial or recreational areas; …
```

**Subagent** (`.claude/agents/corpus-researcher.md`), limited to `mcp__corpus__search`:

```text
You research EU drone rules using only the `search` tool.
- Search first, every time. If the results don't answer the question, search again with other words (max 3 searches).
- Answer only from the returned passages. No outside knowledge, no national rules.
- Cite every claim as `[source, location]`, copied exactly from the search result.
- If the passages don't answer it, reply "The corpus does not cover this." and stop.
- Keep it short: a direct answer in a few sentences or bullets.
```

**Transcript from Claude Code:**
> **TO WRITE (Javier):** paste the transcript here (steps in my notes: open the repo in Claude Code, check `/mcp` shows `corpus`, ask "Use the corpus-researcher subagent: do I need to register a 300 g drone?", copy the exchange).

## Adding a document

`add_document` uses the file name as the source. If chunks with that source already exist and their `file_hash` equals the SHA-256 of the file, nothing happens ("already indexed with the same content"). If the content changed, the old chunks are deleted and the file is re-indexed, so an updated document never sits next to its stale version. New chunks go into the chosen `w150` index only, and only chunks without a saved vector are embedded.

**Unanswerable question (q31):** "What services must a U-space service provider offer to drone operators?"
**Document added:** `eval/extra/reg_2021_664_u-space.html`, Commission Implementing Regulation (EU) 2021/664 on a regulatory framework for the U-space (58 chunks).

- **Before:** "The corpus does not cover this."
- **After:** the four mandatory services (network identification, geo-awareness, UAS flight authorisation and traffic information), each cited to `[reg_2021_664_u-space.html, Article 3]`, plus Article 7 on providing them in all phases of the operation. The full answer is in `eval/add_document_demo.md`.

These 58 chunks are now part of the committed `w150` index, so q31 would be answerable if the agent were re-run. The evaluation results below use the answer from before the document was added. Re-running the retrieval evaluation with the extra chunks gives the same hit rates.

## Harness and guardrails

`agent.py` is a LangChain `create_agent` on `gemini-3.1-flash-lite` (temperature 0). It gets its tools from `server.py` over MCP and uses `AGENTS.md` as the system prompt. The rules are: search before answering (at most 3 searches); answer only from the passages; cite every claim as `[source, location]`; start with "The corpus does not cover this." when the passages don't answer; no national rules; mention the corpus version when dates matter; only add documents when the user asks with a path; be short.

- **Approval step:** a `wrap_tool_call` middleware intercepts every `add_document` call and asks `Allow? [y/N]` in the terminal. With no terminal (evaluation, `check.py`) the answer is no, and the agent receives "The user did not approve adding this document."
- **Call limits:** at most 6 model calls and 3 `search` calls per question.
- **Extra guardrail, citation check:** every `[source, location]` in the answer must match a passage that `search` returned in *this* run, with the same source and a location that matches at a word boundary. If any citation fails, the agent is told which ones and gets one retry; if the retry still fails, the answer is replaced by a refusal.

It fired on **10 of 32** questions. All 10 passed after the retry, and none had to be blocked. Example, q25 ("My drone weighs 600 g. Do I need to register, and could it qualify as class C1?"): the first draft cited `[Reg 2019/947, Article 14(5)(a)(i)]` and `[Reg 2019/945, Part 2(1)]`. Those are finer than any location the index stores, so they could not be checked against a returned passage. The retry cited `[Reg 2019/947, Article 14]` and `[Reg 2019/945, PART 2 Requirements for a class C1 Unmanned aircraft system]`, which match. In q04, the first draft used the passage number as the source (`[5, AMC1 UAS.OPEN.30(1)]`).

## Evaluation

The agent answered all 32 questions (`eval/answers.jsonl`).

**Code checks:**
- *refused*: the answer starts with "The corpus does not cover";
- *citations valid*: every citation matches a returned passage;
- *retrieved expected*: at least one expected passage was returned.

Refusals are graded in code: correct if the question is unanswerable, wrong otherwise.

**LLM judge** (same Gemini model, temperature 0, JSON output). It is used for every answer that is not a refusal. It sees the question, the reference answer, the retrieved passages and the answer, and grades two things:
- *correct* = 1 if the answer states the same rule as the reference (extra correct detail is fine); 0 if it contradicts the reference, misses the key point, or refuses although the reference gives an answer.
- *supported* = 1 if every factual claim appears in the retrieved passages; 0 if any number, condition or exception is not in them, even if it is true.

| Outcome | Count |
|---|---|
| Correct | 21 |
| Unsupported | 4 |
| Wrong | 1 |
| Correct refusal | 5 |
| Wrong refusal | 0 |
| Answered an unanswerable question | 1 |
| **Total** | **32** |

Citations were valid in 100% of answers.

**Agreement with my own grades:**
> **TO WRITE (Javier):** filled in after I grade 20+ answers in `eval/human_labels.csv` (agreement on *correct* and *supported*, and where I disagreed with the judge).

> **TO WRITE (Javier):** my reading of these results, including that 3 of the 4 "unsupported" answers (q07, q08, q10) were flagged only for the version sentence added by AGENTS.md rule 6, and the 4th (q21) mainly for it too. I chose to keep rule 6 and the judge rubric unchanged; explain why.

## Failures

| Question | Kind | Outcome | Cause |
|---|---|---|---|
| q07 | single | unsupported | <!-- TO WRITE --> |
| q08 | single | unsupported | <!-- TO WRITE --> |
| q10 | single | unsupported | <!-- TO WRITE --> |
| q21 | multi | unsupported | <!-- TO WRITE --> |
| q26 | multi | wrong | <!-- TO WRITE --> |
| q32 | unanswerable | answered unanswerable | <!-- TO WRITE --> |

> **TO WRITE (Javier):** counts by cause (retrieval miss / wrong answer from the right passage / unsupported claim / wrong refusal / bad test question) and what to fix first. Causes are picked in the grading page and copied into `eval/failures.csv`.

## Decision

**Cost and latency per question** (32 questions, `eval/answers.jsonl`):

| | Mean | Median |
|---|---|---|
| Tokens (input + output) | 4,367 | 3,294 |
| Seconds | 55.6 | 15.3 |

The mean latency is inflated by free-tier rate-limit waits: q04 took 802 s and q10 took 237 s. The median is the typical case.

> **TO WRITE (Javier):** would I let pilots rely on it, for what kind of question, and with which safeguard?

## AI use

> **TO WRITE (Javier):** what I used AI for when writing the code (tools, which parts).

Claude (Anthropic) was also used to review the repository, to build a web page where I entered my own grades and failure causes, to make `evaluate.py` keep the hand-filled `cause` column when it re-runs, and to draft the factual parts of this report from the files in `eval/` and `index/`.

> **TO WRITE (Javier):** confirm which decisions, the test set, the grades and the analysis are mine, as the brief requires.
