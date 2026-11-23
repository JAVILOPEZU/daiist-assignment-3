# Assignment 3 Report

- **Name**: Javier López-Usero Fernández
- **Student ID**: 18633
- **Group**: BBADBA 5B

## Use case

Drone pilots and operators in the EU ask what the EU rules allow before they fly, buy or register a drone: height and distance limits, which subcategory (A1/A2/A3) and class (C0–C6) applies, registration, pilot competency and authorisations. They act on the answer directly, by flying, buying a drone or registering.

A wrong answer can mean an illegal or unsafe flight, a fine or an accident. "I don't know" only costs the pilot a call to the national aviation authority. So the assistant must refuse rather than guess, and every claim must be traceable to an article the pilot can check.

## Corpus

| Document | URL | Version / date | Licence | Words |
|---|---|---|---|---|
| Commission Implementing Regulation (EU) 2019/947 (operation of unmanned aircraft) | [EUR-Lex CELEX 02019R0947-20220404](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02019R0947-20220404) | consolidated 04/04/2022 | EU legal text; reuse free of charge under the Commission reuse policy (Decision 2011/833/EU), source acknowledged | 19,100 |
| Commission Delegated Regulation (EU) 2019/945 (UAS and third-country operators) | [EUR-Lex CELEX 02019R0945-20200809](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02019R0945-20200809) | consolidated 09/08/2020 | same as above | 20,428 |
| EASA AMC & GM to Part-UAS, Issue 1 (ED Decision 2019/021/R, Annex II) | [EASA PDF](https://www.easa.europa.eu/sites/default/files/dfu/AMC%20&%20GM%20to%20Part-UAS%20%E2%80%94%20Issue%201.pdf) | Issue 1, 9 October 2019 (61 pages) | © EASA; reproduction authorised provided the source is acknowledged | 16,807 |
| **Total** | | | | **56,335** |

All three are public, contain no personal data, and are written for exactly this audience. Word counts are after extraction and cleaning; `main.py index` logs them per document. The EASA count comes from a fresh extraction. The two regulations' counts are rebuilt from the chunks stored in `index/`, and the 150- and 400-word indexes give the same totals. Added later for the add-document demo: Regulation (EU) 2021/664 on U-space (6,150 words), see [Adding a document](#adding-a-document).

## Ingestion

`index.py` downloads the three files into `corpus/` (a source that can't be downloaded is skipped with a warning and its stored vectors are kept) and splits each one along its own legal structure before chunking:

- **EUR-Lex HTML** (both regulations): text is split at each `Article N`, `ANNEX`, `PART`, `CHAPTER`, `SECTION`, `Appendix` heading and at each annex point such as `UAS.OPEN.020 UAS operations in subcategory A1`. Headings like `PART 2` take their title from the next line, so the location reads `PART 2 Requirements for a class C1 Unmanned aircraft system`.
- **EASA PDF**: text is split at each AMC/GM heading (for example `AMC1 UAS.OPEN.020(1) and (2) …`), and the page number is added to the location. Table-of-contents pages and the repeated header/footer lines (`Annex II to ED Decision 2019/021/R`, `Page x of y`) are dropped.
- **Cleaning**: EUR-Lex amendment markers (▼M1…) and scripts are removed, words hyphenated across PDF lines are rejoined, and whitespace is normalised.
- **Chunking**: word windows of 150 or 400 words with 20% overlap (step 120 or 320), built *inside* each section, so a chunk never mixes two articles.

**Extraction problem fixed:** in the EUR-Lex HTML, list labels such as `(a)` or `(1)` sit alone on their own line. A plain line-by-line split left them as orphan fragments, separated from the rule they introduce. The extractor now glues a lone label to the line that follows, so `(a) weather conditions` stays one unit.

**Known leftovers:** a few EASA headings keep PDF spacing artefacts (for example `GM1 UAS.SPEC.1 00 The u se of certified equipment`). The A2 guidance headings come out of the PDF as `AMC1 UAS.OPEN.30(1)` rather than `030`. They are kept as extracted, so citations match the stored locations.

**Chunk metadata:** `source` (for example `Reg 2019/947`), `title`, `location` (article / UAS point / AMC-GM heading + page), `chunk` index and `file_hash` (SHA-256 of the file, used by `add_document`). The embedded text is `source | location` followed by the chunk, so headings help retrieval. Embeddings: `gemini-embedding-2`, 768 dimensions, cosine distance, ChromaDB in `index/chroma` (13.0 MB, as `check.py` measures it; limit 90 MB). Vectors are saved batch by batch, and only chunks without a saved vector are embedded.

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

I wrote the 32 questions myself, working from a summary of the most important regulations. I covered the most restrictive topics and the ones pilots deal with daily. The unanswerable questions stay close to the topic: national fees, fines, UK rules, wind limits, U-space services and authorisation deadlines.

## Retrieval

Hit rate over the 26 answerable questions (a hit means an expected source and location appears among the top k):

| Design | hit@1 | hit@3 | hit@5 | multi_all@1 | multi_all@3 | multi_all@5 | Words added to the prompt (top 5) |
|---|---|---|---|---|---|---|---|
| **`w150`** | **0.846** | **1.000** | **1.000** | 0.000 | 0.167 | 0.333 | **633** |
| `w400` | 0.808 | 0.962 | 1.000 | 0.000 | 0.333 | 0.500 | 1,236 |

**Chosen design: `w150`.** It is better at k = 1 and k = 3, ties at k = 5, and adds about half as many words to each prompt. The price is multi-passage retrieval: `w400` returns both passages of a multi question more often (`multi_all@5` 0.500 against 0.333). I keep `w150` and accept that trade-off.

For a multi question, hit@k counts a hit if *either* expected passage is retrieved, which overstates retrieval for the 6 multi questions. `multi_all@k` counts only those with *every* expected passage in the top k (`eval/retrieval.csv`). It is 0 at k = 1 by construction, since a chunk never spans two sections. At k = 5, `w150` finds both passages for 2 of 6 (q24, q26) and `w400` for 3 of 6 (q21, q22, q24). With `w150` the top 5 misses `PART 3` in q21, `UAS.OPEN.070` in q22, `PART 16` in q23 and `Article 14` in q25.

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

**Transcript from Claude Code** (CLI 2.1.330 in headless mode, `claude -p`, run on 23 November 2026 from a clean clone of this repo; the `corpus` server loaded from `.mcp.json`):

```text
$ claude -p "Use the corpus-researcher subagent to answer this question: Do I need to register a 300 g drone to fly it in the open category?"
  (Claude Code 2.1.330, model claude-opus-5-5; MCP server "corpus" from .mcp.json: connected)

● Agent(subagent_type="corpus-researcher")
  prompt: "Question: Do I need to register a 300 g drone to fly it in the open category (EU rules, Regulations 2019/947 and 2019/945)? …"
  ⎿ mcp__corpus__search(query="UAS operator registration open category MTOM 250 g kinetic energy 80 J sensor personal data toy")
  ⎿ mcp__corpus__search(query="operator registration number displayed on unmanned aircraft")
  ⎿ mcp__corpus__search(query="C1 class remote pilot online training theoretical knowledge examination subcategory A1")
     results for "UAS operator registration open category …": [1] Reg 2019/947, Article 14 (0.843); [2] Reg 2019/947, Article 14 (0.816); [3] Reg 2019/947, PART A UAS OPERATIONS IN THE ‘OPEN’ CATEGORY (0.816); [4] Reg 2019/947, ANNEX (0.804); [5] Reg 2019/947, Article 4 (0.797) (passage text omitted)
     results for "operator registration number displayed o…": [1] Reg 2019/947, Article 14 (0.709); [2] Reg 2019/947, Article 14 (0.706); [3] Reg 2019/945, PART 2 Requirements for a class C1 Unmanned aircraft system (0.701); [4] Reg 2019/947, Article 14 (0.694); [5] Reg 2019/945, Article 40 (0.691) (passage text omitted)
     results for "C1 class remote pilot online training th…": [1] Reg 2019/947, UAS.OPEN.020 UAS operations in subcategory A1 (0.819); [2] Reg 2019/947, UAS.OPEN.020 UAS operations in subcategory A1 (0.813); [3] Reg 2019/947, ANNEX (0.796); [4] Reg 2019/947, UAS.OPEN.030 UAS operations in subcategory A2 (0.782); [5] Reg 2019/947, PART A UAS OPERATIONS IN THE ‘OPEN’ CATEGORY (0.782) (passage text omitted)
  ⎿ mcp__corpus__search(query="class C0 MTOM less than 250 g; class C1 MTOM less than 900 g; subcategory A3 remote pilot training")
     results for "class C0 MTOM less than 250 g; class C1 …": [1] Reg 2019/947, UAS.OPEN.040 UAS operations in subcategory A3 (0.852); [2] Reg 2019/947, UAS.OPEN.020 UAS operations in subcategory A1 (0.848); [3] Reg 2019/947, UAS.OPEN.040 UAS operations in subcategory A3 (0.837); [4] Reg 2019/945, PART 4 Requirements for a class C3 Unmanned aircraft system (0.837); [5] Reg 2019/947, UAS.OPEN.020 UAS operations in subcategory A1 (0.832) (passage text omitted)
  ⎿ corpus-researcher report:
    **Yes. A 300 g drone flown in the open category requires you, the operator, to register.** It is over the 250 g threshold, so its class, its camera and any other feature do not change that.

    **(1) The Article 14 registration triggers**
    - In the open category, UAS operators must register when they fly an unmanned aircraft that (i) has an MTOM of 250 g or more, or could transfer more than 80 J of kinetic energy to a person on impact, or (ii) "is equipped with a sensor able to capture personal data, unless it complies with Directive 2009/48/EC" [Reg 2019/947, Article 14].
    - Any drone flown in the specific category triggers registration, whatever its mass [Reg 2019/947, Article 14].
    - Some of the terms you asked about are not in the returned text:
      - The Article 14 passage names only "Directive 2009/48/EC". It does not use the word "toy".
      - It says "personal data" but does not mention GDPR.
    - A 300 g drone already meets criterion (i), so the camera and toy questions don't affect it.

    **(2) Operator vs. drone, and certified drones**
    - Registration is for the operator, not the aircraft. Operators register in the Member State where they live (natural persons) or have their principal place of business (legal persons). An operator cannot be registered in more than one Member State at a time [Reg 2019/947, Article 14].
    - Member States issue a unique registration number for operators and for any UAS that requires registration [Reg 2019/947, Article 14].
    - Drones whose design is subject to certification are the exception. Their owner must register the aircraft itself, with a nationality and registration mark that follows ICAO Annex 7 [Reg 2019/947, Article 14].

    **(3) Does the class matter for a 300 g drone?**
    - No. Article 14 bases registration on mass, energy and sensor, not on class [Reg 2019/947, Article 14].
    - Class does decide which subcategory you can fly in. A1 is open to C0 and C1 drones (C1 needs active, up-to-date direct remote ID and geo-awareness). Privately built drones only qualify for A1 if they are under 250 g and slower than 19 m/s [Reg 2019/947, UAS.OPEN.020 UAS operations in subcategory A1].
    - The returned passages do not give the C0 and C1 mass limits (under 250 g and under 900 g), so I can't confirm them from the corpus.

    **(4) Related obligations**
    - **Showing the number:** operators must display their registration number on every aircraft that meets the Article 14(5) conditions [Reg 2019/947, Article 14].
    - **Remote ID:** a C1 drone must let you upload the operator registration number and must broadcast it during flight [Reg 2019/945, PART 2 Requirements for a class C1 Unmanned aircraft system].
    - **Pilot competency, A1 with C1:** you need an online training course and an online theory exam. The exam has 40 multiple-choice questions and you need at least 75% to pass. It covers air safety, airspace restrictions, aviation regulation, human performance limitations, operational procedures, UAS general knowledge, privacy and data protection, insurance, and security [Reg 2019/947, UAS.OPEN.020 UAS operations in subcategory A1].
    - **Pilot competency, A3:** the same online training and exam, plus familiarity with the manufacturer's instructions [Reg 2019/947, UAS.OPEN.040 UAS operations in subcategory A3].
    - **A2:** this also requires a certificate of remote pilot competency [Reg 2019/947, UAS.OPEN.030 UAS operations in subcategory A2].

    **Not covered by the corpus:** the exact C0 and C1 class mass limits, any "toy" wording, any mention of GDPR, and national procedures or fees for registering.

● Claude Code answer:
  **Yes. You have to register to fly a 300 g drone in the open category.** I checked this with the corpus-researcher agent against EU Regulations 2019/947 and 2019/945.

  **Why:** Under Article 14 of Reg. 2019/947, you must register if your drone:
  - weighs 250 g or more (maximum take-off mass), or could hit a person with more than 80 J of energy, **or**
  - has a sensor that can capture personal data, such as a camera, unless it meets the EU Toy Safety Directive (2009/48/EC).

  A 300 g drone is over the 250 g limit, so you must register whether or not it has a camera.

  **Key points:**
  - **You register, not the drone.** You register as the *operator* in the EU country where you live, or where your business is based, and only in one country at a time. You get one registration number that covers all your drones. The exception is drones whose design has to be certified, which are registered as aircraft.
  - **Put your number on the drone.** It must be shown on every drone that needs registration. If the drone is class C1, you also upload the number so the drone broadcasts it in flight (remote ID).
  - **The drone's class decides where you can fly, not whether you register.**
    - A C1 drone can fly in subcategory A1.
    - A home-built drone can only fly in A1 if it is under 250 g and slower than 19 m/s.
  - **You also need pilot training.**
    - For A1 (with a C1 drone) or A3: an online course and an online exam with 40 multiple-choice questions, pass mark 75%.
    - For A2: you also need a certificate of remote pilot competency.

  **Not in the regulations:** how to register and what it costs. That depends on your country's aviation authority, for example AESA in Spain. The searched text also didn't give the exact C0 and C1 weight limits.

  (2 turns, 28.7 s, $0.49 of Claude usage; the subagent made 4 search calls)
```

The subagent made 4 searches, although its instructions allow at most 3. In Claude Code that limit is only an instruction, while `agent.py` enforces it with middleware. The subagent's report cites the regulation for each rule and lists what the corpus does not cover. The main session's closing answer adds two details that are not in the passages ("such as a camera" and "AESA in Spain").

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
- **Call limits:** at most 6 model calls and 3 `search` calls per agent run. A 4th search is blocked and the agent gets "Tool call limit exceeded" (q23 requested 4; 3 ran). The citation retry is a second run with fresh limits.
- **Extra guardrail, citation check:** every `[source, location]` in the answer must match a passage that `search` returned in *this* run, with the same source and a location that matches at a word boundary. If any citation fails, the agent is told which ones and gets one retry; if the retry still fails, the answer is replaced by a refusal.

It fired on **10 of 32** questions. All 10 passed after the retry, and none had to be blocked. Example, q25 ("My drone weighs 600 g. Do I need to register, and could it qualify as class C1?"): the first draft cited `[Reg 2019/947, Article 14(5)(a)(i)]` and `[Reg 2019/945, Part 2(1)]`. Those are finer than any location the index stores, so they could not be checked against a returned passage. The retry cited `[Reg 2019/947, Article 14]` and `[Reg 2019/945, PART 2 Requirements for a class C1 Unmanned aircraft system]`, which match. In q04, the first draft used the passage number as the source (`[5, AMC1 UAS.OPEN.30(1)]`).

## Evaluation

The agent answered all 32 questions (`eval/answers.jsonl`).

**Code checks:**
- *refused*: the answer starts with "The corpus does not cover";
- *citations valid*: every citation matches a returned passage;
- *retrieved expected*: at least one expected passage was returned.

Refusals are graded in code: correct if the question is unanswerable, wrong otherwise.

**LLM judge** (same Gemini model, temperature 0, JSON output). It grades the 26 answers to answerable questions that are not refusals. It sees the question, the reference answer, the retrieved passages and the answer, and grades two things:
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

**Agreement with my own grades:** I graded 20 answers myself (q01–q20, `eval/human_labels.csv`).

| | Agreement with the judge |
|---|---|
| *correct* | **100%** (20 / 20) |
| *supported* | **85%** (17 / 20) |

All 20 are single-passage questions. The three disagreements are q07, q08 and q10. I graded them *supported*. The judge graded them *unsupported* because of the closing sentence naming the corpus versions, which comes from AGENTS.md rule 6 rather than from a retrieved passage. In my grading I treat that sentence as a disclaimer required by rule 6, not as a factual claim, so it does not make an answer unsupported. I applied this to all 20 answers; the judge applies the rubric literally, but only to 4 of the 25 answers that contain the note.

Three of the four "unsupported" answers (q07, q08, q10) were flagged only for the version sentence that AGENTS.md rule 6 adds, and so was the fourth (q21). I kept rule 6, the judge rubric and the strict refusal check unchanged. The aim of the evaluation is to show how the system behaves, not to raise its score: strict rules that produce both positive and negative cases explain more than rules changed until every case passes.

**Judge limitations:**
- The judge is the same model as the agent (`gemini-3.1-flash-lite`), so it may favour answers written the way it writes (self-preference bias).
- It is inconsistent on the version sentence: that sentence is in 25 of the 26 judged answers, and the judge marked only 4 of them unsupported (q07, q08, q10, q21).
- Its reason for q21 is wrong. It says the answer cites Regulations 2019/947 and 2019/945, which were not retrieved. The answer cites only EASA `AMC1 UAS.OPEN.30(1)`, the two regulations are named only in the version sentence, and Reg 2019/945 `PART 2` and `PART 3` were among the retrieved passages (`PART 3` was retrieved but not cited).

## Failures

| Question | Kind | Outcome | Cause |
|---|---|---|---|
| q07 | single | unsupported | unsupported claim |
| q08 | single | unsupported | unsupported claim |
| q10 | single | unsupported | unsupported claim |
| q21 | multi | unsupported | unsupported claim |
| q26 | multi | wrong | retrieval miss |
| q32 | unanswerable | answered unanswerable | bad test question |

| Cause | Count |
|---|---|
| Unsupported claim | 4 |
| Retrieval miss | 1 |
| Bad test question | 1 |
| Wrong answer from the right passage | 0 |
| Wrong refusal | 0 |

What each failure shows:
- **q07, q08, q10 and q21:** the core facts are cited to the right passages. The sentence the judge flagged is the closing version note that AGENTS.md rule 6 asks for. By my grading convention that note is not a factual claim, so these four are failures for the judge, not real errors.
- **q26:** the agent searched once, with its own query, and `UAS.OPEN.050` point (7) (everyone involved in an A2 or A3 operation must be informed of the risks and explicitly agree) was not among the results. The agent then said the rules don't cover that part. With the question text as the query, the `w150` retrieval evaluation does return `UAS.OPEN.050` in the top 5.
- **q32:** the agent replied "The corpus does not specify a set number of days…" and cited `UAS.SPEC.040`, which says the authority issues the authorisation "without undue delay". The code only counts answers starting "The corpus does not cover" as refusals. I classify it as a bad test question: the corpus does answer it (no fixed number of days, only "without undue delay"), so it should not have been labelled unanswerable.

**What to fix first:** the answers don't define key terms before giving the evidence. In my own grading I noted that q02 should say what the open category is and q03 should define a toy. I would fix that first.

## Decision

**Cost and latency per question** (32 questions, `eval/answers.jsonl`):

| | Mean | Median |
|---|---|---|
| Tokens (input + output) | 4,367 | 3,294 |
| Seconds | 55.6 | 15.3 |

The mean is pulled up by two outliers, most likely free-tier rate-limit waits: q04 (802 s) and q10 (237 s). The median is the typical case. Tokens are the agent's model tokens; the judge's are not logged. The evaluation run made 37 searches (38 requested, 1 blocked), each one query embedding.

**Would I let users rely on it?** Yes: as a way to find and read the EU rule that applies, not as a replacement for the pilot's judgement. On the 32 questions, 26 were handled correctly (21 correct answers and 5 correct refusals) and every citation pointed to a passage that was actually retrieved. The four "unsupported" answers were flagged only for the version note. The one real error was q26, which left out the consent rule for people taking part in A2; q32 is a test-question problem (see Failures). The safeguard is the pilot. They have the knowledge the assistant lacks to interpret ambiguous terms, and each answer cites the article they can open and check before flying.

## AI use

AI wrote the code.

Claude (Anthropic) was also used to review the repository, to build a grading page and to walk me through the answers one by one (with neutral summaries of each answer and its passages) while I gave my own grades and failure causes, to make `evaluate.py` keep the hand-filled `cause` column when it re-runs, and to draft the factual parts of this report from the files in `eval/` and `index/`. Claude also turned my notes for the analysis sections (test set, evaluation, failures, decision) into sentences.

In a final review, Claude also added the `multi_all@k` metric to `evaluate.py`, reusing the saved query vectors. It made `index.py` skip a source it can't download instead of crashing. It re-ran `main.py index` and `main.py evaluate` with no new embeddings or judge calls, and ran `check.py` on a fresh clone, which passed. It corrected facts in this report (word counts, call limits, judge scope and limitations, the q26 and q32 evidence) and built a second blind grading sheet for q21–q32. At my request it produced the subagent transcript by running the Claude Code CLI on a clean clone of this repo with my Gemini key. It also rewrote the ship decision and the evaluation framing from my earlier notes in this chat, so they agree with the use case.

I wrote the test set myself and graded 20 answers myself. I chose the cause of each failure, the decision to keep rule 6 and the strict checks, what to fix first and the ship decision.
