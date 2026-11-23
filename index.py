"""
Build the index (`main.py index`).

Downloads the drone-rules corpus into corpus/, splits it into sections
(article / UAS.xxx point / AMC-GM heading), chunks each section into word
windows, embeds with Gemini and stores everything in index/ (ChromaDB).

Two chunk designs are built side by side (150 and 400 words) so evaluate.py
can compare them. Vectors are saved batch by batch and only missing chunks
get embedded, so a crash or the daily quota never loses work.
"""

import hashlib
import re
import sys
import time
from pathlib import Path

import chromadb
import httpx
from bs4 import BeautifulSoup
from google import genai
from google.genai import errors, types
from pypdf import PdfReader

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / "index"

EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768  # smaller vectors keep index/ well under the 90 MB limit
BATCH = 20

DESIGNS = {"w150": 150, "w400": 400}  # design name -> words per chunk
CHOSEN_DESIGN = "w150"  # set from the hit-rate numbers in eval/retrieval.csv

SOURCES = [
    {
        "source": "Reg 2019/947",
        "title": "Commission Implementing Regulation (EU) 2019/947 on the rules and procedures for the operation of unmanned aircraft (consolidated 04/04/2022)",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02019R0947-20220404",
        "file": "c947_20220404.html",
    },
    {
        "source": "Reg 2019/945",
        "title": "Commission Delegated Regulation (EU) 2019/945 on unmanned aircraft systems and third-country operators (consolidated 09/08/2020)",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02019R0945-20200809",
        "file": "c945_20200809.html",
    },
    {
        "source": "EASA AMC-GM Part-UAS",
        "title": "EASA Acceptable Means of Compliance and Guidance Material to Part-UAS, Issue 1 (ED Decision 2019/021/R, Annex II)",
        "url": "https://www.easa.europa.eu/sites/default/files/dfu/AMC%20&%20GM%20to%20Part-UAS%20%E2%80%94%20Issue%201.pdf",
        "file": "easa_amc_gm_part_uas.pdf",
    },
]

# Headings that start a new section in the EUR-Lex text: a whole line like
# "Article 4" / "PART 2", or an annex point like "UAS.OPEN.020 UAS operations in ...".
HTML_HEADING = re.compile(
    r"^(?:(?:Article \d+[a-z]?|ANNEX|Appendix \d+|PART [A-Z0-9]+|CHAPTER [IVX]+|SECTION \d+)$|UAS\.[A-Z0-9-]+\.\d+\b)"
)
TITLED = re.compile(r"^(PART|CHAPTER|SECTION|Appendix) ")  # their title is on the next line
# AMC/GM headings in the EASA PDF, e.g. "AMC1 UAS.OPEN.020(1) and (2) UAS operations in subcategory A1".
PDF_HEADING = re.compile(r"^((?:AMC|GM)\d+ (?:UAS|Article)\S*(?: \S+)?)")
PDF_NOISE = re.compile(r"^(Annex I{1,2} to ED Decision 2019/021/R|Page \d+ of \d+)\s*$")


def log(*args) -> None:
    # stderr, so server.py can import this module without touching stdout
    print(*args, file=sys.stderr, flush=True)


# ---------- download & extract ----------

MIN_FILE_BYTES = 20_000  # every corpus file is far larger; an empty or blocked response is not


def download(src: dict) -> Path | None:
    """Local copy of a source, downloading it if needed. None if it can't be fetched:
    the vectors already committed in index/ are kept and nothing is re-embedded."""
    path = CORPUS_DIR / src["file"]
    if not path.exists() or path.stat().st_size < MIN_FILE_BYTES:
        CORPUS_DIR.mkdir(exist_ok=True)
        log(f"downloading {src['source']}")
        try:
            r = httpx.get(src["url"], headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True, timeout=120)
            r.raise_for_status()
        except httpx.HTTPError as e:
            log(f"  could not download {src['source']} ({e}); keeping its vectors in index/")
            return None
        if len(r.content) < MIN_FILE_BYTES:
            log(f"  {src['source']}: the site returned {len(r.content)} bytes; keeping its vectors in index/")
            return None
        path.write_bytes(r.content)
    return path


def html_sections(path: Path) -> list[dict]:
    """EUR-Lex HTML -> [{location, text}], one per article / annex point."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup.select(".modref, .arrow, script, style"):  # amendment markers like ▼M1
        tag.decompose()
    lines = [ln.replace("\xa0", " ").strip() for ln in soup.get_text("\n").splitlines()]
    lines = [ln for ln in lines if ln]

    # list labels "(a)" sit alone on a line; glue them to the text that follows
    merged = []
    for ln in lines:
        if merged and re.fullmatch(r"\(?[a-z0-9ivx]+[.)]", merged[-1]):
            merged[-1] += " " + ln
        else:
            merged.append(ln)

    sections, loc, buf = [], "Preamble", []
    for i, ln in enumerate(merged):
        if HTML_HEADING.match(ln):
            if buf:
                sections.append({"location": loc, "text": " ".join(buf)})
            loc, buf = ln, []
            nxt = merged[i + 1] if i + 1 < len(merged) else ""
            if TITLED.match(ln) and nxt and not HTML_HEADING.match(nxt) and len(nxt.split()) <= 15:
                loc += " " + nxt  # "PART 2" -> "PART 2 Requirements for a class C1 ..."
            loc = loc[:90]
        else:
            buf.append(ln)
    if buf:
        sections.append({"location": loc, "text": " ".join(buf)})

    # drop the EUR-Lex "documentation tool" disclaimer and empty title stubs
    return [s for s in sections if s["location"] != "Preamble" and len(s["text"].split()) > 5]


def pdf_sections(path: Path) -> list[dict]:
    """PDF -> [{location, text}]. Location = nearest AMC/GM heading + page."""
    sections, loc, buf, start_page = [], "Front matter", [], 1
    for pno, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        if text.count("....") > 5:  # table of contents page
            continue
        for ln in (l.strip() for l in text.splitlines()):
            if not ln or PDF_NOISE.match(ln):
                continue
            m = PDF_HEADING.match(ln)
            if m:
                if buf:
                    sections.append({"location": f"{loc} (p. {start_page})", "text": " ".join(buf)})
                loc, buf, start_page = ln[:90], [], pno
            else:
                buf.append(ln)
    if buf:
        sections.append({"location": f"{loc} (p. {start_page})", "text": " ".join(buf)})
    return sections


def clean(text: str) -> str:
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)  # words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def extract(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        secs = pdf_sections(path)
    elif ext in {".html", ".htm"}:
        secs = html_sections(path)
    else:  # plain text: one section per paragraph block
        blocks = [b for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
        secs = [{"location": f"paragraph {i + 1}", "text": b} for i, b in enumerate(blocks)]
    return [{**s, "text": clean(s["text"])} for s in secs]


# ---------- chunk ----------

def chunk_sections(sections: list[dict], size: int) -> list[dict]:
    """Word windows inside each section, so a chunk never mixes two articles."""
    step = size - size // 5  # 20% overlap
    chunks = []
    for s in sections:
        words = s["text"].split()
        for start in range(0, max(len(words) - size // 5, 1), step):
            chunks.append({"location": s["location"], "text": " ".join(words[start:start + size])})
    return chunks


def chunk_file(path: Path, source: str, title: str, design: str) -> list[dict]:
    """Extract + chunk one file; each chunk gets an id and its metadata."""
    digest = file_hash(path)
    chunks = chunk_sections(extract(path), DESIGNS[design])
    for i, c in enumerate(chunks):
        c["id"] = hashlib.sha1(f"{source}|{design}|{i}|{c['text']}".encode()).hexdigest()[:16]
        c["meta"] = {"source": source, "title": title, "location": c["location"], "chunk": i, "file_hash": digest}
    return chunks


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- embed & store ----------

_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=load_api_key())
    return _client


def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed with a few retries on rate limits (per-minute limits clear quickly)."""
    for attempt in range(5):
        try:
            res = client().models.embed_content(
                model=EMBED_MODEL,
                # one Content per text: a plain list would be embedded as a single item
                contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM),
            )
            return [e.values for e in res.embeddings]
        except errors.ClientError as e:
            if e.code != 429 or attempt == 4:
                raise
            log(f"rate limited, waiting {20 * (attempt + 1)}s")
            time.sleep(20 * (attempt + 1))


def collection(design: str):
    db = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    return db.get_or_create_collection(f"drones_{design}", metadata={"hnsw:space": "cosine"})


def embed_and_store(chunks: list[dict], design: str) -> int:
    """Embed only chunks not yet stored; save each batch straight away. Returns how many were added."""
    col = collection(design)
    have = set(col.get(ids=[c["id"] for c in chunks], include=[])["ids"]) if chunks else set()
    todo = [c for c in chunks if c["id"] not in have]
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        # title + location go into the embedded text so headings help retrieval
        vecs = embed([f"{c['meta']['source']} | {c['location']}\n{c['text']}" for c in batch])
        col.add(ids=[c["id"] for c in batch], embeddings=vecs,
                documents=[c["text"] for c in batch], metadatas=[c["meta"] for c in batch])
        log(f"  {design}: {min(i + BATCH, len(todo))}/{len(todo)} embedded")
    return len(todo)


def loc_match(wanted: str, location: str) -> bool:
    """'PART 1' matches 'PART 1 Requirements...' but not 'PART 12'."""
    w, l = wanted.strip().lower(), location.strip().lower()
    return l.startswith(w) and (len(l) == len(w) or not l[len(w)].isalnum())


def retrieve(query: str, design: str = CHOSEN_DESIGN, k: int = 5, query_vec: list[float] | None = None) -> list[dict]:
    """Top-k chunks for a query. Pass query_vec to reuse an embedding across designs."""
    vec = query_vec or embed([query], task="RETRIEVAL_QUERY")[0]
    res = collection(design).query(query_embeddings=[vec], n_results=k)
    return [{"id": i, "text": d, **m, "score": round(1 - dist, 3)}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]


def main() -> None:
    for src in SOURCES:
        path = download(src)
        if path is None:
            continue
        for design in DESIGNS:
            chunks = chunk_file(path, src["source"], src["title"], design)
            words = sum(len(s["text"].split()) for s in extract(path))
            log(f"{src['source']}: {words} words, {len(chunks)} chunks ({design})")
            embed_and_store(chunks, design)
    for design in DESIGNS:
        log(f"{design}: {collection(design).count()} chunks in index/")


if __name__ == "__main__":
    main()
