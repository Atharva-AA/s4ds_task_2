import os
import glob
import hashlib
import threading
import time
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from llmops.metrics import compute_grounding_score, detect_hallucination
from llmops.hallucination_guard import guard as hallucination_guard
from llmops.prompt_manager import PromptManager
from observability.worker import enqueue_event, start_worker, stop_worker
from observability.logger import init_telemetry

# ── Config ───────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")

logger = logging.getLogger("uvicorn.error")
prompt_mgr = PromptManager()

# ── Globals ──────────────────────────────────────────────────────────
rag_chain = None
vector_store = None
retriever = None
_embedding = None
_store_lock = threading.Lock()          # guards vector_store writes
_ingested_hashes: set[str] = set()     # dedup: sha256 of content


# ── Helper functions ─────────────────────────────────────────────────
def get_embedding():
    """Return the global embedding instance, creating it once if needed."""
    global _embedding
    if _embedding is None:
        _embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _embedding


def load_pdfs(data_dir: str):
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        return []
    docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    return docs


def _get_splitter() -> CharacterTextSplitter:
    return CharacterTextSplitter(
        separator="\n", chunk_size=768, chunk_overlap=128, length_function=len
    )


def build_vector_store(docs):
    chunks = _get_splitter().split_documents(docs)
    emb = get_embedding()
    vs = Chroma.from_documents(
        documents=chunks, embedding=emb, persist_directory=CHROMA_DIR
    )
    return vs


_MIN_CONTENT_CHARS = 200   # ignore pages with less content than this
_MAX_CHUNKS_PER_DOC = 300  # cap chunks per single document to avoid runaway ingestion


def _content_hash(text: str) -> str:
    """SHA-256 of text content for deduplication."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def add_documents_to_store(docs: list, rebuild_chain: bool = True) -> int:
    """Incrementally add new documents to the existing vector store.

    - Thread-safe via _store_lock
    - Deduplicates by content hash
    - Caps chunks per document
    - Creates the store if it doesn't exist yet
    - Returns number of chunks actually added
    """
    global vector_store, rag_chain, _ingested_hashes

    if not docs:
        return 0

    # Deduplicate: skip docs whose content we've already ingested
    new_docs = []
    for doc in docs:
        h = _content_hash(doc.page_content)
        if h not in _ingested_hashes:
            new_docs.append((doc, h))

    if not new_docs:
        logger.info("[ingestion] all documents already ingested, skipping")
        return 0

    chunks = _get_splitter().split_documents([d for d, _ in new_docs])
    if not chunks:
        return 0

    # Cap chunks to avoid runaway ingestion from huge documents
    chunks = chunks[:_MAX_CHUNKS_PER_DOC]

    emb = get_embedding()
    with _store_lock:
        if vector_store is None:
            vector_store = Chroma(
                persist_directory=CHROMA_DIR, embedding_function=emb
            )
        try:
            vector_store.add_documents(chunks)
            # Mark hashes as ingested only after successful write
            for _, h in new_docs:
                _ingested_hashes.add(h)
        except Exception as exc:
            logger.error(f"[ingestion] Chroma write failed: {exc}")
            raise

        if rebuild_chain:
            rag_chain = build_chain(vector_store)

    return len(chunks)


def _validate_url(url: str) -> None:
    """Raise ValueError if URL is not a valid HTTP/HTTPS URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme '{parsed.scheme}' — only http/https allowed")
    if not parsed.netloc:
        raise ValueError(f"Invalid URL — missing host: {url}")


# Ordered list of CSS selectors that typically contain main legal/article content.
# Tried in order — first match with sufficient text wins.
_CONTENT_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".judgement", ".judgment", ".case-content", ".case-body",
    ".act-content", ".act-body", ".legal-text", ".law-content",
    ".content-area", ".main-content", ".page-content", ".post-content",
    ".entry-content", ".article-body", ".doc-content",
    "#content", "#main", "#main-content", "#judgement", "#judgment",
    "div.container > div.row",   # common Bootstrap law sites
]

_BOILERPLATE_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "form", "button", "iframe", "noscript", "figure", "figcaption",
]


def _extract_text(soup: BeautifulSoup) -> str:
    """Try content selectors first; fall back to full body text."""
    for selector in _CONTENT_SELECTORS:
        try:
            node = soup.select_one(selector)
            if node:
                text = node.get_text(separator="\n", strip=True)
                if len(text) >= _MIN_CONTENT_CHARS:
                    return text
        except Exception:
            continue

    # Fallback: strip boilerplate from full page
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def scrape_url(url: str) -> list[Document]:
    """Scrape text content from a URL and return as LangChain Documents.

    Strategy:
    1. Validate URL scheme
    2. Fetch with browser-like headers + encoding detection
    3. Try targeted CSS selectors for main content (works across different site structures)
    4. Fall back to full-body extraction if no selector matches
    5. Enforce minimum content length (rejects login walls, empty pages)
    """
    _validate_url(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }
    resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
    resp.raise_for_status()

    # Skip non-HTML responses (PDFs, images, JSON APIs, etc.)
    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type and "text" not in content_type:
        raise ValueError(f"Unsupported content type '{content_type}' — only HTML pages supported")

    # Detect encoding from HTTP headers or meta charset
    resp.encoding = resp.apparent_encoding or "utf-8"
    html = resp.text

    # Warn if page looks JS-rendered (very little text in raw HTML)
    if html.count("<p") < 3 and html.count("<div") > 20:
        logger.warning(f"[scraper] {url} may be JS-rendered — content may be incomplete")

    soup = BeautifulSoup(html, "html.parser")

    # Remove cookie banners and GDPR popups by common class names
    for node in soup.select(".cookie-banner, .cookie-notice, .gdpr, .popup, .modal, .overlay"):
        node.decompose()

    text = _extract_text(soup)

    # Collapse blank lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    text = "\n".join(lines)

    if len(text) < _MIN_CONTENT_CHARS:
        raise ValueError(
            f"Page content too short ({len(text)} chars) — "
            "likely a login wall, JS-rendered page, or empty page"
        )

    return [Document(
        page_content=text,
        metadata={
            "source": url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "content_length": len(text),
        }
    )]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vs):
    global retriever
    llm = ChatGroq(
        temperature=0.5,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
    )
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    active_prompt = prompt_mgr.get_active_prompt()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | active_prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, vector_store
    logger.info("Loading PDFs from data/ ...")
    docs = load_pdfs(DATA_DIR)
    if docs:
        logger.info(f"{len(docs)} pages loaded. Building vector store ...")
        vector_store = build_vector_store(docs)
        rag_chain = build_chain(vector_store)
        logger.info(f"RAG chain ready (prompt: {prompt_mgr.active_name})")
    else:
        logger.info("No PDFs found – upload via /upload first.")

    # Initialise telemetry — reuse already-loaded embedding, never reload
    init_telemetry(get_embedding())
    await start_worker()

    yield

    await stop_worker()
    logger.info("Shutting down.")


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="RAG ChatBot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Latency middleware ────────────────────────────────────────────────
@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    request.state.t0 = time.time()
    request.state.qid = None
    response = await call_next(request)
    return response


# ── Schemas ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    query_id: str


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["up", "down"]


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "chain_ready": rag_chain is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if rag_chain is None:
        raise HTTPException(
            status_code=503, detail="RAG chain not initialised. Upload PDFs first."
        )

    t0 = time.time()
    qid = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Log query (fire-and-forget)
    try:
        enqueue_event({"kind": "query", "qid": qid, "text": req.question, "timestamp": timestamp})
    except Exception:
        pass

    # 2. Retrieve context
    context_docs = retriever.invoke(req.question)
    context_text = format_docs(context_docs)

    # 3. Compute avg retrieval distance
    try:
        raw = vector_store.similarity_search_with_score(req.question, k=3)
        distances = [score for _, score in raw]
        avg_distance = sum(distances) / len(distances) if distances else 0.0
    except Exception:
        avg_distance = 0.0

    # 4. Log retrieval (fire-and-forget)
    try:
        enqueue_event({"kind": "retrieval", "qid": qid, "avg_distance": avg_distance, "k": 3})
    except Exception:
        pass

    # 5. Run LLM chain
    answer = rag_chain.invoke(req.question)
    latency_ms = (time.time() - t0) * 1000

    # 6. Compute scores
    grounding = compute_grounding_score(answer, context_text)
    hal_score = detect_hallucination(answer, context_text)

    # 7. Hallucination guard
    guard_result = hallucination_guard(answer, context_text)
    final_answer = guard_result["answer"]
    blocked = guard_result["blocked"]

    # 8. Log response (fire-and-forget)
    try:
        enqueue_event({"kind": "response", "qid": qid, "answer": final_answer, "latency_ms": latency_ms, "tokens": 0})
    except Exception:
        pass

    # 9. Log eval (fire-and-forget)
    try:
        enqueue_event({"kind": "eval", "qid": qid, "groundedness": grounding, "hallucination_score": hal_score, "blocked": blocked})
    except Exception:
        pass

    logger.info(
        f"QUERY: \"{req.question}\" | "
        f"id={qid} | "
        f"grounding={grounding:.4f} | "
        f"hallucination={hal_score:.4f} | "
        f"blocked={blocked} | "
        f"latency={latency_ms:.0f}ms | "
        f"prompt={prompt_mgr.active_name}"
    )

    return ChatResponse(answer=final_answer, query_id=qid)


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    logger.info(f"FEEDBACK: id={req.query_id} | rating={req.rating}")
    return {"status": "ok"}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)

    saved = []
    errors = []
    total_chunks = 0

    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            errors.append(f"{f.filename or 'unnamed'}: not a PDF")
            continue
        # Sanitise filename — strip path separators
        safe_name = os.path.basename(f.filename)
        dest = os.path.join(DATA_DIR, safe_name)
        try:
            content = await f.read()
            if not content:
                errors.append(f"{safe_name}: empty file")
                continue

            with open(dest, "wb") as out:
                out.write(content)

            loader = PyPDFLoader(dest)
            docs = loader.load()
            if not docs:
                errors.append(f"{safe_name}: no pages extracted from PDF")
                continue

            # Don't rebuild chain per-file; do it once at the end
            chunks_added = add_documents_to_store(docs, rebuild_chain=False)
            total_chunks += chunks_added
            saved.append(safe_name)
            logger.info(f"Embedded {safe_name} ({chunks_added} chunks)")
        except Exception as exc:
            errors.append(f"{safe_name}: {exc}")
            logger.warning(f"Failed to process {safe_name}: {exc}")
        finally:
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except OSError:
                    pass

    if not saved:
        raise HTTPException(
            status_code=400,
            detail=f"No valid PDF files processed. Errors: {errors}"
        )

    # Rebuild chain once after all files are embedded
    with _store_lock:
        if vector_store is not None:
            rag_chain = build_chain(vector_store)

    logger.info(f"Uploaded {len(saved)} file(s), {total_chunks} chunks added.")
    return {
        "message": f"Uploaded {len(saved)} file(s), {total_chunks} chunks added to index.",
        "files": saved,
        "errors": errors,
        "total_chunks": total_chunks,
    }


@app.get("/documents")
async def list_documents():
    """List PDF files currently present in DATA_DIR (before deletion)."""
    pdfs = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    return {"documents": [os.path.basename(p) for p in pdfs]}


@app.get("/documents-stats")
async def documents_stats():
    """Return ChromaDB collection stats (actual ingested documents/chunks)."""
    global vector_store
    if vector_store is None:
        try:
            emb = get_embedding()
            vector_store = Chroma(
                persist_directory=CHROMA_DIR, embedding_function=emb
            )
        except Exception as exc:
            return {"error": f"Failed to load Chroma store: {exc}"}
    try:
        count = vector_store._collection.count()
        # Sample a few documents to show sources
        sample_meta = []
        if count > 0:
            sample = vector_store._collection.peek(limit=5)
            for doc in sample.get("metadatas", []):
                src = doc.get("source", "unknown")
                if src not in sample_meta:
                    sample_meta.append(src)
        return {
            "total_chunks": count,
            "sample_sources": sample_meta,
            "status": "ok",
        }
    except Exception as exc:
        return {"error": f"Failed to query Chroma: {exc}"}


class IngestUrlRequest(BaseModel):
    urls: list[str]


# ── Curated site lists for bulk ingestion ──────────────────────────────
_CATEGORIZED_SITES = {
    "indian_laws": [
        "https://legislative.gov.in/indiancode",
        "https://legislative.gov.in/acts-of-parliament-from-the-year",
        "https://indiankanoon.org/",
        "https://sci.gov.in/judgements",
        "https://delhihighcourt.nic.in/judgements",
        "https://bombayhighcourt.nic.in/judgements",
        "https://mhc.tn.gov.in/judgements",
        "https://hphighcourt.nic.in/judgements",
        "https://mphc.nic.in/judgements",
        "https://rhcourt.rajasthan.gov.in/judgements",
    ],
    "us_laws": [
        "https://www.law.cornell.edu/uscode/text",
        "https://www.congress.gov/bills-with-text",
        "https://supreme.justia.com/cases/federal/us/",
        "https://courtlistener.com/opinions/",
        "https://law.justia.com/cases/federal/appellate-courts/",
        "https://www.govinfo.gov/app/collection/uscode",
    ],
    "uk_laws": [
        "https://www.legislation.gov.uk/ukpga",
        "https://www.supremecourt.uk/decided-cases/",
        "https://www.judiciary.uk/judgments/",
        "https://www.bailii.org/uk/cases/UKSC/",
        "https://www.bailii.org/uk/cases/UKPC/",
    ],
    "international": [
        "https://www.un.org/en/about-us/un-charter/full-text",
        "https://www.echr.coe.int/Pages/home.aspx?p=case-law",
        "https://www.ohchr.org/en/professionalinterest/pages/ccpr.aspx",
        "https://www.icj-cij.org/en/cases",
        "https://www.wto.org/english/tratop_e/dispu_e/cases_e.htm",
    ],
    "regulatory": [
        "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes",
        "https://www.rbi.org.in/scripts/BS_ViewBS.aspx",
        "https://www.irs.gov/forms-pubs",
        "https://www.federalregister.gov/",
        "https://www.europarl.europa.eu/activities/plenary.do",
    ],
}


def _get_site_urls(category: str) -> list[str]:
    """Return list of URLs for a category, raise if unknown."""
    urls = _CATEGORIZED_SITES.get(category.lower())
    if not urls:
        available = ", ".join(sorted(_CATEGORIZED_SITES.keys()))
        raise ValueError(f"Unknown category '{category}'. Available: {available}")
    return urls


@app.post("/ingest-url")
async def ingest_url(req: IngestUrlRequest):
    """Scrape one or more URLs and add their content to the vector store."""
    if not req.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")

    ingested = []
    errors = []
    total_chunks = 0

    # Validate all URLs upfront before fetching anything
    for url in req.urls:
        try:
            _validate_url(url)
        except ValueError as exc:
            errors.append(f"{url}: {exc}")

    valid_urls = [u for u in req.urls if not any(u in e for e in errors)]

    for url in valid_urls:
        try:
            docs = scrape_url(url)
            if not docs:
                errors.append(f"{url}: no text content found")
                continue
            # Don't rebuild chain per-URL; do it once at the end
            chunks_added = add_documents_to_store(docs, rebuild_chain=False)
            if chunks_added == 0:
                errors.append(f"{url}: duplicate content, already in index")
                continue
            total_chunks += chunks_added
            ingested.append(url)
            logger.info(f"Ingested URL: {url} ({chunks_added} chunks)")
        except requests.exceptions.Timeout:
            errors.append(f"{url}: request timed out")
            logger.warning(f"Timeout fetching {url}")
        except requests.exceptions.ConnectionError:
            errors.append(f"{url}: connection error")
            logger.warning(f"Connection error fetching {url}")
        except requests.exceptions.HTTPError as exc:
            errors.append(f"{url}: HTTP {exc.response.status_code}")
            logger.warning(f"HTTP error fetching {url}: {exc}")
        except requests.exceptions.RequestException as exc:
            errors.append(f"{url}: request failed — {exc}")
            logger.warning(f"Request failed {url}: {exc}")
        except ValueError as exc:
            errors.append(f"{url}: {exc}")
            logger.warning(f"Validation error for {url}: {exc}")
        except Exception as exc:
            errors.append(f"{url}: unexpected error — {exc}")
            logger.warning(f"Failed to ingest {url}: {exc}")

    # Rebuild chain once after all URLs are embedded
    if ingested:
        with _store_lock:
            if vector_store is not None:
                rag_chain = build_chain(vector_store)

    if not ingested:
        raise HTTPException(
            status_code=400,
            detail=f"No URLs ingested. Errors: {errors}"
        )

    return {
        "message": f"Ingested {len(ingested)} URL(s), {total_chunks} chunks added.",
        "ingested": ingested,
        "errors": errors,
        "total_chunks": total_chunks,
    }


class IngestCategoryRequest(BaseModel):
    category: str
    limit: int | None = None   # optional: only ingest first N URLs


@app.post("/ingest-category")
async def ingest_category(req: IngestCategoryRequest):
    """Scrape a predefined category of law/government sites and add their content to the vector store."""
    try:
        urls = _get_site_urls(req.category)
        if req.limit:
            urls = urls[:req.limit]
        if not urls:
            raise HTTPException(status_code=400, detail="No URLs found for this category.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Reuse the same ingestion logic as /ingest-url
    fake_req = IngestUrlRequest(urls=urls)
    return await ingest_url(fake_req)


@app.get("/ingest-categories")
async def list_categories():
    """Return all available categories for bulk ingestion."""
    return {
        "categories": sorted(_CATEGORIZED_SITES.keys()),
        "description": {
            "indian_laws": "Indian statutes, Supreme Court, and High Court judgments",
            "us_laws": "US Code, Congressional bills, Supreme Court & appellate opinions",
            "uk_laws": "UK legislation, Supreme Court & Privy Council cases",
            "international": "UN Charter, ECHR case law, ICC, WTO disputes",
            "regulatory": "SEBI, RBI, IRS, Federal Register, EU Parliament",
        },
    }


@app.get("/analytics")
async def analytics():
    from observability.metrics import get_hallucination_rate, get_avg_latency, get_retrieval_failure_rate
    return {
        "hallucination": get_hallucination_rate(last_n=100),
        "latency": get_avg_latency(last_n=100),
        "retrieval": get_retrieval_failure_rate(last_n=100),
    }


