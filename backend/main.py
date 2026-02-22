import os
import glob
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# ── Config ───────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

logger = logging.getLogger("uvicorn.error")
prompt_mgr = PromptManager()

# ── Globals ──────────────────────────────────────────────────────────
rag_chain = None
vector_store = None
retriever = None


# ── Helper functions ─────────────────────────────────────────────────
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def load_pdfs(data_dir: str):
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        return []
    docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    return docs


def build_vector_store(docs):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=768, chunk_overlap=128, length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    embedding = get_embedding()
    vs = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=CHROMA_DIR
    )
    return vs


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
    yield
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

    # Retrieve context
    context_docs = retriever.invoke(req.question)
    context_text = format_docs(context_docs)

    # Run chain
    answer = rag_chain.invoke(req.question)
    latency_ms = (time.time() - t0) * 1000

    # Compute scores
    grounding = compute_grounding_score(answer, context_text)
    hal_score = detect_hallucination(answer, context_text)

    # Hallucination guard
    guard_result = hallucination_guard(answer, context_text)
    final_answer = guard_result["answer"]
    blocked = guard_result["blocked"]

    # Print to uvicorn terminal
    query_id = uuid.uuid4().hex[:12]
    logger.info(
        f"QUERY: \"{req.question}\" | "
        f"id={query_id} | "
        f"grounding={grounding:.4f} | "
        f"hallucination={hal_score:.4f} | "
        f"blocked={blocked} | "
        f"latency={latency_ms:.0f}ms | "
        f"prompt={prompt_mgr.active_name}"
    )

    return ChatResponse(answer=final_answer, query_id=query_id)


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    logger.info(f"FEEDBACK: id={req.query_id} | rating={req.rating}")
    return {"status": "ok"}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    global rag_chain, vector_store
    os.makedirs(DATA_DIR, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename.endswith(".pdf"):
            continue
        dest = os.path.join(DATA_DIR, f.filename)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved.append(f.filename)

    if not saved:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded.")

    docs = load_pdfs(DATA_DIR)
    vector_store = build_vector_store(docs)
    rag_chain = build_chain(vector_store)
    logger.info(f"Uploaded {len(saved)} file(s). Index rebuilt.")

    return {"message": f"Uploaded {len(saved)} file(s). Index rebuilt.", "files": saved}


@app.get("/documents")
async def list_documents():
    pdfs = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    return {"documents": [os.path.basename(p) for p in pdfs]}


