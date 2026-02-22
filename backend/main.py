import os
import glob
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── Config ───────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context.
You need to answer the questions based on the context received and try to make correlation with the cases.
If you cannot find the answer in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Your tone should be explaining.
Answer:"""
)

# ── Globals ──────────────────────────────────────────────────────────
rag_chain = None
vector_store = None


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
    llm = ChatGroq(
        temperature=0.5,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
    )
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, vector_store
    print("[startup] Loading PDFs from data/ ...")
    docs = load_pdfs(DATA_DIR)
    if docs:
        print(f"[startup] {len(docs)} pages loaded. Building vector store ...")
        vector_store = build_vector_store(docs)
        rag_chain = build_chain(vector_store)
        print("[startup] RAG chain ready.")
    else:
        print("[startup] No PDFs found – upload via /upload first.")
    yield
    print("[shutdown] Bye!")


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


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "chain_ready": rag_chain is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialised. Upload PDFs first.")
    answer = rag_chain.invoke(req.question)
    return ChatResponse(answer=answer)


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

    # Rebuild index with ALL pdfs in data/
    docs = load_pdfs(DATA_DIR)
    vector_store = build_vector_store(docs)
    rag_chain = build_chain(vector_store)

    return {"message": f"Uploaded {len(saved)} file(s). Index rebuilt.", "files": saved}


@app.get("/documents")
async def list_documents():
    pdfs = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    return {"documents": [os.path.basename(p) for p in pdfs]}
