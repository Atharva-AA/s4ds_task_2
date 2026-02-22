import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context.
You need to answer the questions based on the context recieved and try to make correlation with the cases. If there is someone asking something and if there is some context with you in the given data then only you need to answer else

If you cannot find the answer in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Your tone should be explaining.
Answer:"""
)


def load_pdfs(data_dir):
    """Load all PDFs from the data directory."""
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        exit(1)

    docs = []
    for pdf_path in pdf_files:
        print(f"  Loading: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())

    print(f"  Loaded {len(docs)} pages from {len(pdf_files)} PDF(s)\n")
    return docs


def build_vector_store(docs):
    """Split documents into chunks and create a Chroma vector store."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=768,
        chunk_overlap=128,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Vector store saved to {CHROMA_DIR}\n")
    return vector_store


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vector_store):
    """Create a RAG chain using LCEL with Groq LLM."""
    llm = ChatGroq(
        temperature=0.5,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    print("=" * 50)
    print("  RAG ChatBot  -  Groq llama-3.3-70b-versatile")
    print("=" * 50)

    print("\n[1/3] Loading PDFs from data/ ...")
    docs = load_pdfs(DATA_DIR)

    print("[2/3] Building vector store ...")
    vector_store = build_vector_store(docs)

    print("[3/3] Ready! Ask questions about your documents.")
    print('Type "exit" to quit.\n')

    chain = create_rag_chain(vector_store)

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            print("Goodbye!")
            break

        answer = chain.invoke(query)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
