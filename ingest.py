import sys
import pickle
import subprocess
from pathlib import Path

from markitdown import MarkItDown
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "nomic-embed-text"


def convert_pdf_to_markdown(pdf_path: Path, md_path: Path):
    print("\nConverting PDF to Markdown...")

    try:
        md = MarkItDown()
        result = md.convert(str(pdf_path))
        md_path.write_text(result.text_content, encoding="utf-8")
    except Exception:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "markitdown",
                str(pdf_path),
                "-o",
                str(md_path),
            ],
            check=True,
        )

    print("Markdown created.")


def load_markdown(md_path: Path):
    print("\nLoading markdown...")

    text = md_path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_text(text)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def build_bm25(chunks, store_dir):
    print("\nBuilding BM25 index...")

    tokenized = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    with open(store_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(store_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("BM25 saved.")


def build_chroma(chunks, store_dir):
    print("\nGenerating embeddings with Ollama...")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    chroma_dir = store_dir / "chroma"

    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
        collection_name="documents",
    )

    print("Vector store saved.")

def run_pipeline(pdf_path: Path, vectorstore_name: str = None):
    """
    Create a hybrid vector store from a PDF.
    """

    print("\n--------------------------------")
    print("Starting Ingestion Pipeline")
    print("--------------------------------")

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} does not exist.")

    # Use PDF filename if no store name is provided
    if not vectorstore_name:
        vectorstore_name = pdf_path.stem

    store_dir = Path("vector_store") / vectorstore_name
    store_dir.mkdir(parents=True, exist_ok=True)

    print(f"PDF          : {pdf_path.name}")
    print(f"Vector Store : {vectorstore_name}")

    # Markdown path
    md_path = store_dir / f"{vectorstore_name}.md"

    print("\n[1/4] Converting PDF to Markdown...")
    convert_pdf_to_markdown(pdf_path, md_path)

    print("[2/4] Loading Markdown...")
    chunks = load_markdown(md_path)

    print(f"Loaded {len(chunks)} chunks.")

    print("[3/4] Building BM25...")
    build_bm25(chunks, store_dir)

    print("[4/4] Building Chroma...")
    build_chroma(chunks, store_dir)

    print("\n--------------------------------")
    print("Hybrid Vector Store Created Successfully")
    print(f"Location : {store_dir}")
    print("--------------------------------\n")

    return store_dir


def main():
    pdf = input("Enter PDF path: ").strip()

    pdf_path = Path(pdf)

    if not pdf_path.exists():
        print("PDF not found.")
        return

    store_dir = Path("vector_store") / pdf_path.stem
    store_dir.mkdir(parents=True, exist_ok=True)

    md_path = store_dir / f"{pdf_path.stem}.md"

    convert_pdf_to_markdown(pdf_path, md_path)

    chunks = load_markdown(md_path)

    build_bm25(chunks, store_dir)

    build_chroma(chunks, store_dir)

    print("\n--------------------------------")
    print("Hybrid Vector Store Created")
    print(f"Location : {store_dir}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
