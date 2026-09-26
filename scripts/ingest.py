"""Build the index: load sample docs, chunk them, embed the chunks, persist to disk.

Usage: python scripts/ingest.py [docs_folder]

Writes data/index/chunks.json (chunk metadata) and data/index/embeddings.npy
(row i = embedding for chunk i in the JSON list). This is a throwaway,
inspectable format for the CLI-first v1 — delete data/index/ and rerun this
script any time. It gets replaced by Qdrant once that phase arrives.
"""

import json
import sys
from pathlib import Path

import numpy as np

from rag_lab.embeddings.local import MiniLMEmbedder
from rag_lab.ingestion.chunker import chunk_document
from rag_lab.ingestion.loader import load_documents

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_FOLDER = REPO_ROOT / "data" / "sample_docs"
INDEX_FOLDER = REPO_ROOT / "data" / "index"


def main(docs_folder: Path) -> None:
    documents = load_documents(docs_folder)
    print(f"Loaded {len(documents)} document(s) from {docs_folder}")

    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    print(f"Split into {len(chunks)} chunk(s)")

    embedder = MiniLMEmbedder()
    embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
    print(f"Embedded chunks into shape {embeddings.shape}")

    INDEX_FOLDER.mkdir(parents=True, exist_ok=True)
    chunks_path = INDEX_FOLDER / "chunks.json"
    embeddings_path = INDEX_FOLDER / "embeddings.npy"

    chunks_path.write_text(json.dumps([chunk.model_dump() for chunk in chunks], indent=2))
    np.save(embeddings_path, embeddings)

    print(f"Wrote {chunks_path} and {embeddings_path}")


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCS_FOLDER
    main(folder)
