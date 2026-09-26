"""Loads .txt and .md files from a folder into Document objects."""

from pathlib import Path

from rag_lab.models import Document

SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_documents(folder: str | Path) -> list[Document]:
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir() if p.suffix in SUPPORTED_SUFFIXES)
    return [
        Document(id=path.stem, source=str(path), text=path.read_text(encoding="utf-8"))
        for path in paths
    ]
