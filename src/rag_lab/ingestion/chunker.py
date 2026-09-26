"""Splits documents into overlapping, word-based chunks for embedding."""

from rag_lab.models import Chunk, Document

DEFAULT_CHUNK_SIZE = 150
DEFAULT_OVERLAP = 30


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Split `text` into overlapping chunks of whole words.

    YOUR TURN: implement this.

    Rules:
    - Split `text` on whitespace into a list of words.
    - A chunk is `chunk_size` consecutive words, joined back with single spaces.
    - Consecutive chunks overlap by `overlap` words: chunk N's last `overlap`
      words are the same as chunk N+1's first `overlap` words. So each new
      chunk starts `chunk_size - overlap` words after the previous chunk started.
    - The final chunk may be shorter than `chunk_size` (whatever words are
      left) — still include it, don't drop it.
    - Empty or whitespace-only `text` returns an empty list.
    - Raise ValueError if `overlap >= chunk_size` (the sliding window would
      never advance, or would go backwards).

    Why word-based, not character-based: word count is a much closer proxy
    for token count than character count, and it never cuts a word in half
    at a chunk boundary.
    """

    text_list = text.split()
    chunks = []
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size.")
    if not text_list:
        return chunks
    for i in range(0, len(text_list), chunk_size - overlap):
        chunk = " ".join(text_list[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def chunk_document(document: Document, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    texts = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            id=f"{document.id}::{position}",
            document_id=document.id,
            source=document.source,
            text=text,
            position=position,
        )
        for position, text in enumerate(texts)
    ]
