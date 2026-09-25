from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


@dataclass
class ChildChunk:
    text: str
    source: str
    child_index: int  # global index across the document's children
    parent_index: int  # index into the PARENT chunking (matches Chunk.chunk_index)


def chunk_document(
    text: str, source: str, chunk_size_tokens: int, chunk_overlap_tokens: int
) -> list[Chunk]:
    # ~4 chars/token is a rough English average; good enough for chunk sizing,
    # the splitter still recurses on headers/paragraphs/sentences first.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens * 4,
        chunk_overlap=chunk_overlap_tokens * 4,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    return [Chunk(text=p, source=source, chunk_index=i) for i, p in enumerate(pieces)]


def chunk_document_with_children(
    text: str,
    source: str,
    parent_size_tokens: int,
    parent_overlap_tokens: int,
    child_size_tokens: int,
    child_overlap_tokens: int,
) -> tuple[list[Chunk], list[ChildChunk]]:
    """Parent-Child chunking (ADR-003): parents use the SAME chunking as
    chunk_document() (so existing ground truth, built against parent
    boundaries, stays valid unchanged) — each parent is then independently
    sub-split into smaller children. Children are what gets embedded/matched
    at query time; parents are what gets returned to the LLM for synthesis."""
    parents = chunk_document(text, source, parent_size_tokens, parent_overlap_tokens)

    children = []
    child_idx = 0
    for parent in parents:
        for sub in chunk_document(parent.text, source, child_size_tokens, child_overlap_tokens):
            children.append(
                ChildChunk(
                    text=sub.text,
                    source=source,
                    child_index=child_idx,
                    parent_index=parent.chunk_index,
                )
            )
            child_idx += 1

    return parents, children
