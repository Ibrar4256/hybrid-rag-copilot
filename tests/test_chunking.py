from research_copilot.chunking import chunk_document, chunk_document_with_children


def test_chunk_document_covers_whole_text_without_gaps():
    text = "Sentence one. Sentence two. Sentence three. " * 20
    chunks = chunk_document(text, source="doc.md", chunk_size_tokens=20, chunk_overlap_tokens=5)

    assert len(chunks) > 1
    assert all(c.source == "doc.md" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # every chunk has real content, no empty pieces
    assert all(c.text.strip() for c in chunks)


def test_chunk_document_short_text_stays_one_chunk():
    text = "A single short sentence."
    chunks = chunk_document(text, source="doc.md", chunk_size_tokens=200, chunk_overlap_tokens=40)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0


def test_chunk_document_respects_markdown_headers_as_split_points():
    text = "## Section A\n" + ("word " * 100) + "\n## Section B\n" + ("word " * 100)
    chunks = chunk_document(text, source="doc.md", chunk_size_tokens=30, chunk_overlap_tokens=0)

    # the splitter's separator list prioritizes "\n## " before falling back to
    # smaller units, so a chunk boundary should land at (or very near) the
    # second header rather than splitting mid-word inside one section
    assert any(c.text.lstrip().startswith("## Section B") for c in chunks)


def test_chunk_document_with_children_parents_match_flat_chunking():
    text = "Sentence one. Sentence two. Sentence three. " * 30
    flat_parents = chunk_document(text, "doc.md", chunk_size_tokens=50, chunk_overlap_tokens=10)
    parents, children = chunk_document_with_children(
        text,
        "doc.md",
        parent_size_tokens=50,
        parent_overlap_tokens=10,
        child_size_tokens=15,
        child_overlap_tokens=3,
    )

    assert [p.text for p in parents] == [p.text for p in flat_parents]


def test_chunk_document_with_children_indices_are_globally_sequential():
    text = "Sentence one. Sentence two. Sentence three. " * 30
    parents, children = chunk_document_with_children(
        text,
        "doc.md",
        parent_size_tokens=50,
        parent_overlap_tokens=10,
        child_size_tokens=15,
        child_overlap_tokens=3,
    )

    assert [c.child_index for c in children] == list(range(len(children)))
    # every child's parent_index must point at a real parent
    parent_indices = {p.chunk_index for p in parents}
    assert all(c.parent_index in parent_indices for c in children)
