import pytest

from app.services.chunker import chunk_markdown


def test_chunk_markdown_preserves_headers_and_overlap() -> None:
    first_section_body = "A" * 50 + "OVERLAP_MARKER" + "B" * 40
    markdown = (
        f"## QueuePool Exhaustion\n{first_section_body}\n\n"
        "### Mitigation\nIncrease pool_size in settings."
    )

    chunks = chunk_markdown(markdown, chunk_size=70, chunk_overlap=20)

    assert len(chunks) >= 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    assert chunks[0].section_header == "QueuePool Exhaustion"
    assert chunks[0].content.startswith("[Section: QueuePool Exhaustion]\n")
    assert chunks[1].section_header == "QueuePool Exhaustion"
    assert "OVERLAP_MARKER" in chunks[0].content
    assert "OVERLAP_MARKER" in chunks[1].content

    assert chunks[-1].section_header == "Mitigation"
    assert chunks[-1].content == "[Section: Mitigation]\nIncrease pool_size in settings."


def test_chunk_markdown_ignores_hashes_inside_code_blocks() -> None:
    markdown = (
        "## Diagnostics\n"
        "Run the command below:\n"
        "```bash\n"
        "# This is a bash comment, not a heading\n"
        "docker compose ps\n"
        "```\n"
    )

    chunks = chunk_markdown(markdown, chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].section_header == "Diagnostics"
    assert "# This is a bash comment, not a heading" in chunks[0].content


def test_chunk_markdown_rejects_invalid_window_params() -> None:
    with pytest.raises(ValueError):
        chunk_markdown("sample text", chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        chunk_markdown("sample text", chunk_size=100, chunk_overlap=150)

    with pytest.raises(ValueError):
        chunk_markdown("sample text", chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError):
        chunk_markdown("sample text", chunk_size=100, chunk_overlap=-10)

