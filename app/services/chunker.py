import re
from dataclasses import dataclass

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class ChunkData:
    chunk_index: int
    section_header: str | None
    content: str


def _split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_header: str | None = None
    buffer: list[str] = []
    in_code_block = False

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            buffer.append(line)
            continue

        if not in_code_block:
            match = _HEADER_RE.match(line)
            if match:
                section_body = "\n".join(buffer).strip()
                if section_body:
                    sections.append((current_header, section_body))
                current_header = match.group(2)[:255]
                buffer = []
                continue

        buffer.append(line)

    tail = "\n".join(buffer).strip()
    if tail:
        sections.append((current_header, tail))

    if not sections and text.strip():
        sections.append((None, text.strip()))

    return sections


def chunk_markdown(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[ChunkData]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_size <= chunk_overlap:
        raise ValueError("chunk size must be positive and greater than chunk_overlap")
    
    step = chunk_size - chunk_overlap
    chunks: list[ChunkData] = []

    for header, section_text in _split_markdown_sections(text):
        start = 0
        while start < len(section_text):
            piece = section_text[start:start+chunk_size].strip()
            if piece:
                content = f"[Section: {header}]\n{piece}" if header else piece
                chunks.append(ChunkData(chunk_index=len(chunks), section_header=header, content=content))
            if start + chunk_size >= len(section_text):
                break
            start += step
    return chunks

