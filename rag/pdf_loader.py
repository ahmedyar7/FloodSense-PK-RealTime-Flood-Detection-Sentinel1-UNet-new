"""
Structure-aware, page-tracking PDF loader for the RAG knowledge base.

Loads the FloodSense-PK architecture PDF and splits it into hierarchical
sections based on its numbered headings (e.g. "1. Institutional and Regulatory
Framework", "3. Discharge Capacities ..."). Each top-level section becomes a
document so that downstream chunking keeps related content — including the
discharge/travel-time tables and the metadata JSON schema — grouped together.

In addition to the section structure, every loaded document records the
``page_number`` on which its heading first appears so that retrieval can offer
clean source attribution back to the original PDF page.
"""

import re
from pathlib import Path

from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF = DATA_DIR / "Structured-Knowledge-Instruction-Pipeline.pdf"

# Top-level numbered headings look like "1. Title Words" with a capitalised word
# following the number. Sub-list items (the "1./2./3." inside section 7) and
# stray numbers like dates are filtered out by requiring sequential numbering.
_HEADING_RE = re.compile(r"(\d{1,2})\.\s+[A-Z]")


def normalize_whitespace(text: str) -> str:
    """Collapse the per-token line breaks pypdf emits into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_pages(path: str | Path = DEFAULT_PDF) -> list[dict]:
    """
    Extract per-page text from a PDF.

    Returns a list of ``{"page_number": int, "text": str}`` dicts (1-indexed
    page numbers) with each page's text whitespace-normalised. Blank pages are
    preserved so page numbering stays aligned with the source document.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    return [
        {"page_number": i, "text": normalize_whitespace(page.extract_text() or "")}
        for i, page in enumerate(reader.pages, start=1)
    ]


def _build_full_text(pages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """
    Concatenate normalised page texts into one string and return a map of the
    character offset at which each page begins. The offsets let us trace a
    section (located by its position in the full text) back to its source page.
    """
    parts: list[str] = []
    boundaries: list[tuple[int, int]] = []  # (start_offset, page_number)
    offset = 0
    for page in pages:
        text = page["text"]
        if not text:
            continue
        boundaries.append((offset, page["page_number"]))
        parts.append(text)
        offset += len(text) + 1  # +1 for the single-space join below
    return " ".join(parts), boundaries


def _page_for_offset(offset: int, boundaries: list[tuple[int, int]]) -> int:
    """Return the page number whose text span contains ``offset``."""
    page = boundaries[0][1] if boundaries else 1
    for start, number in boundaries:
        if offset >= start:
            page = number
        else:
            break
    return page


def extract_pdf_text(path: str | Path = DEFAULT_PDF) -> str:
    """Extract and whitespace-normalise the full text of a PDF."""
    pages = extract_pdf_pages(path)
    full_text, _ = _build_full_text(pages)
    return full_text


def _derive_title(section_text: str, max_words: int = 9) -> str:
    """Build a short heading from the start of a section's text."""
    # Drop the leading "N. " marker, then take the first few words as a title.
    body = re.sub(r"^\d{1,2}\.\s*", "", section_text).strip()
    words = body.split()
    return " ".join(words[:max_words])


def split_into_sections(
    text: str, page_boundaries: list[tuple[int, int]] | None = None
) -> list[dict]:
    """
    Split normalised text into hierarchical sections by sequential numbered
    headings. Returns dicts with ``number``, ``title``, ``content`` and the
    ``page_number`` on which the section heading begins.
    """
    # Keep only headings whose number continues the 1, 2, 3, ... sequence so
    # that sub-items and dates embedded in the body are not treated as sections.
    boundaries: list[tuple[int, int]] = []
    expected = 1
    for match in _HEADING_RE.finditer(text):
        number = int(match.group(1))
        if number == expected:
            boundaries.append((match.start(), number))
            expected += 1

    page_boundaries = page_boundaries or []

    if not boundaries:
        # No recognisable headings — fall back to the whole document.
        return [
            {
                "number": 1,
                "title": _derive_title(text),
                "content": text,
                "page_number": _page_for_offset(0, page_boundaries),
            }
        ]

    sections: list[dict] = []
    for i, (start, number) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        content = text[start:end].strip()
        sections.append(
            {
                "number": number,
                "title": _derive_title(content),
                "content": content,
                "page_number": _page_for_offset(start, page_boundaries),
            }
        )
    return sections


def load_pdf_documents(path: str | Path = DEFAULT_PDF) -> list[dict]:
    """
    Load a PDF into ingestion-ready document dicts.

    Each returned dict matches the schema consumed by ``ingest_documents``:
    ``{"source", "title", "content", "page_number"}``. One dict is produced per
    top-level numbered section so chunking respects the document's hierarchy,
    and ``page_number`` records where that section starts in the source PDF.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = extract_pdf_pages(path)
    full_text, page_boundaries = _build_full_text(pages)
    sections = split_into_sections(full_text, page_boundaries)
    source = path.name
    return [
        {
            "source": source,
            "title": f"{section['number']}. {section['title']}",
            "content": section["content"],
            "page_number": section["page_number"],
        }
        for section in sections
    ]
