"""
Structure-aware PDF loader for the RAG knowledge base.

Loads the FloodSense-PK architecture PDF and splits it into hierarchical
sections based on its numbered headings (e.g. "1. Institutional and Regulatory
Framework", "3. Discharge Capacities ..."). Each top-level section becomes a
document so that downstream chunking keeps related content — including the
discharge/travel-time tables and the metadata JSON schema — grouped together.
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


def extract_pdf_text(path: str | Path = DEFAULT_PDF) -> str:
    """Extract and whitespace-normalise the full text of a PDF."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return normalize_whitespace("\n".join(pages))


def normalize_whitespace(text: str) -> str:
    """Collapse the per-token line breaks pypdf emits into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _derive_title(section_text: str, max_words: int = 9) -> str:
    """Build a short heading from the start of a section's text."""
    # Drop the leading "N. " marker, then take the first few words as a title.
    body = re.sub(r"^\d{1,2}\.\s*", "", section_text).strip()
    words = body.split()
    return " ".join(words[:max_words])


def split_into_sections(text: str) -> list[dict]:
    """
    Split normalised text into hierarchical sections by sequential numbered
    headings. Returns dicts with ``number``, ``title`` and ``content``.
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

    if not boundaries:
        # No recognisable headings — fall back to the whole document.
        return [{"number": 1, "title": _derive_title(text), "content": text}]

    sections: list[dict] = []
    for i, (start, number) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        content = text[start:end].strip()
        sections.append(
            {
                "number": number,
                "title": _derive_title(content),
                "content": content,
            }
        )
    return sections


def load_pdf_documents(path: str | Path = DEFAULT_PDF) -> list[dict]:
    """
    Load a PDF into ingestion-ready document dicts.

    Each returned dict matches the schema consumed by ``ingest_documents``:
    ``{"source", "title", "content"}``. One dict is produced per top-level
    numbered section so chunking respects the document's hierarchy.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    text = extract_pdf_text(path)
    sections = split_into_sections(text)
    source = path.name
    return [
        {
            "source": source,
            "title": f"{section['number']}. {section['title']}",
            "content": section["content"],
        }
        for section in sections
    ]
