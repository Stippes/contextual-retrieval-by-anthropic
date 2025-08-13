from __future__ import annotations

"""High level document ingestion utilities."""

import mimetypes

from unstructured.documents.elements import Element

from .tika_adapter import TikaAdapter
from .unstructured_extractor import extract_unstructured


def ingest_file(
    path: str,
    mime: str,
    prefer: str = "unstructured",
    tika: TikaAdapter | None = None,
) -> list[Element]:
    """Ingest a file using Unstructured and/or Tika."""
    tika = tika or TikaAdapter()
    extractors: list[str] = [prefer, "tika" if prefer == "unstructured" else "unstructured"]
    def _text_len(els: list[Element]) -> int:
        return sum(len(getattr(el, "text", "").strip()) for el in els)

    elements: list[Element] = []
    for extractor in extractors:
        try:
            if extractor == "unstructured":
                elements = extract_unstructured(path, mime)
            else:
                elements = tika.extract(path, mime)
                if (
                    mime == "application/pdf"
                    and _text_len(elements) < 10
                ):
                    elements = tika.extract(path, mime, ocr="ocr_and_text")
        except Exception:
            elements = []
        if elements:
            break
    return [el for el in elements if getattr(el, "text", "").strip()]


def load_documents(paths: list[str]) -> list[Element]:
    """Load and normalize documents from a list of file paths."""
    tika = TikaAdapter()
    all_elements: list[Element] = []
    for path in paths:
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            continue
        elements = ingest_file(path, mime, tika=tika)
        for el in elements:
            if hasattr(el, "text") and isinstance(el.text, str):
                el.text = el.text.strip()
            all_elements.append(el)
    return all_elements


__all__ = ["TikaAdapter", "extract_unstructured", "ingest_file", "load_documents"]
