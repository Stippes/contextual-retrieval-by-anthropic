"""Utilities for chunking structured document elements.

This module provides a :func:`chunk_elements` function which groups
structured elements extracted from documents into text chunks while
preserving relevant metadata. It aims to create semantically meaningful
chunks for downstream indexing and retrieval.

Features
--------
* Titles are merged with their following narrative blocks.
* Consecutive list items are aggregated under the nearest preceding heading.
* Table headers are preserved and large tables are split by row groups.
* One chunk is created per PPTX slide and slide notes are emitted as
  separate chunks.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, MutableMapping

import tiktoken

__all__ = ["chunk_elements"]


def _get_type(el: Any) -> str:
    if isinstance(el, MutableMapping):
        return str(el.get("type") or el.get("element_type") or el.get("category") or "")
    return str(getattr(el, "type", getattr(el, "category", "")) or "")


def _get_text(el: Any) -> str:
    if isinstance(el, MutableMapping):
        return str(el.get("text") or "")
    return str(getattr(el, "text", "") or "")


def _get_metadata(el: Any) -> Dict[str, Any]:
    if isinstance(el, MutableMapping):
        return dict(el.get("metadata") or {})
    return dict(getattr(el, "metadata", {}) or {})


def chunk_elements(
    elements: Iterable[Any],
    target_tokens: int = 500,
    max_tokens: int = 1200,
) -> List[Dict[str, Any]]:
    """Chunk document ``elements`` into text blocks with metadata.

    Parameters
    ----------
    elements:
        Iterable of objects representing document elements. Each element
        should expose ``text`` and ``metadata`` attributes (or dictionary
        keys) and optionally a ``type``/``category`` describing the
        element kind (e.g. ``"Title"``, ``"NarrativeText"``, ``"ListItem"``,
        ``"Table"``, ``"Slide"``, ``"SlideNote"``).
    target_tokens:
        Desired size of each chunk. Buffers exceeding this size are
        flushed. Defaults to ``500``.
    max_tokens:
        Hard limit for chunk size. Very long tables or slides are split
        so that no chunk exceeds this limit. Defaults to ``1200``.

    Returns
    -------
    list of dict
        A list where each item contains ``{"text": str, "metadata": dict}``.
    """

    try:
        encoding = tiktoken.get_encoding("cl100k_base")

        def count_tokens(text: str) -> int:
            return len(encoding.encode(text))

    except Exception:
        # Fallback simple tokenizer when encoding files cannot be downloaded
        def count_tokens(text: str) -> int:  # type: ignore[no-redef]
            return len(text.split())

    # First separate PPTX slide content and notes from other elements
    slides: Dict[Any, List[Any]] = {}
    notes: Dict[Any, List[Any]] = {}
    others: List[Any] = []
    for el in elements:
        md = _get_metadata(el)
        typ = _get_type(el).lower()
        slide_id = md.get("slide_number") or md.get("slide_id") or md.get("slide")
        if slide_id is not None:
            if "note" in typ:
                notes.setdefault(slide_id, []).append(el)
            else:
                slides.setdefault(slide_id, []).append(el)
        else:
            others.append(el)

    chunks: List[Dict[str, Any]] = []

    heading_types = {"title", "heading", "header"}
    list_types = {"list_item", "unordered_list_item", "ordered_list_item"}

    current_heading: str | None = None
    buffer: List[str] = []
    buffer_tokens = 0
    current_md: Dict[str, Any] = {}
    list_buffer: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer, buffer_tokens, current_md
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        md = dict(current_md)
        if current_heading:
            md.setdefault("section_title", current_heading)
        chunks.append({"text": text, "metadata": md})
        buffer = []
        buffer_tokens = 0
        current_md = {}

    def flush_list(md: Dict[str, Any]) -> None:
        nonlocal list_buffer
        if not list_buffer:
            return
        list_text = "\n".join(f"- {li}" for li in list_buffer)
        list_buffer = []
        add_text(list_text, md)

    def add_text(text: str, md: Dict[str, Any]) -> None:
        nonlocal buffer_tokens, buffer, current_md
        tks = count_tokens(text)
        if buffer and buffer_tokens + tks > target_tokens:
            flush_buffer()
        if not buffer:
            current_md = md.copy()
        buffer.append(text)
        buffer_tokens += tks
        if buffer_tokens >= target_tokens:
            flush_buffer()

    def handle_table(el: Any, md: Dict[str, Any]) -> None:
        nonlocal current_heading
        flush_list(md)
        flush_buffer()
        # Expect table text with newline-separated rows
        rows = [r for r in _get_text(el).splitlines() if r.strip()]
        if not rows:
            return
        header = rows[0]
        data_rows = rows[1:]
        group: List[str] = []
        tokens = count_tokens(header)
        for row in data_rows:
            row_tokens = count_tokens(row)
            if group and tokens + row_tokens > target_tokens:
                chunk_text = "\n".join([header] + group)
                table_md = md.copy()
                if current_heading:
                    table_md.setdefault("section_title", current_heading)
                chunks.append({"text": chunk_text, "metadata": table_md})
                group = [row]
                tokens = count_tokens(header) + row_tokens
            else:
                group.append(row)
                tokens += row_tokens
            if tokens >= target_tokens:
                chunk_text = "\n".join([header] + group)
                table_md = md.copy()
                if current_heading:
                    table_md.setdefault("section_title", current_heading)
                chunks.append({"text": chunk_text, "metadata": table_md})
                group = []
                tokens = count_tokens(header)
        if group:
            chunk_text = "\n".join([header] + group)
            table_md = md.copy()
            if current_heading:
                table_md.setdefault("section_title", current_heading)
            chunks.append({"text": chunk_text, "metadata": table_md})

    # Process non-slide elements respecting structure
    for el in others:
        typ = _get_type(el).lower()
        text = _get_text(el).strip()
        md = _get_metadata(el)

        if typ in heading_types:
            flush_list(md)
            flush_buffer()
            current_heading = text
            current_md = md.copy()
            buffer.append(text)
            buffer_tokens = count_tokens(text)
            continue

        if typ in list_types:
            list_buffer.append(text)
            current_md = md.copy()
            continue

        if typ == "table":
            handle_table(el, md)
            continue

        flush_list(md)
        add_text(text, md)

    flush_list(current_md)
    flush_buffer()

    # Finally handle slide content and notes
    def _process_slide(slide_id: Any, elems: List[Any], note: bool = False) -> None:
        text = "\n".join(_get_text(e).strip() for e in elems if _get_text(e).strip())
        if not text:
            return
        md: Dict[str, Any] = {"slide_id": slide_id}
        if note:
            md["section_title"] = "slide_note"
        tokens = count_tokens(text)
        if tokens <= max_tokens:
            chunks.append({"text": text, "metadata": md})
            return
        # Split oversized slide by lines
        lines = text.splitlines()
        group: List[str] = []
        tks = 0
        for line in lines:
            line_tokens = count_tokens(line)
            if group and tks + line_tokens > target_tokens:
                chunks.append({"text": "\n".join(group).strip(), "metadata": md})
                group = [line]
                tks = line_tokens
            else:
                group.append(line)
                tks += line_tokens
        if group:
            chunks.append({"text": "\n".join(group).strip(), "metadata": md})

    for slide_id, elems in sorted(slides.items()):
        _process_slide(slide_id, elems, note=False)
    for slide_id, elems in sorted(notes.items()):
        _process_slide(slide_id, elems, note=True)

    return chunks
