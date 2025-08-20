from __future__ import annotations

"""Utilities for extracting documents using the Unstructured library."""

from typing import Callable

from unstructured.documents.elements import Element


def extract_unstructured(path: str, mime: str) -> list[Element]:
    """Extract elements from ``path`` using Unstructured.

    Parameters
    ----------
    path: str
        The path to the file to extract.
    mime: str
        The MIME type for the file.

    Returns
    -------
    list[Element]
        The extracted elements.
    """
    if mime == "application/pdf":
        from unstructured.partition.pdf import partition_pdf

        return partition_pdf(filename=path)
    if mime == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        from unstructured.partition.pptx import partition_pptx
        from unstructured.documents.elements import NarrativeText

        elements = partition_pptx(filename=path, include_slide_notes=True)
        for el in elements:
            md = getattr(el, "metadata", None)
            slide_no = getattr(md, "page_number", None)
            if md is not None and slide_no is not None:
                try:
                    md.slide_number = slide_no  # type: ignore[attr-defined]
                except Exception:
                    setattr(md, "slide_number", slide_no)
            if isinstance(el, NarrativeText):
                el.category = "SlideNote"
        return elements
    if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from unstructured.partition.docx import partition_docx

        return partition_docx(filename=path)
    if mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        from unstructured.partition.xlsx import partition_xlsx
        from bs4 import BeautifulSoup

        elements = partition_xlsx(filename=path, include_header=True)
        for el in elements:
            md = getattr(el, "metadata", None)
            html = getattr(md, "text_as_html", None)
            if html:
                soup = BeautifulSoup(html, "html.parser")
                rows: list[str] = []
                for tr in soup.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    if cells:
                        rows.append(" ".join(cells))
                el.text = "\n".join(rows)
            sheet_name = getattr(md, "page_name", None)
            if md is not None and sheet_name:
                try:
                    md.sheet = sheet_name  # type: ignore[attr-defined]
                except Exception:
                    setattr(md, "sheet", sheet_name)
        return elements
    raise ValueError(f"Unsupported MIME type: {mime}")
