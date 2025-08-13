from __future__ import annotations

"""Utilities for extracting documents using the Unstructured library."""

from typing import Callable, Dict

from unstructured.documents.elements import Element
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx


_PARTITIONERS: Dict[str, Callable[..., list[Element]]] = {
    "application/pdf": partition_pdf,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": partition_pptx,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": partition_docx,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": partition_xlsx,
}


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
    partitioner = _PARTITIONERS.get(mime)
    if partitioner is None:
        raise ValueError(f"Unsupported MIME type: {mime}")
    return partitioner(filename=path)
