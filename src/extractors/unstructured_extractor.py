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
    partitioner: Callable[..., list[Element]]
    if mime == "application/pdf":
        from unstructured.partition.pdf import partition_pdf

        partitioner = partition_pdf
    elif mime == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        from unstructured.partition.pptx import partition_pptx

        partitioner = partition_pptx
    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from unstructured.partition.docx import partition_docx

        partitioner = partition_docx
    elif mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        from unstructured.partition.xlsx import partition_xlsx

        partitioner = partition_xlsx
    else:
        raise ValueError(f"Unsupported MIME type: {mime}")
    return partitioner(filename=path)
