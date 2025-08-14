import os
import sys
sys.path.insert(0, os.path.abspath("."))

from src.extractors import unstructured_extractor
from src.extractors.unstructured_extractor import extract_unstructured


def test_extract_pdf(sample_pdf, monkeypatch):
    import fitz
    from unstructured.documents.elements import Text

    def _simple_partition_pdf(filename: str, **_: str):
        doc = fitz.open(filename)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return [Text(text)]

    monkeypatch.setattr(unstructured_extractor, "partition_pdf", _simple_partition_pdf)
    monkeypatch.setitem(
        unstructured_extractor._PARTITIONERS,
        "application/pdf",
        _simple_partition_pdf,
    )
    elements = extract_unstructured(str(sample_pdf), "application/pdf")
    assert any("Hello PDF" in el.text for el in elements)


def test_extract_docx(sample_docx):
    elements = extract_unstructured(
        str(sample_docx),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert any("Hello DOCX" in el.text for el in elements)


def test_extract_pptx(sample_pptx):
    elements = extract_unstructured(
        str(sample_pptx),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    assert any("Hello PPTX" in el.text for el in elements)

