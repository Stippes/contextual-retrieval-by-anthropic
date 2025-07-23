import os
import sys

sys.path.insert(0, os.path.abspath("."))

from llama_index.readers.file import PyMuPDFReader
import fitz  # PyMuPDF


def test_pymupdf_reader(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello pdf")
    doc.save(pdf_path)
    doc.close()

    reader = PyMuPDFReader()
    docs = reader.load_data(file_path=str(pdf_path))
    assert any("hello pdf" in d.text for d in docs)
