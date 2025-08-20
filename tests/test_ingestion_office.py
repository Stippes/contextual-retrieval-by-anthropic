import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

from src.extractors import load_documents
from src.ingest.chunking import chunk_elements


def _decode_fixture(tmp_path: Path, name: str) -> str:
    b64_path = Path("tests") / "data" / f"{name}.b64"
    data = base64.b64decode(b64_path.read_text())
    out_path = tmp_path / name
    out_path.write_bytes(data)
    return str(out_path)


def test_ingestion_office(tmp_path):
    pptx = _decode_fixture(tmp_path, "sample.pptx")
    xlsx = _decode_fixture(tmp_path, "sample.xlsx")

    elements = load_documents([pptx, xlsx])
    chunks = chunk_elements(elements, target_tokens=5, max_tokens=10)

    slide_chunks = [
        c for c in chunks if c["metadata"].get("slide_id") == 1 and c["metadata"].get("section_title") != "slide_note"
    ]
    assert any("Slide body" in c["text"] for c in slide_chunks)

    note_chunks = [c for c in chunks if c["metadata"].get("section_title") == "slide_note"]
    assert note_chunks and "Note body" in note_chunks[0]["text"]

    sheet1 = [c for c in chunks if c["metadata"].get("sheet") == "Sheet1"]
    assert sheet1 and "H1A" in sheet1[0]["text"]

    sheet2 = [c for c in chunks if c["metadata"].get("sheet") == "Sheet2"]
    assert sheet2 and "H2A" in sheet2[0]["text"]
