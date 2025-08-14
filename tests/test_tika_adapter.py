import importlib.util
import os
from pathlib import Path
from unittest.mock import Mock, patch, call

import pytest
from unstructured.documents.elements import Element, Text

# Dynamically load the TikaAdapter without importing the heavy package
# initialization in ``src.extractors``.
TIKA_ADAPTER_PATH = Path("src/extractors/tika_adapter.py")
spec = importlib.util.spec_from_file_location("tika_adapter", TIKA_ADAPTER_PATH)
_tika_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_tika_mod)  # type: ignore[misc]
TikaAdapter = _tika_mod.TikaAdapter


def _fake_response(text: str) -> Mock:
    resp = Mock()
    resp.json.return_value = [{"content": text}]
    resp.raise_for_status.return_value = None
    return resp


def load_ingest_file() -> callable:
    """Load ``ingest_file`` from ``src/extractors/__init__.py`` without importing
    the package and its heavy dependencies."""
    source = Path("src/extractors/__init__.py").read_text()
    start = source.index("def ingest_file")
    end = source.index("def load_documents")
    ingest_source = source[start:end]
    namespace = {
        "TikaAdapter": TikaAdapter,
        "extract_unstructured": lambda *a, **k: [],
        "Element": Element,
    }
    exec(ingest_source, namespace)
    return namespace["ingest_file"]


def test_tika_adapter_uses_env_and_ocr(tmp_path, monkeypatch):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello")

    monkeypatch.setenv("TIKA_URL", "http://tika")
    monkeypatch.setenv("TIKA_TIMEOUT", "10")
    monkeypatch.setenv("X_TIKA_WRITELIMIT", "5")
    monkeypatch.setenv("X_TIKA_MAX_EMBEDDED_RESOURCES", "7")

    adapter = TikaAdapter()

    fake_response = Mock()
    fake_response.json.return_value = [{"content": "text"}]
    fake_response.raise_for_status.return_value = None

    with patch("requests.put", return_value=fake_response) as mock_put:
        adapter.extract(str(file_path), "text/plain", ocr="ocr_only")

    assert mock_put.call_args.kwargs["headers"]["X-Tika-PDFOcrStrategy"] == "ocr_only"
    assert mock_put.call_args.kwargs["headers"]["X-Tika-WriteLimit"] == "5"
    assert mock_put.call_args.kwargs["headers"]["X-Tika-MaxEmbeddedResources"] == "7"
    assert mock_put.call_args.kwargs["timeout"] == 10
    assert mock_put.call_args.args[0] == "http://tika/rmeta/text"


def test_ingest_file_retries_with_ocr(tmp_path):
    ingest_file = load_ingest_file()
    pdf_path = tmp_path / "file.pdf"
    pdf_path.write_text("dummy")

    tika = Mock(spec=TikaAdapter)
    tika.extract.side_effect = [
        [Text("short")],
        [Text("this is long enough text to pass the threshold")],
    ]

    elements = ingest_file(str(pdf_path), "application/pdf", prefer="tika", tika=tika)

    assert any("long enough" in el.text for el in elements)
    assert tika.extract.call_args_list == [
        call(str(pdf_path), "application/pdf"),
        call(str(pdf_path), "application/pdf", ocr="ocr_and_text"),
    ]


def test_extract_pdf_with_tika(sample_pdf):
    adapter = TikaAdapter()
    with patch("requests.put", return_value=_fake_response("Hello PDF")):
        elements = adapter.extract(str(sample_pdf), "application/pdf")
    assert any("Hello PDF" in el.text for el in elements)


def test_extract_docx_with_tika(sample_docx):
    adapter = TikaAdapter()
    with patch(
        "requests.put",
        return_value=_fake_response("Hello DOCX"),
    ):
        elements = adapter.extract(
            str(sample_docx),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    assert any("Hello DOCX" in el.text for el in elements)


def test_extract_pptx_with_tika(sample_pptx):
    adapter = TikaAdapter()
    with patch(
        "requests.put",
        return_value=_fake_response("Hello PPTX"),
    ):
        elements = adapter.extract(
            str(sample_pptx),
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    assert any("Hello PPTX" in el.text for el in elements)
