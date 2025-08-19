import os
from types import SimpleNamespace
import sys

sys.path.insert(0, os.path.abspath('.'))

from src.extractors import load_documents
from src.ingest.chunking import chunk_elements


def _el(text: str, typ: str, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(text=text, type=typ, metadata=metadata)


def test_ingestion_office(monkeypatch):
    pptx = os.path.join('tests', 'data', 'sample.pptx')
    xlsx = os.path.join('tests', 'data', 'sample.xlsx')
    unsupported = os.path.join('tests', 'data', 'unsupported.txt')

    def fake_extract(path: str, mime: str):
        if path.endswith('.pptx'):
            return [
                _el('Slide body', 'Slide', {'slide_number': 1}),
                _el('Note body', 'SlideNote', {'slide_number': 1}),
            ]
        if path.endswith('.xlsx'):
            return [
                _el('Header\nRow1', 'Table', {'sheet': 'Sheet1'}),
                _el('Header2\nRow2', 'Table', {'sheet': 'Sheet2'}),
            ]
        raise ValueError('unsupported')

    tika_calls = {}

    def fake_tika_extract(self, path: str, mime: str, ocr: str | None = None):
        tika_calls['path'] = path
        return [_el('Fallback body', 'NarrativeText', {})]

    monkeypatch.setattr('src.extractors.extract_unstructured', fake_extract)
    monkeypatch.setattr('src.extractors.TikaAdapter.extract', fake_tika_extract)

    elements = load_documents([pptx, xlsx, unsupported])
    chunks = chunk_elements(elements, target_tokens=5, max_tokens=10)

    slide_chunks = [c for c in chunks if c['metadata'].get('slide_id') == 1]
    assert any('Slide body' in c['text'] for c in slide_chunks)

    note_chunks = [c for c in chunks if c['metadata'].get('section_title') == 'slide_note']
    assert note_chunks and 'Note body' in note_chunks[0]['text']

    sheet1 = [c for c in chunks if c['metadata'].get('sheet') == 'Sheet1']
    assert sheet1 and 'Header' in sheet1[0]['text']

    sheet2 = [c for c in chunks if c['metadata'].get('sheet') == 'Sheet2']
    assert sheet2

    assert tika_calls.get('path', '').endswith('unsupported.txt')
