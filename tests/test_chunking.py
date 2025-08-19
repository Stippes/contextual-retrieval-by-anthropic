import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.ingest.chunking import chunk_elements


def test_heading_and_list_grouping():
    elements = [
        {"type": "Title", "text": "Heading", "metadata": {"page_number": 1}},
        {"type": "NarrativeText", "text": "Paragraph", "metadata": {"page_number": 1}},
        {"type": "ListItem", "text": "Item 1", "metadata": {"page_number": 1}},
        {"type": "ListItem", "text": "Item 2", "metadata": {"page_number": 1}},
    ]
    chunks = chunk_elements(elements, target_tokens=100, max_tokens=200)
    assert len(chunks) == 1
    text = chunks[0]["text"]
    assert "Heading" in text and "Item 1" in text and "Item 2" in text
    assert chunks[0]["metadata"].get("section_title") == "Heading"


def test_slide_and_notes():
    elements = [
        {"type": "Slide", "text": "Slide content", "metadata": {"slide_number": 1}},
        {"type": "SlideNote", "text": "Note content", "metadata": {"slide_number": 1}},
    ]
    chunks = chunk_elements(elements, target_tokens=50, max_tokens=100)
    assert any(c["metadata"].get("slide_id") == 1 for c in chunks)
    note_chunks = [c for c in chunks if c["metadata"].get("section_title") == "slide_note"]
    assert note_chunks and "Note content" in note_chunks[0]["text"]


def test_xlsx_sheet_metadata():
    elements = [
        {"type": "NarrativeText", "text": "Cell A", "metadata": {"page_name": "Sheet1"}},
        {"type": "NarrativeText", "text": "Cell B", "metadata": {"page_name": "Sheet1"}},
        {"type": "NarrativeText", "text": "Cell C", "metadata": {"page_name": "Sheet2"}},
    ]
    chunks = chunk_elements(elements, target_tokens=50, max_tokens=100)
    assert chunks[0]["metadata"].get("sheet") == "Sheet1"
    assert chunks[1]["metadata"].get("sheet") == "Sheet2"
