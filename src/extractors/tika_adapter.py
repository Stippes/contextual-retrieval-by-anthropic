from __future__ import annotations

"""Adapter for interacting with an Apache Tika server."""

import os
from typing import List

import requests
from unstructured.documents.elements import Element, Text


class TikaAdapter:
    """Client for Apache Tika's ``/rmeta/text`` endpoint."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url or os.environ.get("TIKA_URL", "http://localhost:9998")

    def extract(self, path: str, mime: str) -> list[Element]:
        """Extract elements from ``path`` using Apache Tika."""
        headers = {
            "Content-Disposition": f'attachment; filename="{os.path.basename(path)}"',
            "X-Tika-PDFOcrStrategy": "auto",
            "X-Tika-WriteLimit": "-1",
            "X-Tika-MaxEmbeddedResources": "1000",
            "Accept": "application/json",
            "Content-Type": mime,
        }
        with open(path, "rb") as f:
            response = requests.put(
                f"{self.url}/rmeta/text",
                headers=headers,
                data=f.read(),
                timeout=60,
            )
        response.raise_for_status()
        data = response.json()
        elements: List[Element] = []
        for item in data:
            content = item.get("X-TIKA:content") or item.get("content")
            if content:
                elements.append(Text(content.strip()))
        return elements
