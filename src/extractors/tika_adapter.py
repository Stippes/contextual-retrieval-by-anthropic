from __future__ import annotations

"""Adapter for interacting with an Apache Tika server."""

import os
from typing import List

import requests
from unstructured.documents.elements import Element, Text


class TikaAdapter:
    """Client for Apache Tika's ``/rmeta/text`` endpoint."""

    def __init__(
        self,
        url: str | None = None,
        *,
        timeout: int | None = None,
        write_limit: str | None = None,
        max_embedded_resources: str | None = None,
    ) -> None:
        self.url = url or os.environ.get("TIKA_URL", "http://localhost:9998")
        self.timeout = timeout or int(os.environ.get("TIKA_TIMEOUT", "60"))
        self.write_limit = write_limit or os.environ.get(
            "X_TIKA_WRITELIMIT", "-1"
        )
        self.max_embedded_resources = max_embedded_resources or os.environ.get(
            "X_TIKA_MAX_EMBEDDED_RESOURCES", "1000"
        )

    def extract(self, path: str, mime: str, ocr: str | None = None) -> list[Element]:
        """Extract elements from ``path`` using Apache Tika."""
        headers = {
            "Content-Disposition": f'attachment; filename="{os.path.basename(path)}"',
            "X-Tika-PDFOcrStrategy": ocr or "auto",
            "X-Tika-WriteLimit": self.write_limit,
            "X-Tika-MaxEmbeddedResources": self.max_embedded_resources,
            "Accept": "application/json",
            "Content-Type": mime,
        }
        with open(path, "rb") as f:
            response = requests.put(
                f"{self.url}/rmeta/text",
                headers=headers,
                data=f.read(),
                timeout=self.timeout,
            )
        response.raise_for_status()
        data = response.json()
        elements: List[Element] = []
        for item in data:
            content = item.get("X-TIKA:content") or item.get("content")
            if content:
                elements.append(Text(content.strip()))
        return elements
