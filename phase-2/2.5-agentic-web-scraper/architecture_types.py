from __future__ import annotations

from typing import Literal, TypedDict


class CrawlTask(TypedDict):
    url: str
    depth: int


class CrawlState(TypedDict):
    queue: list[CrawlTask]
    visited_urls: list[str]
    extracted_items: list[dict[str, str]]
    status: Literal["planned", "fetched", "extracted", "blocked"]
