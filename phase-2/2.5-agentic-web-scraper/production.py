# requirements: langgraph==0.2.55 langchain-openai==0.2.14 httpx==0.28.1 beautifulsoup4==4.12.3 lxml==5.3.0 structlog==24.4.0
from __future__ import annotations

import ipaddress
from urllib.parse import urljoin, urlparse
from typing import Literal, TypedDict

import httpx
import structlog
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph


structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, timeout=10.0)


class CrawlState(TypedDict):
    allowed_domains: list[str]
    queue: list[dict[str, object]]
    visited_urls: list[str]
    extracted_items: list[dict[str, str]]
    status: Literal["planned", "fetched", "extracted", "blocked"]


def is_safe_url(url: str, allowed_domains: set[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.hostname is None or parsed.hostname not in allowed_domains:
        return False
    try:
        ip = ipaddress.ip_address(parsed.hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False
    except ValueError:
        pass
    return True


def fetch_node(state: CrawlState) -> CrawlState:
    current = state["queue"].pop(0)
    url = str(current["url"])
    if not is_safe_url(url, set(state["allowed_domains"])):
        state["status"] = "blocked"
        return state
    response = httpx.get(url, timeout=10.0, headers={"User-Agent": "AgenticAIZeroToHeroBot/1.0"})
    response.raise_for_status()
    state["visited_urls"].append(url)
    state["queue"].insert(0, {"url": url, "depth": current["depth"], "html": response.text})
    state["status"] = "fetched"
    return state


def extract_node(state: CrawlState) -> CrawlState:
    current = state["queue"].pop(0)
    html = str(current["html"])
    url = str(current["url"])
    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)[:8000]
    response = model.invoke(f"Extract title and summary from this page text only: {text}")
    state["extracted_items"].append({"title": "page", "summary": str(response.content), "source_url": url})
    if int(current["depth"]) < 1:
        soup = BeautifulSoup(html, "lxml")
        for anchor in soup.select("a[href]")[:5]:
            candidate = urljoin(url, anchor.get("href", ""))
            if candidate not in state["visited_urls"] and is_safe_url(candidate, set(state["allowed_domains"])):
                state["queue"].append({"url": candidate, "depth": int(current["depth"]) + 1})
    state["status"] = "extracted"
    return state


def route_after_extract(state: CrawlState) -> str:
    if not state["queue"]:
        return END
    return "fetch"


graph = StateGraph(CrawlState)
graph.add_node("fetch", fetch_node)
graph.add_node("extract", extract_node)
graph.set_entry_point("fetch")
graph.add_edge("fetch", "extract")
graph.add_conditional_edges("extract", route_after_extract, {END: END, "fetch": "fetch"})
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("web_scraper.sqlite3"))
