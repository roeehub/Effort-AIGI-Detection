"""
MCP Web Tools Server — gives Copilot agents real web access.

Tools provided:
  - fetch_url        : raw HTML from any URL (truncated)
  - fetch_text       : cleaned readable text from any URL
  - fetch_arxiv      : structured abstract/title/authors from arXiv
  - search_web       : DuckDuckGo search results
  - fetch_github_raw : raw file content from a GitHub repo

Run via VS Code MCP integration (stdio transport).
"""

from __future__ import annotations

import re
import urllib.parse
from textwrap import dedent

import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
mcp = FastMCP("web-tools")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 15  # seconds


def _get(url: str, **kwargs) -> requests.Response:
    """Shared GET with default headers/timeout."""
    return requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, **kwargs)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def fetch_url(url: str, max_chars: int = 20_000) -> str:
    """Fetch raw HTML from a URL. Returns first `max_chars` characters.

    Use this when you need the raw page source (e.g. to parse specific
    elements yourself). For readable text, prefer `fetch_text`.
    """
    try:
        r = _get(url)
        r.raise_for_status()
        return r.text[:max_chars]
    except Exception as exc:
        return f"ERROR fetching {url}: {exc}"


@mcp.tool()
def fetch_text(url: str, max_chars: int = 15_000) -> str:
    """Fetch a webpage and return **clean readable text** (no HTML tags).

    Strips scripts, styles, navs, footers, and ads.  Good for reading
    documentation, blog posts, and papers.
    """
    try:
        r = _get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "iframe", "svg"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:max_chars]
    except Exception as exc:
        return f"ERROR fetching text from {url}: {exc}"


@mcp.tool()
def fetch_arxiv(arxiv_id: str) -> str:
    """Fetch structured info from an arXiv paper.

    Args:
        arxiv_id: The arXiv ID (e.g. '2301.00001' or '2301.00001v2')
                  or a full arXiv URL. Automatically extracts the ID.

    Returns:
        Title, authors, abstract, and links in a clean format.
    """
    # Accept full URLs like https://arxiv.org/abs/2301.00001
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", arxiv_id)
    if not m:
        return f"ERROR: Could not parse arXiv ID from '{arxiv_id}'"
    aid = m.group(0)

    api_url = f"http://export.arxiv.org/api/query?id_list={aid}"
    try:
        r = _get(api_url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        entry = soup.find("entry")
        if not entry:
            return f"No arXiv entry found for {aid}"

        title = entry.find("title").get_text(strip=True)
        authors = [a.find("name").get_text(strip=True)
                    for a in entry.find_all("author")]
        abstract = entry.find("summary").get_text(strip=True)
        published = entry.find("published").get_text(strip=True)[:10]
        pdf_link = f"https://arxiv.org/pdf/{aid}"

        return dedent(f"""\
            arXiv:{aid}
            Title: {title}
            Authors: {', '.join(authors)}
            Published: {published}
            PDF: {pdf_link}

            Abstract:
            {abstract}
        """)
    except Exception as exc:
        return f"ERROR fetching arXiv {aid}: {exc}"


@mcp.tool()
def search_web(query: str, max_results: int = 8) -> str:
    """Search the web using DuckDuckGo and return top results.

    Returns a list of titles + URLs + snippets.  Use this to find papers,
    repos, documentation, or any web content.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 8).
    """
    try:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        r = _get(url)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for i, result_div in enumerate(soup.select(".result")):
            if i >= max_results:
                break
            title_el = result_div.select_one(".result__title a,  .result__a")
            snippet_el = result_div.select_one(".result__snippet")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            # DuckDuckGo wraps URLs in redirects; try to extract actual URL
            if "uddg=" in href:
                actual = urllib.parse.parse_qs(
                    urllib.parse.urlparse(href).query
                ).get("uddg", [href])[0]
                href = urllib.parse.unquote(actual)
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append(f"{i+1}. {title}\n   URL: {href}\n   {snippet}")

        if not results:
            return f"No results found for '{query}'"
        return "\n\n".join(results)
    except Exception as exc:
        return f"ERROR searching for '{query}': {exc}"


@mcp.tool()
def fetch_github_raw(owner: str, repo: str, path: str,
                     branch: str = "main") -> str:
    """Fetch a raw file from a public GitHub repository.

    Args:
        owner: GitHub username or org (e.g. 'pytorch')
        repo: Repository name (e.g. 'pytorch')
        path: File path within the repo (e.g. 'README.md')
        branch: Branch name (default 'main')

    Returns:
        Raw file content (first 20000 chars).
    """
    url = (f"https://raw.githubusercontent.com/"
           f"{owner}/{repo}/{branch}/{path}")
    try:
        r = _get(url)
        r.raise_for_status()
        return r.text[:20_000]
    except Exception as exc:
        return f"ERROR fetching {url}: {exc}"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
