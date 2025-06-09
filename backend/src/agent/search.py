"""Utility functions for web search."""

import os
from typing import Dict, List

import requests

SERPAPI_URL = "https://serpapi.com/search.json"


def serpapi_search(query: str, *, num_results: int = 5) -> List[Dict[str, str]]:
    """Query SerpAPI and return a list of search results.

    Each result dictionary contains ``title``, ``link`` and ``snippet`` keys.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if api_key is None:
        raise ValueError("SERPAPI_API_KEY is not set")

    params = {"q": query, "api_key": api_key, "num": num_results}
    response = requests.get(SERPAPI_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results: List[Dict[str, str]] = []
    for item in data.get("organic_results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )
    return results
