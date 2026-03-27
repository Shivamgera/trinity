"""Fetch real AAPL-tagged headlines from Polygon.io REST API.

Uses the /v2/reference/news endpoint with ticker=AAPL to retrieve
historical news articles. Paginates through the full date range
(Jan 2020 – Dec 2024), deduplicates to one headline per trading day,
and saves as JSON.

API docs: https://polygon.io/docs/rest/stocks/news
Free tier: 5 requests/minute, no daily cap, historical access.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
output_path = project_root / "data" / "processed" / "headlines.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Polygon.io configuration
BASE_URL = "https://api.polygon.io/v2/reference/news"
TICKER = "AAPL"
DATE_START = "2020-01-01"  # warmup period start (expanded for retraining)
DATE_END = "2024-12-31"    # test period end
PAGE_LIMIT = 1000          # max allowed per request
RATE_LIMIT_SLEEP = 12.5    # seconds between requests (5 req/min = 12s)


def fetch_all_articles(api_key: str) -> list[dict]:
    """Fetch all AAPL-tagged articles from Polygon.io, paginating via next_url.

    Returns raw article dicts from the API with keys like 'title',
    'published_utc', 'publisher', 'tickers', etc.
    """
    articles: list[dict] = []
    url = BASE_URL
    params = {
        "ticker": TICKER,
        "published_utc.gte": f"{DATE_START}T00:00:00Z",
        "published_utc.lte": f"{DATE_END}T23:59:59Z",
        "limit": PAGE_LIMIT,
        "sort": "published_utc",
        "order": "asc",
        "apiKey": api_key,
    }

    page = 0
    while url:
        page += 1
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  Request failed on page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        articles.extend(results)
        print(f"  Page {page}: fetched {len(results)} articles "
              f"(total: {len(articles)})")

        # Pagination: Polygon returns a next_url with a cursor token.
        # On subsequent requests, use next_url directly (no params needed).
        next_url = data.get("next_url")
        if next_url:
            # Append API key to next_url (Polygon requires it on every request)
            url = f"{next_url}&apiKey={api_key}"
            params = {}  # next_url already contains all query params
            time.sleep(RATE_LIMIT_SLEEP)
        else:
            url = None

    return articles


def load_trading_dates(ticker: str) -> set[str]:
    """Load the set of actual trading dates from the raw OHLCV file.

    These are dates where the market was open and AAPL traded.
    Used to filter headlines to only trading days.
    """
    ohlcv_path = project_root / "data" / "raw" / f"{ticker.lower()}_ohlcv.parquet"
    if not ohlcv_path.exists():
        raise FileNotFoundError(
            f"Raw OHLCV not found at {ohlcv_path}. Run P1-T1 first."
        )
    df = pd.read_parquet(ohlcv_path)
    return {d.strftime("%Y-%m-%d") for d in df.index}


def deduplicate_to_one_per_day(
    articles: list[dict],
    trading_dates: set[str],
) -> list[dict]:
    """Select one headline per trading day from the fetched articles.

    For each trading day, picks the first article (by published_utc)
    that mentions AAPL in its tickers list. Articles on non-trading
    days (weekends, holidays) are discarded.

    Returns headline dicts in the output schema:
        {date, ticker, headline, source}
    """
    # Group articles by trading date
    day_to_articles: dict[str, list[dict]] = {}
    for article in articles:
        pub = article.get("published_utc", "")
        if not pub:
            continue
        date_str = pub[:10]  # "YYYY-MM-DD"

        if date_str not in trading_dates:
            continue

        # Confirm AAPL is actually tagged (not just a passing mention)
        tickers = article.get("tickers", [])
        if TICKER not in tickers:
            continue

        if date_str not in day_to_articles:
            day_to_articles[date_str] = []
        day_to_articles[date_str].append(article)

    # For each trading day, take the first article (already sorted by
    # published_utc ascending from the API)
    headlines = []
    for date_str in sorted(day_to_articles.keys()):
        article = day_to_articles[date_str][0]
        publisher = article.get("publisher", {})
        headlines.append({
            "date": date_str,
            "ticker": TICKER,
            "headline": article.get("title", "").strip(),
            "source": publisher.get("name", "unknown"),
        })

    return headlines


def main():
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set in environment.")
        print("Sign up for a free account at https://polygon.io")
        print("Then add POLYGON_API_KEY=pk_... to your .env file.")
        sys.exit(1)

    print(f"Fetching {TICKER} headlines from Polygon.io...")
    print(f"Date range: {DATE_START} to {DATE_END}")
    print(f"Rate limit: {RATE_LIMIT_SLEEP}s between pages\n")

    # Step 1: Fetch all articles from the API
    articles = fetch_all_articles(api_key)
    print(f"\nTotal articles fetched: {len(articles)}")

    if not articles:
        print("ERROR: No articles returned. Check your API key and try again.")
        sys.exit(1)

    # Step 2: Load trading calendar from P1-T1 output
    trading_dates = load_trading_dates(TICKER)
    print(f"Trading dates in OHLCV: {len(trading_dates)}")

    # Step 3: Deduplicate to one headline per trading day
    headlines = deduplicate_to_one_per_day(articles, trading_dates)
    print(f"Headlines after deduplication: {len(headlines)}")

    # Step 4: Report coverage
    covered_dates = {h["date"] for h in headlines}
    # Only count trading dates within our target range
    target_dates = {
        d for d in trading_dates
        if DATE_START <= d <= DATE_END
    }
    coverage = len(covered_dates & target_dates) / len(target_dates) if target_dates else 0
    missing = sorted(target_dates - covered_dates)

    print(f"Coverage: {coverage:.1%} of {len(target_dates)} trading days")
    if missing:
        print(f"Missing dates ({len(missing)}):")
        for d in missing[:20]:
            print(f"  {d}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    # Step 5: Save
    with open(output_path, "w") as f:
        json.dump(headlines, f, indent=2)

    print(f"\nSaved {len(headlines)} headlines to {output_path}")
    print(f"Date range: {headlines[0]['date']} to {headlines[-1]['date']}")

    # Summary statistics
    sources = {}
    for h in headlines:
        sources[h["source"]] = sources.get(h["source"], 0) + 1
    print("\nHeadlines by source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1])[:10]:
        print(f"  {src}: {count}")

    # Warn if coverage is low
    if coverage < 0.90:
        print(f"\nWARNING: Coverage is {coverage:.1%}, below the 90% target.")
        print("The C-Gate handles missing Analyst signals by treating them as")
        print("conflict (delta=1.0, action=flat), but low coverage degrades")
        print("the system's ability to demonstrate agreement regimes.")


if __name__ == "__main__":
    main()
