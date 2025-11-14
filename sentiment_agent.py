#!/usr/bin/env python3
"""Daily sentiment monitoring agent for Cape Town's city center.

The script fetches public text from multiple sources, applies VADER sentiment
analysis, tags each entry with city themes (safety, vibrancy, etc.), and
appends the output to a CSV data store for downstream analytics.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError as exc:  # pragma: no cover - nltk should be installed by user
        raise SystemExit(
            "nltk is required. Install it with `pip install nltk`."
        ) from exc

try:
    import snscrape.modules.twitter as sntwitter
except (ImportError, AttributeError):  # pragma: no cover - optional dependency
    sntwitter = None


# Keyword buckets for high-level themes.
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Safety": [
        "safe",
        "unsafe",
        "danger",
        "dangerous",
        "crime",
        "mugging",
        "secure",
        "security",
        "police",
        "pickpocket",
        "watch your pockets",
    ],
    "Vibrancy": [
        "vibrant",
        "lively",
        "buzzing",
        "nightlife",
        "party",
        "quiet",
        "boring",
        "dull",
        "ghost town",
    ],
    "Beauty": [
        "beautiful",
        "pretty",
        "scenic",
        "clean",
        "dirty",
        "picturesque",
        "charming",
        "ugly",
        "views",
        "street art",
    ],
    "Convenience": [
        "central",
        "accessible",
        "walkable",
        "transport",
        "uber",
        "location",
        "close to",
    ],
    "Cost": [
        "expensive",
        "pricey",
        "affordable",
        "cheap",
        "value",
        "cost",
    ],
}


URL_PATTERN = re.compile(r"http\S+")
HANDLE_PATTERN = re.compile(r"@\w+")
HASH_PATTERN = re.compile(r"#")


@dataclass
class RawRecord:
    """Unprocessed text snippet gathered from any source."""

    source: str
    text: str
    created_at: datetime
    url: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProcessedRecord:
    """Record enriched with sentiment and category labels."""

    timestamp: datetime
    source: str
    text: str
    sentiment: str
    sentiment_score: float
    categories: List[str]
    url: Optional[str]
    metadata: Dict[str, str]


class DataSource:
    """Base interface for any source."""

    name: str

    def fetch(self) -> List[RawRecord]:  # pragma: no cover - interface only
        raise NotImplementedError


class TwitterSource(DataSource):
    """Fetches tweets via snscrape using keyword queries."""

    def __init__(
        self,
        queries: List[str],
        max_results: int,
        lookback_hours: int,
    ) -> None:
        self.queries = queries
        self.max_results = max_results
        self.lookback_hours = lookback_hours
        self.name = "twitter"

    def fetch(self) -> List[RawRecord]:
        if sntwitter is None:
            logging.warning("snscrape is not installed; skipping Twitter collection.")
            return []

        since_time = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        records: List[RawRecord] = []

        for query in self.queries:
            search_query = f"{query} since:{since_time.date()}"
            scraper = sntwitter.TwitterSearchScraper(search_query)
            count = 0

            for tweet in scraper.get_items():
                if count >= self.max_results:
                    break
                if tweet.date.tzinfo is None:
                    tweet_time = tweet.date.replace(tzinfo=timezone.utc)
                else:
                    tweet_time = tweet.date
                if tweet_time < since_time:
                    break

                records.append(
                    RawRecord(
                        source=self.name,
                        text=tweet.content,
                        created_at=tweet_time.astimezone(timezone.utc).replace(tzinfo=None),
                        url=f"https://twitter.com/{tweet.user.username}/status/{tweet.id}",
                        metadata={
                            "username": tweet.user.username,
                            "query": query,
                            "like_count": str(getattr(tweet, "likeCount", 0)),
                            "retweet_count": str(getattr(tweet, "retweetCount", 0)),
                        },
                    )
                )
                count += 1

            logging.info("Collected %s tweets for query '%s'.", count, query)

        return records


class CSVReviewSource(DataSource):
    """Loads pre-downloaded reviews (Booking, Airbnb, etc.) from a CSV."""

    def __init__(
        self,
        csv_path: Path,
        text_column: str,
        date_column: str,
        source_label: str,
    ) -> None:
        self.csv_path = csv_path
        self.text_column = text_column
        self.date_column = date_column
        self.name = source_label

    def fetch(self) -> List[RawRecord]:
        if not self.csv_path.exists():
            logging.warning("CSV review file %s not found; skipping.", self.csv_path)
            return []

        records: List[RawRecord] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text = (row.get(self.text_column) or "").strip()
                if not text:
                    continue
                created_at = parse_datetime(row.get(self.date_column, ""))
                records.append(
                    RawRecord(
                        source=self.name,
                        text=text,
                        created_at=created_at,
                        metadata={k: v for k, v in row.items() if k not in {self.text_column, self.date_column}},
                    )
                )

        logging.info("Loaded %s rows from %s.", len(records), self.csv_path)
        return records


def parse_datetime(value: str | datetime) -> datetime:
    """Parse timestamps from CSV or fall back to now."""
    if isinstance(value, datetime):
        return value

    if not value:
        return datetime.utcnow()

    cleaned = value.strip()
    iso_candidate = cleaned.replace("Z", "+00:00")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ):
        try:
            parsed = datetime.strptime(iso_candidate, fmt)
            if parsed.tzinfo:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            continue

    logging.warning("Could not parse timestamp '%s'; using current time.", cleaned)
    return datetime.utcnow()


def clean_text(text: str) -> str:
    """Remove links, handles, and duplicate whitespace."""
    lowered = text.replace("\n", " ").strip()
    lowered = URL_PATTERN.sub("", lowered)
    lowered = HANDLE_PATTERN.sub("", lowered)
    lowered = HASH_PATTERN.sub("", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def label_sentiment(compound_score: float) -> str:
    if compound_score >= 0.05:
        return "Positive"
    if compound_score <= -0.05:
        return "Negative"
    return "Neutral"


def categorize_text(text: str) -> List[str]:
    text_lower = text.lower()
    categories: List[str] = []
    for label, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            categories.append(label)
    return categories or ["General"]


def ensure_vader_ready() -> SentimentIntensityAnalyzer:
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


class SentimentAgent:
    """Co-ordinates data collection, enrichment, and persistence."""

    def __init__(
        self,
        sources: Iterable[DataSource],
        output_path: Path,
        min_length: int = 20,
    ) -> None:
        self.sources = list(sources)
        self.output_path = output_path
        self.analyzer = ensure_vader_ready()
        self.min_length = min_length

    def run(self) -> int:
        raw_records: List[RawRecord] = []
        for source in self.sources:
            try:
                raw_records.extend(source.fetch())
            except Exception:  # pragma: no cover - runtime protection
                logging.exception("Failed to fetch from %s", source.name)

        processed = [
            record
            for record in (self._process_record(raw) for raw in raw_records)
            if record is not None
        ]

        if not processed:
            logging.info("No records processed in this run.")
            return 0

        self._append_to_csv(processed)
        logging.info("Wrote %s records to %s", len(processed), self.output_path)
        return len(processed)

    def _process_record(self, record: RawRecord) -> Optional[ProcessedRecord]:
        cleaned = clean_text(record.text)
        if len(cleaned) < self.min_length:
            return None

        scores = self.analyzer.polarity_scores(cleaned)
        sentiment_label = label_sentiment(scores["compound"])
        categories = categorize_text(cleaned)
        return ProcessedRecord(
            timestamp=record.created_at,
            source=record.source,
            text=cleaned,
            sentiment=sentiment_label,
            sentiment_score=scores["compound"],
            categories=categories,
            url=record.url,
            metadata=record.metadata,
        )

    def _append_to_csv(self, records: List[ProcessedRecord]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp",
            "source",
            "text",
            "sentiment",
            "sentiment_score",
            "categories",
            "url",
            "metadata",
        ]

        file_exists = self.output_path.exists()
        with self.output_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "timestamp": record.timestamp.isoformat(),
                        "source": record.source,
                        "text": record.text,
                        "sentiment": record.sentiment,
                        "sentiment_score": f"{record.sentiment_score:.4f}",
                        "categories": ";".join(record.categories),
                        "url": record.url or "",
                        "metadata": "|".join(f"{k}:{v}" for k, v in record.metadata.items()),
                    }
                )


def build_sources(args: argparse.Namespace) -> List[DataSource]:
    sources: List[DataSource] = []
    if args.twitter_queries:
        queries = [q.strip() for q in args.twitter_queries if q.strip()]
        if queries:
            sources.append(
                TwitterSource(
                    queries=queries,
                    max_results=args.twitter_max_results,
                    lookback_hours=args.lookback_hours,
                )
            )

    if args.local_review_csv:
        sources.append(
            CSVReviewSource(
                csv_path=Path(args.local_review_csv),
                text_column=args.local_text_column,
                date_column=args.local_date_column,
                source_label=args.local_source_label,
            )
        )

    return sources


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor sentiment for Cape Town's CBD across multiple sources."
    )
    parser.add_argument(
        "--output",
        default="data/cape_town_sentiment.csv",
        help="CSV file to append results to (default: data/cape_town_sentiment.csv)",
    )
    parser.add_argument(
        "--twitter-queries",
        nargs="*",
        default=[
            '"Cape Town CBD"',
            '"Bree Street Cape Town"',
            '"Long Street Cape Town"',
            '"Greenmarket Square"',
        ],
        help="List of search queries for snscrape.",
    )
    parser.add_argument(
        "--twitter-max-results",
        type=int,
        default=120,
        help="Maximum tweets to collect per query.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="How many hours back to look when collecting tweets.",
    )
    parser.add_argument(
        "--local-review-csv",
        help="Optional CSV file containing Booking/Airbnb reviews (pre-downloaded).",
    )
    parser.add_argument(
        "--local-text-column",
        default="text",
        help="Column name for review text inside the local CSV.",
    )
    parser.add_argument(
        "--local-date-column",
        default="date",
        help="Column name for timestamps inside the local CSV.",
    )
    parser.add_argument(
        "--local-source-label",
        default="review",
        help="Source label for the local CSV records.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum number of characters required for analysis.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    sources = build_sources(args)
    if not sources:
        raise SystemExit("No sources configured. Provide at least one source.")

    agent = SentimentAgent(
        sources=sources,
        output_path=Path(args.output),
        min_length=args.min_length,
    )
    agent.run()


if __name__ == "__main__":
    main()
