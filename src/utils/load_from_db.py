import csv
import json
import os
import re


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
risk_word_bank_json_path = os.path.join(BASE_DIR, "data", "risk_word_bank.json")
articles_csv_path = os.path.join(BASE_DIR, "data", "articles.csv")
constituents_csv_path = os.path.join(BASE_DIR, "data", "constituents.csv")

with open(risk_word_bank_json_path, "r") as f:
    RISK_KEYWORDS = json.load(f)


GENERIC_NAME_PREFIXES = (
    "A Look At",
    "Assessing",
    "Why",
    "Here",
    "Is",
    "How",
    "What",
    "Tracking",
    "Final Trade",
    "Stock Market Today",
    "Live On",
)


def _clean_company_name(name):
    cleaned = re.sub(r"\s+", " ", name).strip(" ,.-:")

    if not cleaned:
        return None

    if cleaned.startswith(GENERIC_NAME_PREFIXES):
        return None

    if len(cleaned) > 70:
        return None

    return cleaned


def _extract_company_name(text, ticker):
    if not text:
        return None

    patterns = [
        rf"^([A-Z0-9][A-Za-z0-9&.,'\- ]+?)\s+\((?:NYSE|NASDAQ|NasdaqGS|NasdaqGM|NasdaqCM|NYSEARCA):{ticker}\)",
        rf"^([A-Z0-9][A-Za-z0-9&.,'\- ]+?)\s+\({ticker}\)",
        r"^([A-Z0-9][A-Za-z0-9&.,'\- ]+?):",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            cleaned = _clean_company_name(match.group(1))
            if cleaned:
                return cleaned

    return None


def _load_company_metadata_from_dataset():
    metadata = {}

    dataset_path = None
    if os.path.exists(constituents_csv_path):
        dataset_path = constituents_csv_path

    if not dataset_path:
        return metadata

    with open(dataset_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ticker = (row.get("ticker") or row.get("Symbol") or row.get("symbol") or "").strip().upper()
            if not ticker:
                continue

            company_name = (
                row.get("company_name") or row.get("Security") or row.get("security") or ""
            ).strip() or ticker
            logo_url = (row.get("logo_url") or "").strip() or None

            logo_domain = (row.get("logo_domain") or "").strip().lower()
            if logo_domain:
                logo_domain = re.sub(r"^https?://", "", logo_domain)
                logo_domain = logo_domain.split("/")[0]

            if not logo_url and logo_domain:
                logo_url = f"https://logos-api.apistemic.com/domain:{logo_domain}"

            metadata[ticker] = {"company_name": company_name, "logo_url": logo_url}

    return metadata


def _load_company_metadata_from_articles():
    metadata = {}

    with open(articles_csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ticker = row.get("ticker", "").strip().upper()
            if not ticker:
                continue

            entry = metadata.setdefault(
                ticker,
                {"company_name": ticker, "logo_url": None},
            )

            if entry["company_name"] == ticker:
                extracted_name = _extract_company_name(row.get("headline", ""), ticker) or _extract_company_name(
                    row.get("summary", ""), ticker
                )
                if extracted_name:
                    entry["company_name"] = extracted_name

    return metadata

def normalize_headline(h):
    return h.strip().lower()

def _load_article_link_lookup():
    article_links = {}

    if not os.path.exists(articles_csv_path):
        return article_links

    with open(articles_csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            headline = (row.get("headline") or "").strip()
            url = (row.get("url") or "").strip() or None

            if ticker and headline and url:
                article_links[(ticker, headline)] = url

    return article_links


def _load_company_metadata():
    dataset_metadata = _load_company_metadata_from_dataset()

    article_metadata = _load_company_metadata_from_articles()
    combined = {ticker: values.copy() for ticker, values in dataset_metadata.items()}

    for ticker, article_values in article_metadata.items():
        entry = combined.setdefault(ticker, {"company_name": ticker, "logo_url": None})
        if entry.get("company_name", ticker) == ticker and article_values.get("company_name"):
            entry["company_name"] = article_values["company_name"]

    return combined
