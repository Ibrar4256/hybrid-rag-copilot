"""One-off corpus builder: pulls 10-K/10-Q filings for the eval corpus (regional
banks, FY2022-2023) from SEC EDGAR and converts them to clean-ish markdown.
Not part of the research_copilot package — this is a data-prep script, run once."""

import re
import sys
import time
import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

USER_AGENT = "research-copilot-project ibrar.ali@techverx.com"
OUT_DIR = Path("data/sec_filings")

# ticker -> CIK (resolved via SEC's company_tickers.json / EDGAR company search)
COMPANIES = {
    "WAL": "1212545",
    "ZION": "109380",
    "CMA": "28412",
    "MTB": "36270",
    "VLY": "714310",
    "EWBC": "1069157",
}

# Companies to also pull 2023 10-Qs for (temporal-drift spot-check)
TEN_Q_COMPANIES = {"WAL", "ZION"}


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    time.sleep(0.2)  # be polite to SEC's rate limits
    return resp


def _list_filings(cik: str) -> list[dict]:
    data = _get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json").json()
    recent = data["filings"]["recent"]
    return [
        {
            "form": recent["form"][i],
            "filing_date": recent["filingDate"][i],
            "period_end": recent["reportDate"][i],
            "accession": recent["accessionNumber"][i],
            "primary_doc": recent["primaryDocument"][i],
        }
        for i in range(len(recent["form"]))
    ]


def _html_to_text(html: str) -> str:
    # SEC filings are inline-XBRL: the visible document is preceded/interleaved
    # with a huge <ix:header> block of machine-readable tagging facts, plus
    # display:none elements throughout. Both would otherwise get scraped as
    # if they were prose.
    html = re.sub(r"<ix:header.*?</ix:header>", "", html, flags=re.DOTALL | re.IGNORECASE)

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE)):
        tag.decompose()

    # Render tables as pipe-delimited rows instead of flattening them to a
    # single run-on string — preserves row/column structure as a genuine
    # (if messy) test of the chunker, per ADR-003's "tables that break naive
    # chunking" goal, rather than destroying the structure entirely.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")

    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_company(ticker: str, cik: str) -> None:
    filings = _list_filings(cik)
    tenks = [
        f
        for f in filings
        if f["form"] == "10-K" and f["period_end"][:4] in ("2022", "2023")
    ]
    wanted = list(tenks)
    if ticker in TEN_Q_COMPANIES:
        tenqs = [
            f
            for f in filings
            if f["form"] == "10-Q" and f["period_end"].startswith("2023")
        ]
        wanted += tenqs

    for f in wanted:
        acc_nodash = f["accession"].replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{f['primary_doc']}"
        print(f"Fetching {ticker} {f['form']} ({f['period_end']}) from {url}")
        html = _get(url).text
        text = _html_to_text(html)

        out_path = OUT_DIR / f"{ticker}_{f['form']}_{f['period_end']}.md"
        out_path.write_text(
            f"# {ticker} — {f['form']} — period ending {f['period_end']}\n"
            f"Filed: {f['filing_date']} | Source: {url}\n\n" + text
        )
        print(f"  -> {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = sys.argv[1:] or list(COMPANIES)
    for ticker in tickers:
        fetch_company(ticker, COMPANIES[ticker])
