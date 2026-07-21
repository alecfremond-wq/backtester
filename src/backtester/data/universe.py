from __future__ import annotations

# Curated basket of ~90 liquid large/mid-cap US tickers spanning multiple
# sectors, used to give cross-sectional strategies enough breadth to work
# with (a handful of mega-caps isn't representative of the hundreds of names
# academic cross-sectional momentum research is based on).
LARGE_CAP_UNIVERSE: list[str] = [
    # technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
    "ADBE", "CSCO", "ACN", "IBM", "INTC", "AMD", "QCOM", "TXN", "INTU", "NOW",
    "AMAT", "MU", "ADI", "LRCX", "KLAC",
    # healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "MDT", "ISRG", "VRTX", "REGN", "ZTS",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "SPGI",
    "USB", "PNC", "TFC", "COF",
    # consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "HD", "LOW",
    "TGT", "DIS", "CMCSA", "NFLX", "BKNG",
    # industrials
    "HON", "UPS", "CAT", "DE", "RTX", "BA", "LMT", "GE", "MMM", "UNP", "FDX",
    # energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # communications
    "T", "VZ", "TMUS",
]

UNIVERSES: dict[str, list[str]] = {
    "large_cap": LARGE_CAP_UNIVERSE,
}
