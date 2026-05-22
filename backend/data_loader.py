import numpy as np
import contextlib
import io
import json
import time
import urllib.parse
import urllib.request
import yfinance as yf


TICKERS = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD", "TSLA", "SPY"]
CRYPTO_IDS = {"BTC-USD": "bitcoin", "ETH-USD": "ethereum"}
PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
DEMO_START_PRICES = {
    "AAPL": 190.0,
    "MSFT": 420.0,
    "GOOGL": 175.0,
    "TSLA": 250.0,
    "SPY": 520.0,
    "BTC-USD": 70000.0,
    "ETH-USD": 3500.0,
}


def _prices_from_yfinance(ticker: str, period: str) -> np.ndarray:
    last_error = None
    for _ in range(3):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                df = yf.download(
                    ticker,
                    period=period,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                if not df.empty:
                    return df["Close"].values.flatten().astype(np.float32)

                history = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if not history.empty:
                    return history["Close"].values.flatten().astype(np.float32)
        except Exception as exc:
            last_error = exc
        time.sleep(1)

    if last_error is not None:
        raise ValueError(f"Yahoo Finance download failed for {ticker}: {last_error}")
    raise ValueError(f"Yahoo Finance returned no data for {ticker}")


def _prices_from_coingecko(ticker: str, period: str) -> np.ndarray:
    coin_id = CRYPTO_IDS[ticker]
    days = PERIOD_DAYS.get(period, 365)
    query = urllib.parse.urlencode({
        "vs_currency": "usd",
        "days": str(days),
        "interval": "daily",
    })
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "rl-trading-demo/1.0"})

    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    prices = payload.get("prices", [])
    if not prices:
        raise ValueError(f"CoinGecko returned no price data for {ticker}")

    return np.array([point[1] for point in prices], dtype=np.float32)


def _demo_prices(ticker: str, period: str) -> np.ndarray:
    days = PERIOD_DAYS.get(period, 365)
    points = max(22, int(days * 5 / 7))
    start = DEMO_START_PRICES.get(ticker, 100.0)
    seed = sum(ord(char) for char in f"{ticker}:{period}")
    rng = np.random.default_rng(seed)

    drift = 0.0004 if ticker not in CRYPTO_IDS else 0.0009
    volatility = 0.018 if ticker not in CRYPTO_IDS else 0.035
    returns = rng.normal(drift, volatility, points)
    prices = start * np.cumprod(1.0 + returns)
    return np.maximum(prices, start * 0.2).astype(np.float32)


def load_data(ticker: str, period: str = "1y") -> np.ndarray:
    if ticker not in TICKERS:
        raise ValueError(f"Ticker {ticker} not supported. Choose from: {TICKERS}")

    if ticker in CRYPTO_IDS:
        try:
            prices = _prices_from_coingecko(ticker, period)
        except Exception as crypto_error:
            try:
                prices = _prices_from_yfinance(ticker, period)
            except Exception as yahoo_error:
                print(
                    f"Using demo data for {ticker}: "
                    f"CoinGecko error: {crypto_error}. Yahoo error: {yahoo_error}"
                )
                prices = _demo_prices(ticker, period)
    else:
        try:
            prices = _prices_from_yfinance(ticker, period)
        except Exception as yahoo_error:
            print(f"Using demo data for {ticker}: {yahoo_error}")
            prices = _demo_prices(ticker, period)

    if len(prices) < 12:
        raise ValueError(f"Not enough price data returned for {ticker}: {len(prices)} points")

    return prices
