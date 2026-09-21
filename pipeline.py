"""
Financial Sentiment & Market Ingestion Pipeline
Extracts recent market news via Yahoo Finance, classifies contextual sentiment
using FinBERT (ProsusAI/finbert), and computes confidence-weighted daily sentiment
scores aligned with forward asset price movements.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCORE_MAP = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0
}

def load_sentiment_model():
    logging.info("Initializing Hugging Face FinBERT pipeline...")
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def fetch_ticker_data(ticker_symbol: str, num_articles: int = 5):
    logging.info(f"Fetching news and price history for ticker: {ticker_symbol}")
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Fetch News
    raw_news = ticker.news or []
    extracted_headlines = []
    for item in raw_news[:num_articles]:
        title = item.get("content", {}).get("title", item.get("title", ""))
        if title:
            extracted_headlines.append(title)
            
    # 2. Fetch Price History (Last 5 trading days)
    hist = ticker.history(period="5d")
    pct_change_1d = None
    if len(hist) >= 2:
        prev_close = hist["Close"].iloc[-2]
        curr_close = hist["Close"].iloc[-1]
        pct_change_1d = round(((curr_close - prev_close) / prev_close) * 100, 2)
        
    return extracted_headlines, pct_change_1d

def analyze_assets(tickers: list[str], output_csv: str = "sentiment_market_signals.csv"):
    nlp = load_sentiment_model()
    records = []

    for sym in tickers:
        headlines, price_change = fetch_ticker_data(sym)
        if not headlines:
            logging.warning(f"No headlines found for {sym}")
            continue

        ticker_weighted_scores = []

        for text in headlines:
            pred = nlp(text)[0]
            label = pred["label"].lower()
            confidence = float(pred["score"])
            
            # Weighted score: Direction * Confidence
            numerical_impact = SCORE_MAP.get(label, 0.0) * confidence
            ticker_weighted_scores.append(numerical_impact)

            records.append({
                "Ticker": sym,
                "Headline": text,
                "Sentiment_Label": label.upper(),
                "Confidence": round(confidence, 4),
                "Weighted_Score": round(numerical_impact, 4),
                "Forward_1D_Return_%": price_change
            })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    logging.info(f"Analysis saved to {output_csv}")

    # Summary table
    if not df.empty:
        summary = df.groupby("Ticker").agg(
            Avg_Confidence=("Confidence", "mean"),
            Composite_Sentiment=("Weighted_Score", "mean"),
            Price_Change_1D=("Forward_1D_Return_%", "first")
        ).reset_index()
        
        print("\n=== AGGREGATED DAILY ASSET SIGNALS ===")
        print(summary.to_string(index=False))

    return df

if __name__ == "__main__":
    target_tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "AMD"]
    analyze_assets(target_tickers)