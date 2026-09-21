# Financial News Sentiment & Market Ingestion Pipeline

An automated Natural Language Processing (NLP) data pipeline that streams financial headlines for equities, classifies sentiment using a transformer model fine-tuned on financial corpora (`ProsusAI/finbert`), and aligns extracted scores with asset price action for quantitative feature generation.

## Architecture

```text
[Ticker Feed] ──► yfinance API ──► [Unstructured Headlines]
                                          │
                                          ▼
                             [ProsusAI/finbert Pipeline]
                                          │
                                          ▼
                      [Sentiment Direction & Confidence Scoring]
                                          │
                                          ▼
                       Weighted Composite Sentiment Index (t)
                                          │
                    Aligned with Forward 1-Day Price Return (t+1)
                                          │
                                          ▼
                          [Structured CSV Signal Table]

```

## Features

- **Domain-Specific Transformer:** Leverages `ProsusAI/finbert` to accurately classify financial terms that generic models misinterpret (e.g., "rate hikes", "depreciation").
- **Confidence-Weighted Signal Aggregation:** Transforms qualitative classifications (`positive`, `neutral`, `negative`) into continuous numerical signals bounded between `[-1.0, 1.0]`:
  $$\text{Weighted Score} = \text{Direction} \times \text{Confidence}$$
- **Forward Horizon Alignment:** Aligns news sentiment at time $t$ against the subsequent day's price change ($t+1$) to prevent lookahead bias in downstream backtesting.

## Sample Output

### Headline Level Ingestion
| Ticker | Headline | Sentiment | Confidence | Weighted Score | Price Change (1D) |
|---|---|---|---|---|---|
| AAPL | Apple Pay to launch in India next month with Axis Bank | NEUTRAL | 0.7712 | 0.0000 | -0.26% |
| AAPL | tech crunch hikes the price of your next iPhone | NEGATIVE | 0.9421 | -0.9421 | -0.26% |
| NVDA | Nvidia Partners on Next-Gen Data Center Architecture | POSITIVE | 0.8845 | +0.8845 | +1.42% |

### Aggregated Asset Composite
| Ticker | Avg Confidence | Composite Sentiment | Price Change (1D) |
|---|---|---|---|
| AAPL | 0.8566 | -0.4710 | -0.26% |
| NVDA | 0.8845 | +0.8845 | +1.42% |

## Installation & Setup

1. **Clone the repository:**
   ```bash 
   git clone [https://github.com/shantosh-m/financial-sentiment-pipeline.git](https://github.com/shantosh-m/financial-sentiment-pipeline.git)
   cd financial-sentiment-pipeline