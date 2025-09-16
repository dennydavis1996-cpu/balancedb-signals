# Balanced_B Signals — NIFTY100 (Streamlit + Google Sheets)

Mobile, portfolio-aware signal terminal for Balanced_B:
- Run signals any time the market is open (IST 09:15–15:30). Uses live LTP for decisions.
- Daily mark-to-market performance (close-to-close), comparable to NIFTY100 TRI.
- Google Sheets for your state (balances, positions, ledger, config, daily_equity).
- Charts: Equity vs NIFTY100 TRI (fallback labeled), exposure, drawdown, rolling 1Y CAGR.
- Returns: TWR total, TWR CAGR, and XIRR (money-weighted).
- Downloads: trade_ledger.csv, holdings.csv, daily_summary.csv, equity_series.csv.
- Optional: 15:00 IST Telegram message with the day’s signals.

Notes
- Universe: NIFTY 100 from NSE official CSV (auto-updates; Wikipedia fallback if NSE is down).
- Strategy: Balanced_B (MA20 z-score, regime MA60 +0.3%, TP 9%, time-stop 140d, 20D turnover filter, fee 0.11%, div 30/38, hard cap 4 sells/day).
- Backfill Option B: missed trading days are filled on next app run using official closes.

---

## 1) Requirements

- Python 3.10+
- A Google Cloud Service Account with Google Sheets API enabled
- A personal Google Sheet shared with your service account (Editor access)
- Optional: Telegram bot token and your chat_id

Files in this repo:
- balancedb_app.py — Streamlit app (main UI)
- telegram_daily.py — optional 3 pm Telegram notifier
- .github/workflows/telegram.yml — GitHub Actions schedule (optional)
- requirements.txt — dependencies

---

## 2) Setup — Streamlit Cloud (recommended)

1) Create a Google Service Account (with Sheets API)
- In Google Cloud Console: create a Service Account, enable Google Sheets API, and create a JSON key.

2) Add Secrets in Streamlit Cloud
- In your Streamlit app settings, add the following to Secrets (copy-paste and fill in from your JSON):

    [gcp_service_account]
    type = "service_account"
    project_id = "your-project-id"
    private_key_id = "your-private-key-id"
    private_key = "-----BEGIN PRIVATE KEY-----\nYOUR-KEY-CONTENT\n-----END PRIVATE KEY-----\n"
    client_email = "your-sa@your-project.iam.gserviceaccount.com"
    client_id = "your-client-id"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-sa%40your-project.iam.gserviceaccount.com"

Important:
- Keep the private_key exactly as shown, with literal \n newlines inside the quotes.
- Your JSON may have additional fields; include them if present.

3) Share your Google Sheet
- Create a new (empty) Google Sheet (the app auto-creates required tabs).
- Share the sheet with the client_email from your secrets (Editor access).

4) Deploy
- Deploy the repo in Streamlit Cloud with balancedb_app.py as the entry point.
- Open the app on mobile, paste your Sheet URL in the app, and you’re ready.

---

## 3) Setup — Local (optional)

1) Install dependencies

    pip install -r requirements.txt

2) Service account auth
- Save your service account JSON to a file, then:

    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json

3) Share your Google Sheet with your service account’s client_email.

4) Run the app

    streamlit run balancedb_app.py

---

## 4) Using the app

- Market-open gating: Run is enabled only during NSE market hours (Mon–Fri, 09:15–15:30 IST).
- Signals:
  - Regime status, lot cash, sells (TP/time-stop; max 4/day), new buys (ranked), averaging candidates.
  - You tick only what you executed; partial fills supported; only executed orders are saved.
- Funds:
  - Add/withdraw any amount anytime; takes effect immediately.
  - Logged as FUND_IN/FUND_OUT and adjusts base_capital and cash exactly once.
- Portfolio:
  - Metrics: Cash, Equity (last close), Exposure, Realized PnL, Unrealized PnL, Fees.
  - Returns: TWR total, CAGR (TWR), XIRR (money-weighted), Rolling 1Y CAGR.
  - Charts: Equity vs NIFTY100 TRI (fallback labeled), Exposure, Drawdown, Rolling 1Y CAGR.
  - Downloads: trade_ledger.csv, holdings.csv, daily_summary.csv, equity_series.csv.

Backfill (Option B)
- If you don’t open the app after close, the next run backfills the missed trading days using official closes and your ledger.

---

## 5) Optional — Telegram at 3:00 pm IST

1) Create a Telegram bot and get your token (via @BotFather).
2) Find your chat_id (use @userinfobot).
3) Configure GitHub Actions (free) to send daily messages:
   - Add these repo Secrets:
     - SHEET_URL — your Google Sheet URL
     - TELEGRAM_TOKEN — your bot token
     - TELEGRAM_CHAT_ID — your chat_id
     - GCP_SERVICE_ACCOUNT_JSON — paste the full JSON of your service account (as one JSON string)
   - The workflow .github/workflows/telegram.yml runs at 15:00 IST on weekdays and sends you the signal summary.

Run locally (optional):

    export SHEET_URL="https://docs.google.com/..../edit"
    export TELEGRAM_TOKEN="123456:ABC..."
    export TELEGRAM_CHAT_ID="123456789"
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
    python telegram_daily.py

---

## 6) Google Sheet structure (auto-created if missing)

- balances: cash, base_capital (default 500000), realized, fees_paid, last_update
- positions: symbol, shares, avg_cost, last_buy, open_date
- ledger: date, side [BUY/SELL/FUND_IN/FUND_OUT], symbol, shares, price, fee, reason, realized_pnl
- config: fee, divisor, divisor_bear, take_profit, time_stop_days, telegram_token, telegram_chat_id
- daily_equity: date, equity, cash, invested, exposure, source

You can edit balances/positions directly if you did something in your broker account.

---

## 7) Troubleshooting

- “Could not open Sheet”: Make sure you shared the Sheet with the service account client_email.
- “Service account credentials not found”: On Streamlit Cloud, add [gcp_service_account] in Secrets (see block above). Locally, set GOOGLE_APPLICATION_CREDENTIALS.
- Rate limits / missing quotes:
  - LTP fetch may sometimes fall back to Yahoo if NSE blocks; manual price entry is available in the app.
  - TRI may fall back to price index; we label the chart line accordingly.
- Sells cap: If you ran multiple times in a day, the app enforces max 4 executed sells across all runs based on the ledger.

---

## 8) Security

- Your Google Sheet is private; only those with access to the sheet and the service account can read/write.
- Do not share your service account key or bot token publicly.
- The Streamlit app itself requires no login; keep the app link private if you’re concerned about others pairing it with your sheet URL.

---

## 9) Disclaimer

Educational use only. Not investment advice. Market data from free sources (NSE/Yahoo) may be delayed or rate-limited.