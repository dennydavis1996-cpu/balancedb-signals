# ==================================================
# Balanced_B Signals — NIFTY100
# Streamlit App with Google Sheets Integration
# ==================================================

import streamlit as st         # For building interactive web UI
import pandas as pd            # For data manipulation
import numpy as np             # For numerical operations
import yfinance as yf          # For live/daily stock market data
import gspread                 # For Google Sheets API connection
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ----------------- PAGE SETUP -----------------
st.set_page_config(
    page_title="Balanced_B Signals — NIFTY100",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📈 Balanced_B Signals — NIFTY100")

# ----------------- DEFAULT CONFIG -----------------
DEFAULTS = dict(
    base_capital=500000,
    fee=0.0011,
    ma=20,
    bottom_n=16,
    max_new_buys=3,
    avg_dd=0.035,
    take_profit=0.09,
    max_sells_per_day=4,
    time_stop_days=140,
    regime_filter_ma=60,
    divisor=30,
    divisor_bear=38,
    lookback_days=420
)

NIFTY_INDEX = "^NSEI"    # Yahoo symbol for NIFTY 50
N100_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"

# ----------------- GOOGLE SHEETS HELPERS -----------------
def _service_account():
    """ Returns authenticated gspread client using Streamlit secrets (not file). """
    creds_dict = st.secrets["gcp_service_account"]  # put JSON content of credentials.json in Streamlit secrets
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    return gspread.authorize(creds)

def open_sheet(url):
    """Return gspread spreadsheet object given its URL."""
    client = _service_account()
    return client.open_by_url(url)

def ensure_tabs(sh):
    """Ensure required tabs exist and have correct headers."""
    schemas = {
        "balances": ["cash","base_capital","realized","fees_paid","last_update"],
        "positions": ["symbol","shares","avg_cost","last_buy","open_date"],
        "ledger": ["date","side","symbol","shares","price","fee","reason","realized_pnl"],
        "config": ["param","value"],
        "daily_equity": ["date","equity","cash","invested","exposure","source"],
    }
    for tab, cols in schemas.items():
        try:
            ws = sh.worksheet(tab)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=tab, rows=1000, cols=len(cols))
            ws.append_row(cols)

def read_tab(sh, tab):
    """Load a Google Sheet tab into a DataFrame."""
    ws = sh.worksheet(tab)
    df = pd.DataFrame(ws.get_all_records())
    return df

def save_df(sh, tab, df):
    """Overwrite a Google Sheet tab with a DataFrame."""
    ws = sh.worksheet(tab)
    ws.clear()
    ws.update([df.columns.values.tolist()] + df.values.tolist())

# ----------------- MARKET DATA HELPERS -----------------
@st.cache_data(ttl=300)
def safe_yf_download(symbols, start, end):
    """Download from Yahoo Finance safely with fallback empty frames."""
    try:
        data = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=False)
        return data
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance error: {e}")
        return pd.DataFrame()

def fetch_nifty100_symbols():
    """Fetch NIFTY100 symbols from NSE CSV."""
    try:
        df = pd.read_csv(N100_URL)
        symbols = sorted(df["Symbol"].dropna().astype(str).str.upper().tolist())
        return [s+".NS" for s in symbols]
    except Exception:
        st.error("Could not fetch NIFTY100 list. Falling back empty list.")
        return []

# ----------------- SIGNAL COMPUTATION -----------------
def compute_signals(prices, config, positions):
    """Compute BUY, SELL, AVERAGE signals."""
    ma = prices.rolling(config["ma"]).mean()
    std = prices.rolling(config["ma"]).std()
    last = prices.iloc[-1]
    last_ma = ma.iloc[-1]
    last_std = std.iloc[-1]

    signals = {"SELL":[],"NEW":[],"AVERAGE":[]}

    # SELL signals: take-profit
    for sym, pos in positions.iterrows():
        if sym not in last: continue
        price = last[sym]
        if price/pos["avg_cost"]-1 >= config["take_profit"]:
            signals["SELL"].append((sym, price, "TP"))

    # BUY candidates: below MA, sorted by z-score
    elig = [c for c in prices.columns if last[c]<last_ma[c] and not np.isnan(last_ma[c])]
    zmap = {c:(last[c]-last_ma[c])/last_std[c] for c in elig if last_std[c]>0}
    ranked = sorted(zmap, key=zmap.get)[:config["bottom_n"]]
    for sym in ranked:
        if sym not in positions.index:
            signals["NEW"].append((sym,last[sym],"NEW"))
        else:
            pos = positions.loc[sym]
            if last[sym] <= pos["last_buy"]*(1-config["avg_dd"]):
                signals["AVERAGE"].append((sym,last[sym],"AVG"))

    return signals

# ----------------- PORTFOLIO SNAPSHOT -----------------
def position_snapshot(prices, positions):
    """Return holdings dataframe with market value and unrealized PnL."""
    last = prices.iloc[-1]
    snaps=[]
    for sym,pos in positions.iterrows():
        if sym not in last: continue
        price = last[sym]
        mv = price*pos["shares"]
        unr = (price-pos["avg_cost"])*pos["shares"]
        unrpct = (price/pos["avg_cost"]-1)*100
        snaps.append([sym,pos["shares"],pos["avg_cost"],price,mv,unr,unrpct])
    if snaps:
        cols=["symbol","shares","avg_cost","last_price","market_value","unrealized_pnl","unrealized_pct"]
        return pd.DataFrame(snaps,columns=cols).set_index("symbol")
    else:
        return pd.DataFrame(columns=["symbol","shares","avg_cost","last_price","market_value","unrealized_pnl","unrealized_pct"])

# ----------------- STREAMLIT APP -----------------
with st.sidebar:
    st.header("⚙️ Settings")
    sheet_url = st.text_input("Google Sheet URL", value="")
    run_button = st.button("🔄 Run Scan")

if sheet_url:
    sh = open_sheet(sheet_url)
    ensure_tabs(sh)

    # Load base data
    balances = read_tab(sh,"balances")
    positions = read_tab(sh,"positions").set_index("symbol") if not read_tab(sh,"positions").empty else pd.DataFrame().set_index(pd.Index([]))
    config_df = read_tab(sh,"config")
    config = DEFAULTS.copy()
    for _,row in config_df.iterrows():
        if row["param"] in config: config[row["param"]] = float(row["value"])

    tickers = fetch_nifty100_symbols()
    end = datetime.today().date()
    start = end - timedelta(days=int(config["lookback_days"]))
    data = safe_yf_download(tickers, start, end)
    prices = data["Adj Close"] if not data.empty else pd.DataFrame()

    tabs = st.tabs(["📊 Run Signals", "💼 My Portfolio"])

    with tabs[0]:
        st.subheader("📊 Trade Signals")
        if run_button and not prices.empty:
            sigs = compute_signals(prices, config, positions)
            st.success("Signals generated successfully ✅")

            st.markdown("### 🔴 SELL Signals")
            st.write(pd.DataFrame(sigs["SELL"], columns=["symbol","price","reason"]))

            st.markdown("### 🟢 NEW BUY Signals")
            st.write(pd.DataFrame(sigs["NEW"], columns=["symbol","price","reason"]))

            st.markdown("### 🔵 AVERAGING Signals")
            st.write(pd.DataFrame(sigs["AVERAGE"], columns=["symbol","price","reason"]))

    with tabs[1]:
        st.subheader("💼 Portfolio Snapshot")
        snaps = position_snapshot(prices, positions) if not prices.empty else pd.DataFrame()
        if not snaps.empty:
            st.dataframe(snaps.style.bar(subset=["unrealized_pct"], align="mid", color=["red","green"]))
        else:
            st.info("No active positions found.")
