###########################
# balancedb_app.py - Part 1
# Imports, setup, Google Sheets integration
###########################

import streamlit as st                 # Streamlit for the web app UI
import pandas as pd                    # Pandas for dataframes
import numpy as np                     # Numpy for math
import yfinance as yf                  # Yahoo Finance for market data
import datetime as dt                  # Handling datetime
import matplotlib.pyplot as plt        # Charting
import gspread                         # Google Sheets API
from google.oauth2.service_account import Credentials   # Auth
from gspread_dataframe import set_with_dataframe        # Write dataframes
import requests, re, io                # For NSE/Wikipedia scraping, etc.

# ----------------- Streamlit Setup -----------------
st.set_page_config(
    page_title="Balanced_B Signals — NIFTY100",
    page_icon="📈",
    layout="wide",
)

st.title("Balanced_B Signals — NIFTY100 📊")
st.caption("Live portfolio manager, Google Sheets backend, full Balanced_B logic")

# ----------------- Constants / Defaults -----------------
DEFAULT_PARAMS = dict(
    base_capital=500_000,
    fee=0.0011,
    ma=20,
    bottom_n=16,
    max_new_buys=3,
    avg_dd=0.035,
    averaging_requires_regime=False,
    avg_in_bear_z_thresh=-1.8,
    take_profit=0.09,
    max_sells_per_day=4,
    time_stop_days=140,
    regime_filter_ma=60,
    regime_buffer=0.003,
    divisor=30,
    divisor_bear=38,
    min_turnover_cr=8.0,
    turnover_window=20,
    lookback_days=420,
)
# ----------------- Google Sheets Integration -----------------
def _service_account():
    """
    Authenticate with Google Sheets using service account stored in Streamlit secrets.
    """
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(creds)
    return client

@st.cache_resource(show_spinner=False)
def open_sheet(url):
    """
    Open Google Sheet by URL. Cached as a resource so we don’t re‑open on each run.
    """
    client = _service_account()
    return client.open_by_url(url)

def ensure_tabs(sh):
    """
    Ensure required tabs exist with correct schema.
    For 'balances', also create an initial data row so cash persists.
    """
    required = {
        "balances": ["cash", "base_capital", "realized", "fees_paid", "last_update"],
        "positions": ["symbol", "shares", "avg_cost", "last_buy", "open_date"],
        "ledger": ["date", "side", "symbol", "shares", "price", "fee", "reason",
                   "realized_pnl", "cash_before", "cash_after", "holding_days"],
        "config": list(DEFAULT_PARAMS.keys()),
        "daily_equity": ["date", "equity", "cash", "invested", "exposure", "source"],
    }
    existing_titles = [ws.title for ws in sh.worksheets()]
    for tab, cols in required.items():
        if tab not in existing_titles:
            ws = sh.add_worksheet(title=tab, rows="5000", cols=str(len(cols)))
            ws.append_row(cols)
            if tab == "balances":
                # ✅ Add one initial data row: starting cash = base capital
                ws.append_row([
                    DEFAULT_PARAMS["base_capital"],  # cash
                    DEFAULT_PARAMS["base_capital"],  # base_capital (seed)
                    0,                               # realized
                    0,                               # fees_paid
                    dt.date.today().strftime("%Y-%m-%d"),  # last_update
                ])
            st.info(f"Created missing tab: {tab}")

def save_df(sh, tab, df):
    """
    Save DataFrame to a specific tab, overwriting existing content.
    - For balances: keep exactly ONE row (latest state).
    - For others: write entire DataFrame.
    """
    ws = sh.worksheet(tab)
    ws.clear()

    # Always reset index before saving
    df_reset = df.reset_index(drop=True)

    if tab == "balances":
        # ✅ Only write the last row (one row of truth)
        if not df_reset.empty:
            last_row = df_reset.tail(1).reset_index(drop=True)
            set_with_dataframe(ws, last_row)
    else:
        set_with_dataframe(ws, df_reset)

@st.cache_data(ttl=60)
def load_tab(sheet_url, tab):
    """
    Load a specific tab from Google Sheet and sanitize datatypes.
    - balances: always return numeric cash/realized/fees/base_capital
    - positions: numeric shares/avg_cost/last_buy
    - ledger: numeric columns + dates
    - daily_equity: numeric values + dates
    """
    sh = open_sheet(sheet_url)
    ws = sh.worksheet(tab)
    values = ws.get_all_values()

    if not values or len(values) <= 1:
        return pd.DataFrame(columns=values[0] if values else [])

    df = pd.DataFrame(values[1:], columns=values[0])

    if tab == "balances" and not df.empty:
        for col in ["cash","base_capital","realized","fees_paid"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        # Only keep the *latest row* (should be exactly 1 row after save)
        df = df.tail(1).reset_index(drop=True)

    elif tab == "positions" and not df.empty:
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
        df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce").fillna(0.0)
        if "last_buy" in df.columns:
            df["last_buy"] = pd.to_numeric(df["last_buy"], errors="coerce").fillna(0.0)

    elif tab == "ledger" and not df.empty:
        if "shares" in df.columns:
            df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
        for col in ["price","fee","realized_pnl","cash_before","cash_after","holding_days"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    elif tab == "daily_equity" and not df.empty:
        for col in ["equity","cash","invested","exposure"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    return df

@st.cache_data(ttl=60)
def load_all(sheet_url):
    return {
        "balances": load_tab(sheet_url, "balances"),
        "positions": load_tab(sheet_url, "positions"),
        "ledger": load_tab(sheet_url, "ledger"),
        "config": load_tab(sheet_url, "config"),
        "daily_equity": load_tab(sheet_url, "daily_equity"),
    }

# ----------------- Sidebar: Portfolio Selector (with persistence) -----------------
# If no query param is set, default to Wife Portfolio
if "portfolio" not in st.query_params:
    st.query_params["portfolio"] = "Wife Portfolio"   # ✅ set wife as default

# Sidebar radio
portfolio_choice = st.sidebar.radio(
    "Choose portfolio:",
    ["My Portfolio", "Wife Portfolio"],
    index=0 if st.query_params["portfolio"] == "My Portfolio" else 1,
    key="portfolio_choice"
)

# Update query param to persist choice
st.query_params["portfolio"] = portfolio_choice

# Select corresponding Google Sheet
if portfolio_choice == "My Portfolio":
    SHEET_URL = st.secrets["my_sheet_url"]
else:
    SHEET_URL = st.secrets["wife_sheet_url"]

SHEET = open_sheet(SHEET_URL)
ensure_tabs(SHEET)
TABS = load_all(SHEET_URL)

###########################
# balancedb_app.py - Part 2
# Market data functions
###########################
def get_live_prices(tickers):
    """
    Fetch approx live prices (15 min delayed for NSE) from Yahoo Finance.
    Returns a Pandas Series {symbol: last_price}
    """
    prices = {}
    for sym in tickers:
        try:
            px = yf.Ticker(sym).fast_info["last_price"]
            if px and px > 0:
                prices[sym] = px
        except Exception:
            continue
    return pd.Series(prices, dtype=float)
# Helper to normalize symbols (remove spaces/punct)
def _clean_symbol_keep_punct(s):
    return re.sub(r'[^A-Za-z0-9\-\&\.]+', '', str(s)).upper()

def _norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

# --------------------------------------------------
# Fetch NIFTY constituent list
# --------------------------------------------------

def fetch_index_current(index_name="NIFTY100"):
    """
    Fetch current index constituents.
    Priority: NSE official CSV -> fallback Wikipedia table.
    Returns: set(symbols), dict company_name->symbol
    """
    idx = index_name.upper().replace(" ", "")
    file_map = {
        "NIFTY50": "ind_nifty50list.csv",
        "NIFTY100": "ind_nifty100list.csv",
        "NIFTY200": "ind_nifty200list.csv",
    }
    # Try NSE CSV
    if idx in file_map:
        url = f"https://archives.nseindia.com/content/indices/{file_map[idx]}"
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
            sym_col = [c for c in df.columns if re.search(r"(symbol|ticker|nse)", c, re.I)][0]
            name_cols = [c for c in df.columns if re.search(r"(company|name)", c, re.I)]
            name_col = name_cols[0] if name_cols else None
            df[sym_col] = df[sym_col].apply(_clean_symbol_keep_punct)
            return set(df[sym_col].dropna().astype(str)), (
                {_norm(n): s for n,s in zip(df[name_col], df[sym_col])} if name_col else {}
            )
        except Exception as e:
            st.warning(f"⚠️ NSE fetch failed ({e}), falling back to Wikipedia.")

    url_map = {
        "NIFTY50":"https://en.wikipedia.org/wiki/NIFTY_50",
        "NIFTY100":"https://en.wikipedia.org/wiki/NIFTY_100",
        "NIFTY200":"https://en.wikipedia.org/wiki/NIFTY_200",
    }
    url = url_map.get(idx, url_map["NIFTY100"])
    try:
        tables = pd.read_html(url)
        for df in tables:
            cols = [c.lower() for c in df.columns]
            if any("symbol" in c or "ticker" in c or "nse" in c for c in cols) and \
               any("company" in c or "name" in c for c in cols) and len(df) >= 40:
                sym_col = [c for c in df.columns if re.search(r"(symbol|ticker|nse)", c, re.I)][0]
                name_col = [c for c in df.columns if re.search(r"(company|name)", c, re.I)][0]
                df[sym_col] = df[sym_col].apply(_clean_symbol_keep_punct)
                return set(df[sym_col].tolist()), {_norm(n): s for n,s in zip(df[name_col], df[sym_col])}
    except Exception as e:
        st.error(f"❌ Failed fetching symbols from Wikipedia too: {e}")
        st.stop()
    return set(), {}

# --------------------------------------------------
# Safe Yahoo Finance download
# --------------------------------------------------

def safe_yf_download(tickers, start, end, fields=("Adj Close","Volume"), retry=2):
    """
    Download data safely from Yahoo Finance for given tickers.
    Handles multi-index, retries, errors.
    Returns dict(field->DataFrame)
    """
    tickers = list(dict.fromkeys(tickers))
    out = {f: [] for f in fields}

    for attempt in range(retry):
        try:
            df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                for f in fields:
                    sub = df.get(f)
                    if sub is None: continue
                    if isinstance(sub, pd.Series): sub = sub.to_frame()
                    out[f].append(sub)
            else:
                for f in fields:
                    sub = df.get(f)
                    if sub is None: continue
                    if isinstance(sub, pd.Series): sub = sub.to_frame()
                    out[f].append(sub)
            break
        except Exception as e:
            st.warning(f"⚠️ Yahoo download failed (attempt {attempt+1}): {e}")
    # Concatenate collected data
    for f in fields:
        if out[f]:
            joined = pd.concat(out[f], axis=1)
            joined = joined.loc[:, ~joined.columns.duplicated()].sort_index()
            out[f] = joined
        else:
            out[f] = pd.DataFrame()
    return out

# --------------------------------------------------
# Load Market Data (benchmark + NIFTY100 prices/volumes/turnover)
# --------------------------------------------------

@st.cache_data(show_spinner=False, ttl=60*30)
def load_market_data(start=None, end=None):
    """
    Load NIFTY100 price/volume and benchmark (^NSEI).
    Computes 20D rolling turnover median for liquidity filter.
    Cached for 30 minutes.
    """
    end = end or dt.date.today().strftime("%Y-%m-%d")
    start = start or (dt.date.today() - dt.timedelta(days=DEFAULT_PARAMS["lookback_days"])).strftime("%Y-%m-%d")

    # Get symbols
    syms, _ = fetch_index_current("NIFTY100")
    tickers = sorted(s + ".NS" for s in syms)

    # Benchmark (NIFTY 50 index proxy = ^NSEI)
    bench = safe_yf_download(["^NSEI"], start, end, fields=("Adj Close",))["Adj Close"].squeeze()
    bench = bench.rename("NIFTY50").dropna()

    # Prices + Volumes
    raw = safe_yf_download(tickers, start, end, fields=("Adj Close","Volume"))
    prices = raw["Adj Close"].reindex(bench.index)
    vols   = raw["Volume"].reindex(bench.index)

    # Turnover (₹ Cr) = Price*Volume / 1e7
    turnover = (prices * vols) / 1e7
    med_turnover = turnover.rolling(DEFAULT_PARAMS["turnover_window"], min_periods=DEFAULT_PARAMS["turnover_window"]).median()

    return dict(bench=bench, prices=prices, vols=vols, med_turnover=med_turnover, tickers=tickers)
###########################
# balancedb_app.py - Part 3
# Portfolio logic (positions, ledger, performance)
###########################

# --------------------------------------------------
# Helper: today’s date string
# --------------------------------------------------
def today_str():
    return dt.date.today().strftime("%Y-%m-%d")

# --------------------------------------------------
# Ledger + balances update (apply trades)
# --------------------------------------------------
def apply_trade_rows(sh, trades, balances_df, positions_df, ledger_df):
    """
    Apply trades (BUY/SELL/FUND_IN/FUND_OUT) and update balances,
    positions, and ledger in Google Sheets. Then clear cache so reads refresh.
    """

    # --- load last saved balances ---
    if not balances_df.empty:
        base_cap = float(balances_df.iloc[0]["base_capital"])
        cash = float(balances_df.iloc[0]["cash"])
        realized = float(balances_df.iloc[0]["realized"])
        fees_paid = float(balances_df.iloc[0]["fees_paid"])
    else:
        base_cap = DEFAULT_PARAMS["base_capital"]
        cash = base_cap
        realized = 0.0
        fees_paid = 0.0

    st.write("DEBUG Starting Balances -> Cash:", cash,
             "Base_capital:", base_cap,
             "Realized:", realized,
             "Fees:", fees_paid)

    for tr in trades:
        sym, side = tr["symbol"], tr["side"]
        shares, price, fee = tr["shares"], tr["price"], tr["fee"]
        reason = tr.get("reason", "")
        date = tr.get("date", today_str())
        cash_before = cash

        if side == "BUY":
            cost = shares * price + shares * price * fee
            if cash >= cost:
                cash -= cost
                fees_paid += shares * price * fee
                if sym in positions_df["symbol"].values:
                    pos = positions_df.loc[positions_df["symbol"] == sym].iloc[0]
                    old_shares = int(pos["shares"])
                    avg_cost = float(pos["avg_cost"])
                    tot_cost = old_shares * avg_cost + shares * price
                    new_shares = old_shares + shares
                    new_avg = tot_cost / new_shares
                    positions_df.loc[positions_df["symbol"] == sym,
                                     ["shares","avg_cost","last_buy","open_date"]] = [
                        new_shares, new_avg, price, date
                    ]
                else:
                    new_row = dict(symbol=sym, shares=shares, avg_cost=price,
                                   last_buy=price, open_date=date)
                    positions_df = pd.concat([positions_df, pd.DataFrame([new_row])],
                                             ignore_index=True)
            else:
                st.warning(f"❌ Not enough cash for BUY {sym} {shares}@{price}")

        elif side == "SELL":
            pnl = 0.0
            holding_days = 0
            if sym in positions_df["symbol"].values:
                pos = positions_df.loc[positions_df["symbol"] == sym].iloc[0]
                held_shares = int(pos["shares"])
                avg_cost = float(pos["avg_cost"])
                if shares > held_shares:
                    shares = held_shares
                proceeds_gross = shares * price
                fee_amt = proceeds_gross * fee
                proceeds_net = proceeds_gross - fee_amt
                pnl = proceeds_net - shares * avg_cost

                cash += proceeds_net
                realized += pnl
                fees_paid += fee_amt

                st.write("DEBUG SELL -> Sym:", sym,
                         "Gross:", proceeds_gross,
                         "Net:", proceeds_net,
                         "PnL:", pnl,
                         "Cash after sell:", cash,
                         "Realized total:", realized)

                if shares == held_shares:
                    positions_df = positions_df[positions_df["symbol"] != sym]
                else:
                    positions_df.loc[positions_df["symbol"] == sym,"shares"] = held_shares - shares

                holding_days = (dt.date.fromisoformat(date) -
                                dt.date.fromisoformat(pos["open_date"])).days
            else:
                st.warning(f"❌ No shares to sell for {sym}")

        elif side == "FUND_IN":
            cash += tr["amount"]

        elif side == "FUND_OUT":
            cash -= tr["amount"]

        # Append trade to ledger
        ledger_row = dict(date=date, side=side, symbol=sym, shares=shares, price=price,
                          fee=fee, reason=reason, realized_pnl=locals().get("pnl",0),
                          cash_before=cash_before, cash_after=cash,
                          holding_days=locals().get("holding_days",0))
        ledger_df = pd.concat([ledger_df, pd.DataFrame([ledger_row])], ignore_index=True)

    # --- Save updated balances state ---
    balances_new = pd.DataFrame([{
        "cash": round(cash,2),
        "base_capital": base_cap,
        "realized": round(realized,2),
        "fees_paid": round(fees_paid,2),
        "last_update": today_str(),
    }])

    st.write("DEBUG Writing Balances ->", balances_new)

    save_df(sh, "balances", balances_new)
    save_df(sh, "positions", positions_df)
    save_df(sh, "ledger", ledger_df)

    st.cache_data.clear()

    return balances_new, positions_df, ledger_df

# --------------------------------------------------
# Valuations: positions with unrealized PnL
# --------------------------------------------------
def position_snapshot(positions_df, live_prices=None, fallback_prices=None):
    """
    Compute market value & unrealized PnL of positions.
    - live_prices: pd.Series with {symbol: live price}
    - fallback_prices: daily close row (Series) for fallback
    """
    if positions_df.empty:
        return pd.DataFrame()

    positions_df = positions_df.copy()
    positions_df["shares"] = pd.to_numeric(positions_df["shares"], errors="coerce").fillna(0).astype(int)
    positions_df["avg_cost"] = pd.to_numeric(positions_df["avg_cost"], errors="coerce").fillna(0.0)
    if "last_buy" in positions_df.columns:
        positions_df["last_buy"] = pd.to_numeric(positions_df["last_buy"], errors="coerce").fillna(0.0)

    holdings = []
    for _, pos in positions_df.iterrows():
        sym = pos["symbol"]

        # Prefer live price, fallback to last daily close
        price = None
        if live_prices is not None and sym in live_prices.index:
            price = live_prices.get(sym, np.nan)
        if (price is None or pd.isna(price)) and fallback_prices is not None:
            price = fallback_prices.get(sym, np.nan)

        if pd.notna(price) and pos["shares"] > 0:
            mv = pos["shares"] * price
            unr = (price - pos["avg_cost"]) * pos["shares"]
            unr_pct = (price / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] > 0 else np.nan

            holdings.append(dict(
                symbol=sym,
                shares=int(pos["shares"]),
                avg_cost=round(pos["avg_cost"], 2),
                last_price=round(price, 2),
                market_value=round(mv, 2),
                unrealized_pnl=round(unr, 2),
                unrealized_pct=round(unr_pct, 2),
                last_buy=pos["last_buy"],
                open_date=pos["open_date"],
            ))

    return pd.DataFrame(holdings).sort_values("market_value", ascending=False)

# --------------------------------------------------
# Performance metrics
# --------------------------------------------------
def compute_drawdown(series):
    cummax = series.cummax()
    dd = series/cummax - 1
    return dd

def compute_cagr(start_val, end_val, days):
    years = days/365.25
    return (end_val/start_val)**(1/years)-1 if years>0 and start_val>0 else np.nan

def compute_sharpe(returns):
    if len(returns)<2: return np.nan
    ann_vol = returns.std()*np.sqrt(252)
    return (returns.mean()*252)/ann_vol if ann_vol>0 else np.nan

def compute_xirr(cashflows):
    """
    Approximate XIRR (money-weighted return) using numpy. cashflows=[(date, amt)].
    """
    if not cashflows: return np.nan
    days = np.array([(dt.datetime.strptime(str(d),"%Y-%m-%d")-dt.datetime.strptime(str(cashflows[0][0]),"%Y-%m-%d")).days for d,_ in cashflows])
    amts = np.array([a for _,a in cashflows])
    def npv(rate):
        return np.sum(amts/((1+rate)**(days/365)))
    try:
        return np.irr(amts)  # quick hack, IRR not exact XIRR
    except Exception:
        return np.nan
###########################
# balancedb_app.py - Part 4
# Signal generation (Balanced_B full rules)
###########################

def compute_signals(market, positions_df, balances_df, params):
    """
    Compute Balanced_B signals with corrected lot sizing:
    - Lot size = (cash available + realized PnL) / divisor
    - Ensures Bull/Bear divisor logic
    - Returns useful diagnostic fields
    """

    bench = market["bench"]
    prices = market["prices"]
    vols = market["vols"]
    med_turnover = market["med_turnover"]
    today = bench.index[-1]
    # Instead of using stale EOD close, fetch live price snapshot from Yahoo
    today_prices = get_live_prices(market["tickers"])

    # Fallback: if no live data came (empty Series), use yesterday close
    if today_prices.empty:
        today_prices = prices.loc[today]
    today_turnover = med_turnover.loc[today]

    # --- Current balances ---
    cash = float(balances_df.iloc[0]["cash"]) if not balances_df.empty else params["base_capital"]
    realized = float(balances_df.iloc[0].get("realized", 0)) if not balances_df.empty else 0.0

    # Positions
    if not positions_df.empty:
        positions_df = positions_df.copy()
        positions_df["shares"] = pd.to_numeric(positions_df["shares"], errors="coerce").fillna(0).astype(int)
        positions_df["avg_cost"] = pd.to_numeric(positions_df["avg_cost"], errors="coerce").fillna(0.0)
        if "last_buy" in positions_df.columns:
            positions_df["last_buy"] = pd.to_numeric(positions_df["last_buy"], errors="coerce").fillna(0.0)

    # Invested (mark-to-market of current holdings, just for equity calc)
    if not positions_df.empty:
        merged = positions_df.merge(today_prices.to_frame("price"),
                                    left_on="symbol", right_index=True, how="left")
        invested_val = (merged["price"] * merged["shares"]).sum()
    else:
        invested_val = 0.0

    portfolio_val = cash + invested_val

    # --- Regime detection ---
    bench_ma = bench.rolling(params["regime_filter_ma"],
                             min_periods=params["regime_filter_ma"]).mean()
    regime_ok = bool(bench.iloc[-1] >= bench_ma.iloc[-1] * (1 + params["regime_buffer"]))
    div_today = params["divisor"] if regime_ok else params["divisor_bear"]

    # --- ✅ Lot sizing: cash available + realized profit ---
    size_capital = cash + realized
    lot_cash = size_capital / div_today

    # --- Moving avg and std for z-score ---
    ma = prices.rolling(params["ma"], min_periods=params["ma"]).mean()
    std = prices.rolling(params["ma"], min_periods=params["ma"]).std()
    ma_today = ma.loc[today]; std_today = std.loc[today]

    # ---------------- SELL signals ----------------
    sell_signals = []
    for _, pos in positions_df.iterrows():
        sym = pos["symbol"]
        shares = int(pos["shares"])
        buy_price = float(pos["avg_cost"])
        px = today_prices.get(sym, np.nan)
        if pd.isna(px) or shares <= 0:
            continue
        ret = px / buy_price - 1

        # Take profit
        if ret >= params["take_profit"]:
            sell_signals.append(dict(symbol=sym, reason="TP",
                                     shares=shares, price=px,
                                     gain_pct=round(ret * 100, 2)))
        # Time stop
        age = (today - pd.to_datetime(pos["open_date"])).days
        if age >= params["time_stop_days"]:
            sell_signals.append(dict(symbol=sym, reason="TIME_STOP",
                                     shares=shares, price=px,
                                     gain_pct=round(ret * 100, 2)))

    sells_df = (pd.DataFrame(sell_signals)
                if sell_signals else
                pd.DataFrame(columns=["symbol", "reason", "shares", "price", "gain_pct"]))
    if len(sells_df) > params["max_sells_per_day"]:
        sells_df = sells_df.sort_values("gain_pct", ascending=False).head(params["max_sells_per_day"])

    # ---------------- BUY eligibility ----------------
    elig = [c for c in prices.columns
            if pd.notna(ma_today.get(c)) and pd.notna(today_prices.get(c))
            and today_prices[c] < ma_today[c]
            and today_turnover.get(c, 0) >= params["min_turnover_cr"]]

    if params.get("use_zscore", True):
        zmap = {c: (today_prices[c] - ma_today[c]) / std_today[c]
                for c in elig if pd.notna(std_today.get(c)) and std_today[c] > 0}
        ranked = sorted(zmap, key=zmap.get)[:params["bottom_n"]]
    else:
        dist = today_prices / ma_today - 1
        ranked = sorted(elig, key=lambda c: dist[c])[:params["bottom_n"]] if elig else []

    # ---------------- NEW BUY signals ----------------
    new_buys = []
    for sym in ranked:
        if sym in positions_df["symbol"].values:
            continue
        px = today_prices.get(sym, np.nan)
        if pd.notna(px):
            shares = int(lot_cash // (px * (1 + params["fee"])))
            if shares > 0:
                new_buys.append(dict(symbol=sym, price=px, shares=shares, reason="NEW"))

    if len(new_buys) > params["max_new_buys"]:
        new_buys = new_buys[:params["max_new_buys"]]

    new_buys_df = (pd.DataFrame(new_buys)
                   if new_buys else
                   pd.DataFrame(columns=["symbol", "price", "shares", "reason"]))

    # ---------------- AVERAGING signals ----------------
    avgs = []
    if regime_ok:  # Bull regime
        for _, pos in positions_df.iterrows():
            sym = pos["symbol"]; last_buy = pos["last_buy"]
            px = today_prices.get(sym, np.nan)
            if pd.notna(px) and px <= last_buy * (1 - params["avg_dd"]):
                if today_turnover.get(sym, 0) >= params["min_turnover_cr"]:
                    shares = int(lot_cash // (px * (1 + params["fee"])))
                    if shares > 0:
                        avgs.append(dict(symbol=sym, price=px, shares=shares, reason="AVERAGE"))
    else:  # Bear regime
        for _, pos in positions_df.iterrows():
            sym = pos["symbol"]; last_buy = pos["last_buy"]
            px = today_prices.get(sym, np.nan)
            if pd.isna(px): continue
            m, s = ma_today.get(sym), std_today.get(sym)
            if pd.isna(m) or pd.isna(s) or s <= 0: continue
            z = (px - m) / s
            if z <= params["avg_in_bear_z_thresh"] and px <= last_buy * (1 - params["avg_dd"]):
                if today_turnover.get(sym, 0) >= params["min_turnover_cr"]:
                    shares = int(lot_cash // (px * (1 + params["fee"])))
                    if shares > 0:
                        avgs.append(dict(symbol=sym, price=px, shares=shares, reason="AVERAGE"))

    avgs_df = (pd.DataFrame(avgs)
               if avgs else
               pd.DataFrame(columns=["symbol", "price", "shares", "reason"]))

    return dict(
        sells=sells_df,
        new_buys=new_buys_df,
        averaging=avgs_df,
        regime="Bull" if regime_ok else "Bear",
        lot_cash=lot_cash,
        size_capital=size_capital,   # ✅ show cash+realized used for lot sizing
        cash=cash,
        realized=realized,
        portfolio_val=portfolio_val
    )

###########################
# balancedb_app.py - Part 5
# Streamlit UI
###########################

# Load config parameters (from sheet or defaults)
def load_params(sheet_url):
    cfg = load_tab(sheet_url, "config")
    if cfg.empty: 
        return DEFAULT_PARAMS.copy()
    params = DEFAULT_PARAMS.copy()
    for k in params:
        if k in cfg.columns and not pd.isna(cfg.iloc[0][k]):
            params[k] = float(cfg.iloc[0][k]) if re.search(r'[0-9]', str(cfg.iloc[0][k])) else cfg.iloc[0][k]
    return params

# ---------------------- UI -------------------------
tab1, tab2, tab3 = st.tabs(["⚡ Run Signals", "📂 My Portfolio", "📑 Reports & Analytics"])

# ============ TAB 1: Run Signals ===================
with tab1:
    st.subheader("Run Balanced_B signal scan")

    # 🔄 Always reload live balances and positions directly from Google Sheet
    balances_df = load_tab(SHEET_URL, "balances")
    positions_df = load_tab(SHEET_URL, "positions")
    ledger_df    = load_tab(SHEET_URL, "ledger")

    params = load_params(SHEET_URL)
    market = load_market_data()

    # --- Run scan button ---
    if st.button("🔍 Run scan now"):
        sigs = compute_signals(market, positions_df, balances_df, params)
        st.session_state["signals"] = sigs   # ✅ persist scan results
        st.success("Signal scan completed!")

    # --- Show signals if available ---
    if "signals" in st.session_state:
        sigs = st.session_state["signals"]

        # 🟢 Enhanced info: show cash, regime, divisor, lot_cash
        regime = sigs['regime']
        div = params["divisor"] if regime=="Bull" else params["divisor_bear"]
        st.info(
            f"Market Regime: **{sigs['regime']}** | "
            f"Cash: ₹{sigs.get('cash',0):,.0f} | "
            f"Realized PnL: ₹{sigs.get('realized',0):,.0f} | "
            f"Size Capital (cash+realized): ₹{sigs.get('size_capital', sigs.get('cash',0)+sigs.get('realized',0)):,.0f} | "
            f"Divisor used: {params['divisor'] if sigs['regime']=='Bull' else params['divisor_bear']} | "
            f"Lot cash per stock: ₹{sigs.get('lot_cash',0):,.0f}"
        )
        
        # --------------- SELL signals ----------------
        st.markdown("### 🚪 SELL signals")
        selected_sells = []
        if sigs["sells"].empty:
            st.write("No SELL signals today.")
        else:
            for i, row in sigs["sells"].iterrows():
                sym, shares, sugg_price, reason, gain = (
                    row["symbol"], int(row["shares"]), float(row["price"]), row["reason"], row["gain_pct"]
                )
                tick = st.checkbox(f"SELL {sym} (Suggested: ₹{sugg_price:.2f}, {reason})", key=f"sell_{i}")
                if tick:
                    exec_price = st.number_input(
                        f"Enter execution SELL price for {sym}",
                        min_value=0.0, value=sugg_price, step=0.1, key=f"sell_price_{i}"
                    )
                    st.write(f"Shares: {shares}, Gain: {gain}%")
                    trade = dict(date=today_str(), side="SELL", symbol=sym,
                                 shares=shares, price=exec_price,
                                 fee=params["fee"], reason=reason)
                    selected_sells.append(trade)

        st.markdown("---")

        # --------------- NEW BUY signals ---------------
        st.markdown("### 🆕 New BUY signals")
        selected_buys = []
        if sigs["new_buys"].empty:
            st.write("No new BUY signals today.")
        else:
            for i, row in sigs["new_buys"].iterrows():
                sym, shares, sugg_price = row["symbol"], int(row["shares"]), float(row["price"])
                tick = st.checkbox(f"BUY {sym} (Suggested: ₹{sugg_price:.2f}, NEW)", key=f"buy_{i}")
                if tick:
                    exec_price = st.number_input(
                        f"Enter execution BUY price for {sym}",
                        min_value=0.0, value=sugg_price, step=0.1, key=f"buy_price_{i}"
                    )
                    st.write(f"Shares: {shares}")
                    trade = dict(date=today_str(), side="BUY", symbol=sym,
                                 shares=shares, price=exec_price,
                                 fee=params["fee"], reason="NEW")
                    selected_buys.append(trade)

        st.markdown("---")

        # --------------- AVERAGING signals ---------------
        st.markdown("### ➕ Averaging signals")
        selected_avgs = []
        if sigs["averaging"].empty:
            st.write("No averaging signals today.")
        else:
            for i, row in sigs["averaging"].iterrows():
                sym, shares, sugg_price = row["symbol"], int(row["shares"]), float(row["price"])
                tick = st.checkbox(f"AVERAGE {sym} (Suggested: ₹{sugg_price:.2f})", key=f"avg_{i}")
                if tick:
                    exec_price = st.number_input(
                        f"Enter execution AVG price for {sym}",
                        min_value=0.0, value=sugg_price, step=0.1, key=f"avg_price_{i}"
                    )
                    st.write(f"Shares: {shares}")
                    trade = dict(date=today_str(), side="BUY", symbol=sym,
                                 shares=shares, price=exec_price,
                                 fee=params["fee"], reason="AVERAGE")
                    selected_avgs.append(trade)

        # --------------- Confirm Selected Trades ---------------
        if st.button("✅ Confirm Selected Trades"):
            all_trades = selected_sells + selected_buys + selected_avgs
            if all_trades:
                new_bal, new_pos, new_ledger = apply_trade_rows(SHEET, all_trades, balances_df, positions_df, ledger_df)
                st.success(f"Recorded {len(all_trades)} trades.")
                st.cache_data.clear()
                # Clear signals so you don’t accidentally reuse stale data
                del st.session_state["signals"]
            else:
                st.warning("No trades selected.")

    else:
        st.info("Click 'Run scan' to generate signals.")

    # =========== Funds Management Section ===========
    st.markdown("## 💰 Funds Management")
    with st.form("funds_form"):
        action = st.selectbox("Action", ["FUND_IN", "FUND_OUT"])
        amount = st.number_input("Amount (₹)", min_value=1000, step=500)
        submit_f = st.form_submit_button("Apply")
        if submit_f:
            trade = dict(date=today_str(), side=action, symbol="", shares=0,
                         price=0, fee=0, reason="Funds", amount=amount)
            new_bal, new_pos, new_ledger = apply_trade_rows(SHEET, [trade], balances_df, positions_df, ledger_df)
            st.success(f"Funds {action} of ₹{amount} recorded.")
            st.cache_data.clear()

# ============ TAB 2: My Portfolio ==================
with tab2:
    st.subheader("Portfolio Snapshot")

    balances_df = load_tab(SHEET_URL, "balances")
    positions_df = load_tab(SHEET_URL, "positions")
    ledger_df = load_tab(SHEET_URL, "ledger")

    if balances_df.empty:
        st.warning("⚠️ No balances yet. Record a trade or fund injection first.")
    else:
        # Extract balances
        cash = float(balances_df.iloc[0]["cash"])+float(balances_df.iloc[0].get("realized", 0))
        base_cap = float(balances_df.iloc[0]["base_capital"])
        realized = float(balances_df.iloc[0].get("realized", 0))
        fees_paid = float(balances_df.iloc[0].get("fees_paid", 0))

        # Current holdings
        market = load_market_data()
        lastrow = market["prices"].iloc[-1]

        # 🔄 Fetch live prices
        live_prices = get_live_prices(market["tickers"])

        # 🔄 Compose holdings with live front, fallback daily
        holdings = position_snapshot(positions_df,
                             live_prices=live_prices,
                             fallback_prices=lastrow)
        invested = holdings["market_value"].sum() if not holdings.empty else 0.0

        # Total portfolio
        equity_val = cash + invested

        # Display balances summary nicely
        st.markdown("### 💵 Balances & PnL Summary")

        # Extract balances safely
        cash = float(balances_df.iloc[0]["cash"])
        base_cap = float(balances_df.iloc[0]["base_capital"])
        realized = float(balances_df.iloc[0].get("realized", 0))
        fees_paid = float(balances_df.iloc[0].get("fees_paid", 0))

        # Include size_capital = cash + realized
        size_capital = cash + realized

        # Current holdings market value
        invested = holdings["market_value"].sum() if not holdings.empty else 0.0
        unrealized = holdings["unrealized_pnl"].sum() if not holdings.empty else 0.0
        equity_val = cash + invested   # total account equity

        # Show metrics in two rows for clarity
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Base Capital", f"₹{base_cap:,.0f}")
        col2.metric("Cash (current)", f"₹{cash:,.0f}")
        col3.metric("Realized PnL (cumulative)", f"₹{realized:,.0f}")
        col4.metric("Size Capital (Cash+Realized)", f"₹{size_capital:,.0f}")
        col5.metric("Fees Paid", f"₹{fees_paid:,.0f}")

        col6, col7, col8 = st.columns(3)
        col6.metric("Invested", f"₹{invested:,.0f}")
        col7.metric("Unrealized PnL", f"₹{unrealized:,.0f}")
        col8.metric("Equity (Cash + Invested)", f"₹{equity_val:,.0f}")

        st.caption("📌 Lot sizing in signals will be based on **Size Capital (Cash + Realized)** ÷ Divisor.\n"
           "Equity = Cash + Invested. Realized PnL is booked profits, Unrealized PnL is floating.")
        
        # Positions (with unrealized PnL)
        st.markdown("### 📂 Current Holdings")
        if holdings.empty:
            st.write("No open positions.")
        else:
            st.dataframe(holdings)

        # ==== Equity Curve (if logged daily) ====
        df_daily = load_tab(SHEET_URL, "daily_equity")
        if not df_daily.empty:
            st.markdown("### 📈 Equity Curve (Daily)")
            st.line_chart(df_daily.set_index("date")[["equity","cash","invested"]])

            # Performance metrics
            eq = pd.to_numeric(df_daily["equity"])
            eq.index = pd.to_datetime(df_daily["date"])
            start_val, end_val = eq.iloc[0], eq.iloc[-1]
            cagr = compute_cagr(start_val, end_val, (eq.index[-1]-eq.index[0]).days)
            rets = eq.pct_change().dropna()
            sharpe = compute_sharpe(rets)
            dd = compute_drawdown(eq).min()

            st.markdown("### 📊 Performance Metrics")
            colA, colB, colC = st.columns(3)
            colA.metric("CAGR", f"{cagr*100:.2f}%")
            colB.metric("Sharpe", f"{sharpe:.2f}")
            colC.metric("Max Drawdown", f"{dd*100:.2f}%")

        # ==== Downloads ====
        st.markdown("---")
        st.write("📥 Download Data")
        st.download_button("Ledger CSV", ledger_df.to_csv(index=False), "ledger.csv")
        st.download_button("Positions CSV", positions_df.to_csv(index=False), "positions.csv")
        if not df_daily.empty:
            st.download_button("Daily Equity CSV", df_daily.to_csv(index=False), "daily_equity.csv")

# ============ TAB 3: Reports & Analytics ===========
with tab3:
    st.subheader("Reports & Analytics")

    df_daily = load_tab(SHEET_URL,"daily_equity")
    ledger_df = load_tab(SHEET_URL,"ledger")
    if df_daily.empty or ledger_df.empty:
        st.warning("No data yet. Record trades first.")
    else:
        eq = pd.to_numeric(df_daily["equity"])
        eq.index = pd.to_datetime(df_daily["date"])
        dd = compute_drawdown(eq)

        # Drawdown chart
        fig, ax = plt.subplots(figsize=(8,3))
        ax.fill_between(dd.index, dd.values,0, color="red", alpha=0.4)
        ax.set_title("Drawdown curve")
        st.pyplot(fig)

        # Rolling 1Y CAGR
        if len(eq)>252:
            roll_cagr = (eq/eq.shift(252))**(1)-1
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(roll_cagr.index, roll_cagr.values)
            ax.set_title("Rolling 1Y CAGR")
            st.pyplot(fig)

        # Exposure plot
        exp = pd.to_numeric(df_daily["exposure"])
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(df_daily["date"], exp)
        ax.set_title("Exposure over time")
        st.pyplot(fig)

        # Histogram of realized PnL
        if "realized_pnl" in ledger_df.columns and not ledger_df["realized_pnl"].dropna().empty:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(ledger_df["realized_pnl"].dropna(), bins=30, color="blue", alpha=0.6)
            ax.set_title("Realized PnL Distribution")
            st.pyplot(fig)































