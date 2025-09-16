# balancedb_app.py
# Balanced_B mobile signal terminal (NIFTY100, daily MTM, Google Sheets storage)
# - Run any time market is open (IST 09:15–15:30). Signals use LTP at run time.
# - Portfolio lives in your Google Sheet (no app login). Start from empty portfolio.
# - Daily performance uses close-to-close valuation (MTM) with Option B backfill.
# - Charts: Equity vs NIFTY100 TRI (fallback to price index labeled), exposure, drawdown, rolling 1Y CAGR.
# - Downloads: ledger, holdings, daily_summary, equity_series.
#
# Setup (local):
#   pip install -r requirements.txt
#   streamlit run balancedb_app.py
#
# Streamlit Cloud:
#   Add your Google service account JSON in st.secrets as gcp_service_account.
#   Share your Google Sheet with that service account email.
#   Open the app, paste your Sheet URL, run.

import os, re, time, json, math, textwrap
from io import StringIO
from datetime import datetime, timedelta, date
import zoneinfo
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

def safe_yf_download(tickers, **kwargs):
    try:
        df = yf.download(tickers, auto_adjust=False, **kwargs)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        try:
            import streamlit as st
            st.warning(f"Yahoo download failed for {tickers}: {e}")
        except Exception:
            print(f"Yahoo download failed for {tickers}: {e}")
        return pd.DataFrame()


# Google Sheets
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

# ----------------- Strategy Params (defaults; can be edited in Sheet->config) -----------------
DEFAULTS = dict(
    base_capital=500_000.0,
    fee=0.0011,  # 0.11% per side
    index_name="NIFTY100",
    ma=20,
    use_zscore=True,
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
    min_turnover_cr=8.0,
    turnover_window=20,
    apply_turnover_to_averaging=True,
    divisor=30,        # bull
    divisor_bear=38,   # bear
    lookback_days=420, # for MA/std/time stop, liquidity
)

# ----------------- Streamlit config -----------------
st.set_page_config(page_title="Balanced_B Signals (NIFTY100)", page_icon="📈", layout="wide")

# ----------------- Utilities -----------------
def now_ist():
    return datetime.now(IST)

def is_weekday(d):
    return d.weekday() < 5

def is_time_between(t, start_hm=(9,15), end_hm=(15,30)):
    sh, sm = start_hm; eh, em = end_hm
    return (t.hour, t.minute) >= (sh, sm) and (t.hour, t.minute) <= (eh, em)

def is_market_open():
    """Best-effort: weekday + time-window; optionally ping NSE marketStatus."""
    t = now_ist()
    if not is_weekday(t): return False, "Weekend"
    if not is_time_between(t.time() if hasattr(t, "time") else t, (9,15), (15,30)):
        return False, "Outside trading hours (IST 09:15–15:30)"
    # Optional NSE status check (fallback-friendly)
    try:
        s = requests.Session()
        s.headers.update({"User-Agent":"Mozilla/5.0"})
        r = s.get("https://www.nseindia.com/api/marketStatus", timeout=8)
        if r.ok:
            js = r.json()
            for seg in js.get("marketState", []):
                if "Equity" in seg.get("market", ""):
                    if seg.get("marketStatus") == "Open":
                        return True, "Market Open"
            # If structure differs, ignore and fall back
    except Exception:
        pass
    return True, "Market Open (holiday check not confirmed)"

def _norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
def _clean_symbol_keep_punct(s): return re.sub(r'[^A-Za-z0-9\-\&\.]+', '', str(s)).upper()

def read_html_tables(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    html = resp.text
    for kw in ({}, {"flavor":"bs4"}, {"flavor":"html5lib"}):
        try:
            return pd.read_html(StringIO(html), **kw)
        except Exception:
            continue
    raise RuntimeError("Install parsers: pip install beautifulsoup4 html5lib")

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_nifty100_symbols():
    url = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.content.decode("utf-8")))
        sym_col = [c for c in df.columns if re.search(r'(symbol|ticker|nse)', c, re.I)][0]
        df[sym_col] = df[sym_col].apply(_clean_symbol_keep_punct)
        syms = sorted(df[sym_col].dropna().astype(str).tolist())
        return [s + ".NS" for s in syms]
    except Exception:
        # Fallback to Wikipedia
        tables = read_html_tables("https://en.wikipedia.org/wiki/NIFTY_100")
        cons = None
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any('symbol' in c or 'ticker' in c or 'nse' in c for c in cols) and any('company' in c or 'name' in c for c in cols) and len(t) >= 40:
                cons = t.copy(); break
        if cons is None: raise RuntimeError("Could not fetch constituents")
        sym_col = [c for c in cons.columns if re.search(r'(symbol|ticker|code|nse)', c, re.I)][0]
        cons[sym_col] = cons[sym_col].apply(_clean_symbol_keep_punct)
        syms = sorted(cons[sym_col].tolist())
        return [s + ".NS" for s in syms]

def yf_intraday_last(tickers):
    """Get last 1m price for a list (chunks). Returns dict ticker->price."""
    out = {}
    tickers = list(dict.fromkeys(tickers))
    if not tickers: return out
    for i in range(0, len(tickers), 40):
        batch = tickers[i:i+40]
        try:
            data = safe_yf_download(batch, period="1d", interval="1m", progress=False, auto_adjust=False, threads=True)
        except Exception:
            data = None
        if data is None or not isinstance(data, pd.DataFrame): continue
        if isinstance(data.columns, pd.MultiIndex):
            # pick Close
            close = data.get("Close")
            if close is None: continue
            last = close.dropna(how="all").ffill().tail(1).T.squeeze()
            for sym, px in last.items():
                if pd.notna(px): out[sym] = float(px)
        else:
            last = data["Close"].dropna().tail(1).T.squeeze()
            if isinstance(last, pd.Series):
                for sym, px in last.items():
                    if pd.notna(px): out[sym] = float(px)
            else:
                # single ticker Series
                out[batch[0]] = float(last)
    return out

def download_fields(tickers, start, end, fields=("Adj Close","Volume"), chunk=50):
    tickers = list(dict.fromkeys([t for t in tickers if isinstance(t, str)]))
    out = {f: [] for f in fields}
    for i in range(0, len(tickers), chunk):
        t = tickers[i:i+chunk]
        data = safe_yf_download(t, start=start, end=end, progress=False, auto_adjust=False, threads=True)
        if not isinstance(data, pd.DataFrame): continue
        if isinstance(data.columns, pd.MultiIndex):
            for f in fields:
                sub = data.get(f)
                if sub is None: continue
                out[f].append(sub.copy())
        else:
            for f in fields:
                sub = data.get(f)
                if sub is None: continue
                if isinstance(sub, pd.Series): sub = sub.to_frame()
                out[f].append(sub.copy())
    for f in fields:
        if out[f]:
            df = pd.concat(out[f], axis=1)
            df = df.loc[:, ~df.columns.duplicated()].sort_index()
            out[f] = df
        else:
            out[f] = pd.DataFrame()
    return out

@st.cache_data(ttl=2*3600, show_spinner=True)
def load_market_data(lookback_days=420):
    today = now_ist().date()
    start = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end   = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    tickers = fetch_nifty100_symbols()
    # Benchmark (NIFTY50 price index)
    b = safe_yf_download("^NSEI", start=start, end=end, progress=False, auto_adjust=False)
    if not b.empty:
        if "Adj Close" in b:
            series = b["Adj Close"]
        elif "Close" in b:
            series = b["Close"]
        else:
            series = pd.Series(dtype=float)

        if isinstance(series, pd.DataFrame):
            series = series.squeeze("columns")

        bench = series.dropna().rename("NIFTY50")
    else:
        bench = pd.Series(dtype=float, name="NIFTY50")

    fields = download_fields(tickers, start, end, fields=("Adj Close","Volume"))
    prices = fields["Adj Close"].reindex(bench.index).ffill()
    vols   = fields["Volume"].reindex(bench.index).ffill()
    turnover_cr = (prices * vols) / 1e7
    med_turnover = turnover_cr.rolling(20, min_periods=20).median()
    return dict(bench=bench, prices=prices, vols=vols, med_turnover=med_turnover, tickers=list(prices.columns))

@st.cache_data(ttl=12*3600, show_spinner=False)
def fetch_nifty100_tri():
    """
    Try to fetch NIFTY100 TRI history.
    Fallback: NIFTY100 price index (Yahoo '^CNX100') clearly labeled as 'Price index (fallback)'.
    """
    # Try Yahoo NIFTY100 price index first (stable)
    tri = None; label = "NIFTY100 (price index, fallback — TRI unavailable)"
    try:
        y = safe_yf_download("^CNX100", period="max", interval="1d", progress=False, auto_adjust=False)
        ser = (y["Adj Close"] if "Adj Close" in y else y["Close"]).dropna()
        tri = ser.rename(label)
    except Exception:
        pass
    return tri, label

def liq_ok(med_turn_row, sym, min_turnover_cr):
    v = med_turn_row.get(sym, np.nan)
    return (not pd.isna(v)) and (v >= min_turnover_cr)

def shares_from_lot(price, lot_cash, fee):
    per_share = price * (1 + fee)
    if per_share <= 0: return 0
    return max(int(lot_cash // per_share), 0)

# ----------------- Google Sheets helpers -----------------
REQ_TABS = ["balances","positions","ledger","config","daily_equity"]

def _service_account():
    # Use Streamlit secrets (recommended on Streamlit Cloud)
    if "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=["https://www.googleapis.com/auth/spreadsheets"])
        return gspread.authorize(creds), st.secrets["gcp_service_account"].get("client_email", "")
    # Local: look for env var pointing to JSON file
    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path and os.path.exists(sa_path):
        creds = Credentials.from_service_account_file(sa_path, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        return gspread.authorize(creds), json.load(open(sa_path)).get("client_email","")
    st.error("Service account credentials not found. Add to st.secrets['gcp_service_account'] or set GOOGLE_APPLICATION_CREDENTIALS.")
    st.stop()

def open_sheet(url):
    gc, sa_email = _service_account()
    try:
        sh = gc.open_by_url(url)
        return sh, sa_email
    except Exception as e:
        st.error(f"Could not open Sheet. Share it with the service account email. Error: {e}")
        st.stop()

def ensure_tabs(sh):
    titles = [w.title for w in sh.worksheets()]
    def make(name, cols):
        ws = sh.add_worksheet(title=name, rows=1000, cols=max(10,len(cols)+2))
        set_with_dataframe(ws, pd.DataFrame(columns=cols))
    if "balances" not in titles:
        make("balances", ["cash","base_capital","realized","fees_paid","last_update"])
        set_with_dataframe(sh.worksheet("balances"), pd.DataFrame([dict(cash=DEFAULTS["base_capital"],
                                                                        base_capital=DEFAULTS["base_capital"],
                                                                        realized=0.0, fees_paid=0.0,
                                                                        last_update=str(date.today()))]))
    if "positions" not in titles:
        make("positions", ["symbol","shares","avg_cost","last_buy","open_date"])
    if "ledger" not in titles:
        make("ledger", ["date","side","symbol","shares","price","fee","reason","realized_pnl"])
    if "config" not in titles:
        make("config", ["fee","divisor","divisor_bear","take_profit","time_stop_days","telegram_token","telegram_chat_id"])
        set_with_dataframe(sh.worksheet("config"),
                           pd.DataFrame([dict(fee=DEFAULTS["fee"], divisor=DEFAULTS["divisor"], divisor_bear=DEFAULTS["divisor_bear"],
                                              take_profit=DEFAULTS["take_profit"], time_stop_days=DEFAULTS["time_stop_days"],
                                              telegram_token="", telegram_chat_id="")]))
    if "daily_equity" not in titles:
        make("daily_equity", ["date","equity","cash","invested","exposure","source"])

def load_all(sh):
    def df_of(name):
        ws = sh.worksheet(name)
        df = get_as_dataframe(ws, evaluate_formulas=True, header=0).dropna(how="all")
        df.columns = [c.strip() for c in df.columns]
        return df
    balances = df_of("balances")
    positions = df_of("positions")
    ledger = df_of("ledger")
    config = df_of("config")
    daily_eq = df_of("daily_equity")
    # Clean types
    if not balances.empty:
        for c in ["cash","base_capital","realized","fees_paid"]:
            if c in balances.columns:
                balances[c] = pd.to_numeric(balances[c], errors="coerce")
    if not positions.empty:
        positions["shares"] = pd.to_numeric(positions["shares"], errors="coerce").fillna(0).astype(int)
        for c in ["avg_cost","last_buy"]:
            if c in positions.columns: positions[c] = pd.to_numeric(positions[c], errors="coerce")
    if not ledger.empty:
        ledger["shares"] = pd.to_numeric(ledger["shares"], errors="coerce").fillna(0).astype(int)
        for c in ["price","fee","realized_pnl"]:
            if c in ledger.columns: ledger[c] = pd.to_numeric(ledger[c], errors="coerce")
    if not daily_eq.empty:
        for c in ["equity","cash","invested","exposure"]:
            if c in daily_eq.columns: daily_eq[c] = pd.to_numeric(daily_eq[c], errors="coerce")
    return balances, positions, ledger, config, daily_eq

def save_df(sh, name, df):
    ws = sh.worksheet(name)
    ws.clear()
    set_with_dataframe(ws, df)

# ----------------- Portfolio valuation + backfill (daily close MTM) -----------------
def reconstruct_daily_equity(ledger, balances, start_day, end_day, price_df, fee_default):
    """
    Recompute full daily equity series from inception to end_day (inclusive).
    ledger: BUY/SELL/FUND_IN/FUND_OUT with date, shares, price, fee.
    price_df: Adj Close for all symbols seen in ledger+positions, ffilled.
    """
    if price_df.empty:
        # Nothing to value yet
        return pd.DataFrame(columns=["date","equity","cash","invested","exposure","source"])
    # Prep dates
    days = price_df.index[(price_df.index.date >= start_day) & (price_df.index.date <= end_day)]
    if len(days)==0:
        return pd.DataFrame(columns=["date","equity","cash","invested","exposure","source"])
    # Initial state
    cash = float(balances.iloc[0]["cash"]) if not balances.empty else DEFAULTS["base_capital"]
    base_capital = float(balances.iloc[0]["base_capital"]) if not balances.empty else DEFAULTS["base_capital"]
    realized = float(balances.iloc[0]["realized"]) if not balances.empty else 0.0
    fees_paid = float(balances.iloc[0]["fees_paid"]) if not balances.empty else 0.0
    positions = {}  # sym-> dict(shares, avg_cost, last_buy, open_date)
    # Sort ledger by date
    led = ledger.copy()
    if "date" in led.columns:
        led["date"] = pd.to_datetime(led["date"]).dt.date
    else:
        led["date"] = []
    led = led.sort_values(["date","side","symbol"])
    rows=[]
    # Iterate trading days
    for d in days:
        day = d.date()
        # apply today's ledger
        day_trades = led[led["date"]==day] if not led.empty else pd.DataFrame(columns=led.columns)
        for _, tr in day_trades.iterrows():
            side = (tr.get("side") or "").upper()
            sym = str(tr.get("symbol") or "").strip()
            qty = int(tr.get("shares") or 0)
            px  = float(tr.get("price") or 0.0)
            fee = float(tr.get("fee") or (fee_default * qty * px))
            if side == "FUND_IN":
                cash += px
                base_capital += px
            elif side == "FUND_OUT":
                cash -= px
                base_capital -= px
            elif side == "BUY" and sym:
                gross = qty * px
                total = gross + fee
                cash -= total
                fees_paid += fee
                if sym in positions:
                    pos = positions[sym]
                    tot_cost = pos["avg_cost"]*pos["shares"] + gross
                    pos["shares"] += qty
                    pos["avg_cost"] = (tot_cost / pos["shares"]) if pos["shares"]>0 else pos["avg_cost"]
                    pos["last_buy"] = px
                else:
                    positions[sym] = dict(shares=qty, avg_cost=px, last_buy=px, open_date=day)
            elif side == "SELL" and sym:
                if sym not in positions: continue
                pos = positions[sym]
                qty = min(qty, pos["shares"])
                gross = qty * px
                proceeds = gross - fee
                cash += proceeds
                fees_paid += fee
                realized += proceeds - qty*pos["avg_cost"]
                pos["shares"] -= qty
                if pos["shares"] <= 0:
                    del positions[sym]
        # value positions at close
        invested = 0.0
        for sym, pos in positions.items():
            if sym not in price_df.columns: continue
            close_px = float(price_df.loc[d, sym])
            invested += close_px * pos["shares"]
        equity = cash + invested
        exposure = invested / equity if equity>0 else 0.0
        rows.append(dict(date=str(day), equity=round(equity,2), cash=round(cash,2),
                         invested=round(invested,2), exposure=round(exposure,4), source="close"))
    return pd.DataFrame(rows)

def position_snapshot(positions_df, last_close_row):
    rows=[]; mv=0.0
    if positions_df is None or positions_df.empty:
        return pd.DataFrame(columns=["symbol","shares","avg_cost","last_price","market_value","unrealized_pnl","unrealized_pct"]), 0.0
    for _, r in positions_df.iterrows():
        sym = r["symbol"]; sh = int(r["shares"]); avg = float(r["avg_cost"])
        px = float(last_close_row.get(sym, np.nan)) if sym in last_close_row.index else np.nan
        if pd.isna(px): px=0.0
        mval = sh*px; mv += mval
        unr = (px-avg)*sh
        unr_pct = (px/avg - 1) if avg>0 else np.nan
        rows.append(dict(symbol=sym, shares=sh, avg_cost=round(avg,2), last_price=round(px,2),
                         market_value=round(mval,2), unrealized_pnl=round(unr,2),
                         unrealized_pct=round((unr_pct or 0)*100,2)))
    return pd.DataFrame(rows).sort_values("market_value", ascending=False), mv

# ----------------- Signals (Balanced_B) -----------------
def compute_signals(params, mkt, positions_df, balances_df, ledger_df, sells_done_today):
    p = params
    bench = mkt["bench"]; prices = mkt["prices"]; vols = mkt["vols"]; med_turnover = mkt["med_turnover"]
    tickers = mkt["tickers"]
    # Use live prices for today if market open
    live = yf_intraday_last(tickers + ["^NSEI"])
    today = bench.index[-1]  # last trading day in history (close basis)
    # bench MA60 from closes; regime check uses live NIFTY50 if available
    bench_ma60 = bench.rolling(p["regime_filter_ma"], min_periods=p["regime_filter_ma"]).mean().iloc[-1]
    bench_live = live.get("^NSEI", float(bench.iloc[-1]))
    regime_ok = bool(bench_live >= float(bench_ma60)*(1+p["regime_buffer"])) if pd.notna(bench_ma60) else True
    div_today = p["divisor"] if regime_ok else p["divisor_bear"]

    # Price frame with "current" price = live if available else last close
    row_live = prices.iloc[-1].copy()
    for sym in row_live.index:
        if sym in live: row_live[sym] = live[sym]

    # MA/std from closes
    ma  = prices.rolling(p["ma"], min_periods=p["ma"]).mean().iloc[-1]
    std = prices.rolling(p["ma"], min_periods=p["ma"]).std().iloc[-1]
    liq_today = med_turnover.iloc[-1]

    # Balances and lot sizing
    base_capital = float(balances_df.iloc[0]["base_capital"]) if not balances_df.empty else DEFAULTS["base_capital"]
    realized = float(balances_df.iloc[0]["realized"]) if not balances_df.empty else 0.0
    cash = float(balances_df.iloc[0]["cash"]) if not balances_df.empty else base_capital
    lot_cash = (base_capital + realized) / div_today

    # Build positions dict
    positions = {}
    if not positions_df.empty:
        for _, r in positions_df.iterrows():
            positions[str(r["symbol"])] = dict(shares=int(r["shares"]), avg_cost=float(r["avg_cost"]),
                                               last_buy=float(r.get("last_buy", r["avg_cost"])),
                                               open_date=str(r.get("open_date") or ""))

    # Sells (TP then time-stop), cap to remaining today
    sells=[]; left = max(0, p["max_sells_per_day"] - sells_done_today)
    # TP
    for sym, pos in positions.items():
        px = float(row_live.get(sym, np.nan))
        if pd.isna(px) or pos["shares"]<=0: continue
        r = px/pos["avg_cost"] - 1
        if r >= p["take_profit"]:
            sells.append(dict(symbol=sym, price=round(px,2), shares=pos["shares"], reason="TP", ret=round(r*100,2)))
    sells.sort(key=lambda x: x.get("ret",0), reverse=True)
    sells = sells[:left]
    # Time stop if capacity remains
    left -= len(sells)
    if left>0:
        today_d = now_ist().date()
        for sym, pos in positions.items():
            if left<=0: break
            px = float(row_live.get(sym, np.nan))
            if pd.isna(px) or pos["shares"]<=0: continue
            try:
                od = pd.to_datetime(pos["open_date"]).date()
            except Exception:
                od = today_d
            age = (today_d - od).days
            if age >= p["time_stop_days"] and sym not in [s["symbol"] for s in sells]:
                sells.append(dict(symbol=sym, price=round(px,2), shares=pos["shares"], reason="TIME_STOP"))
                left -= 1

    # Eligible for new buys
    elig = [c for c in tickers if pd.notna(ma.get(c)) and pd.notna(row_live.get(c)) and row_live[c] < ma[c] and liq_ok(liq_today, c, p["min_turnover_cr"])]
    ranked_syms=[]
    if p["use_zscore"]:
        zlist=[]
        for c in elig:
            s = std.get(c)
            if pd.isna(s) or s<=0: continue
            z = (row_live[c] - ma[c]) / s
            zlist.append((z,c))
        zlist.sort(key=lambda x:x[0])
        ranked_syms = [c for _,c in zlist[:p["bottom_n"]]]
    else:
        dist = (row_live/ma - 1).dropna()
        ranked_syms = list(dist.sort_values().index[:p["bottom_n"]])

    # New buys
    buys_new = []
    for sym in ranked_syms:
        if sym in positions: continue
        px = float(row_live.get(sym, np.nan))
        if pd.isna(px): continue
        sh = shares_from_lot(px, lot_cash, p["fee"])
        if sh<=0: continue
        cost = sh*px*(1+p["fee"])
        buys_new.append(dict(symbol=sym, price=round(px,2), shares=sh, reason="NEW", est_cost=round(cost,2)))
        if len(buys_new) >= p["max_new_buys"]: break

    # Averaging candidates
    buys_avg=[]
    for sym, pos in positions.items():
        px = float(row_live.get(sym, np.nan))
        if pd.isna(px): continue
        if p["apply_turnover_to_averaging"] and not liq_ok(liq_today, sym, p["min_turnover_cr"]): continue
        m = ma.get(sym); s = std.get(sym)
        if pd.isna(m) or pd.isna(s) or s<=0: continue
        z = (px - m)/s
        price_ok = px <= pos["last_buy"] * (1 - p["avg_dd"])
        regime_gate = regime_ok or (not p["averaging_requires_regime"] and z <= p["avg_in_bear_z_thresh"])
        if price_ok and regime_gate:
            sh = shares_from_lot(px, lot_cash, p["fee"])
            if sh>0:
                buys_avg.append(dict(symbol=sym, price=round(px,2), shares=sh, reason="AVERAGE", z=round(float(z),2)))
    buys_avg.sort(key=lambda x: x.get("z", 0.0))

    return dict(
        regime_ok=regime_ok,
        bench_live=bench_live,
        lot_cash=lot_cash,
        sells=sells,
        buys_new=buys_new,
        buys_avg=buys_avg,
        cash=cash
    )

# ----------------- Apply executed trades -----------------
def apply_trade_rows(sh, trades, fee_rate):
    """trades: list of dicts {date, side, symbol, shares, price, fee, reason}"""
    # Load
    balances, positions, ledger, config, daily_eq = load_all(sh)
    # Current balances
    if balances.empty:
        balances = pd.DataFrame([dict(cash=DEFAULTS["base_capital"], base_capital=DEFAULTS["base_capital"], realized=0.0, fees_paid=0.0, last_update=str(date.today()))])
    cash = float(balances.iloc[0]["cash"]); base_capital=float(balances.iloc[0]["base_capital"])
    realized=float(balances.iloc[0]["realized"]); fees_paid=float(balances.iloc[0]["fees_paid"])
    # Positions map
    pos_map = {}
    for _, r in positions.iterrows():
        pos_map[str(r["symbol"])] = dict(shares=int(r["shares"]), avg_cost=float(r["avg_cost"]),
                                         last_buy=float(r.get("last_buy", r["avg_cost"])),
                                         open_date=str(r.get("open_date") or ""))

    # Append to ledger and update pos/cash
    for tr in trades:
        side = tr["side"].upper()
        sym = tr.get("symbol","").strip()
        qty = int(tr.get("shares",0))
        px  = float(tr.get("price",0.0))
        fee = float(tr.get("fee", px*qty*fee_rate))
        reason = tr.get("reason","")
        dt = tr.get("date", str(now_ist().date()))
        if side == "FUND_IN":
            cash += px; base_capital += px
            ledger = pd.concat([ledger, pd.DataFrame([dict(date=dt, side=side, symbol="", shares=0, price=px, fee=0.0, reason="FUND_IN", realized_pnl=0.0)])], ignore_index=True)
        elif side == "FUND_OUT":
            cash -= px; base_capital -= px
            ledger = pd.concat([ledger, pd.DataFrame([dict(date=dt, side=side, symbol="", shares=0, price=px, fee=0.0, reason="FUND_OUT", realized_pnl=0.0)])], ignore_index=True)
        elif side == "BUY" and sym and qty>0 and px>0:
            gross = qty*px
            total = gross + fee
            if cash < total:
                continue  # insufficient cash; skip
            cash -= total; fees_paid += fee
            if sym in pos_map:
                pos = pos_map[sym]
                tot_cost = pos["avg_cost"]*pos["shares"] + gross
                pos["shares"] += qty
                pos["avg_cost"] = (tot_cost / pos["shares"]) if pos["shares"]>0 else pos["avg_cost"]
                pos["last_buy"] = px
            else:
                pos_map[sym] = dict(shares=qty, avg_cost=px, last_buy=px, open_date=dt)
            ledger = pd.concat([ledger, pd.DataFrame([dict(date=dt, side="BUY", symbol=sym, shares=qty, price=px, fee=round(fee,2), reason=reason, realized_pnl=0.0)])], ignore_index=True)
        elif side == "SELL" and sym and qty>0 and px>0:
            if sym not in pos_map: continue
            pos = pos_map[sym]
            qty = min(qty, pos["shares"])
            gross = qty*px
            proceeds = gross - fee
            pnl = proceeds - qty*pos["avg_cost"]
            cash += proceeds; fees_paid += fee; realized += pnl
            pos["shares"] -= qty
            if pos["shares"] <= 0: del pos_map[sym]
            ledger = pd.concat([ledger, pd.DataFrame([dict(date=dt, side="SELL", symbol=sym, shares=qty, price=px, fee=round(fee,2), reason=reason, realized_pnl=round(pnl,2))])], ignore_index=True)

    # Save positions
    pos_rows=[]
    for sym, pos in pos_map.items():
        pos_rows.append(dict(symbol=sym, shares=pos["shares"], avg_cost=round(pos["avg_cost"],2), last_buy=round(pos.get("last_buy", pos["avg_cost"]),2), open_date=pos.get("open_date","")))
    positions_out = pd.DataFrame(pos_rows).sort_values("symbol") if pos_rows else pd.DataFrame(columns=["symbol","shares","avg_cost","last_buy","open_date"])
    balances_out = pd.DataFrame([dict(cash=round(cash,2), base_capital=round(base_capital,2), realized=round(realized,2), fees_paid=round(fees_paid,2), last_update=str(now_ist().date()))])
    # Persist
    save_df(sh, "positions", positions_out)
    save_df(sh, "ledger", ledger.sort_values(["date","side","symbol"]))
    save_df(sh, "balances", balances_out)

# ----------------- Performance metrics -----------------
def compute_xirr(flows):
    """flows: list of (date, amount), FUND_IN negative, FUND_OUT positive, final equity positive."""
    if not flows or len(flows)<2: return np.nan
    # Convert to ordinal days
    dates = [pd.to_datetime(d).to_pydatetime() for d,_ in flows]
    amounts = [float(a) for _,a in flows]
    t0 = dates[0]
    days = np.array([(d - t0).days for d in dates], dtype=float)
    def xnpv(rate):
        return np.sum([amt / ((1+rate)**(dd/365.0)) for amt,dd in zip(amounts, days)])
    # Bracket root between -0.999 and very high
    low, high = -0.999, 10.0
    for _ in range(100):
        mid = (low+high)/2
        v = xnpv(mid)
        if abs(v) < 1e-6: return mid
        v_low = xnpv(low)
        if v_low * v < 0:
            high = mid
        else:
            low = mid
    return np.nan

def compute_twr_cagr(daily_eq_df, ledger):
    if daily_eq_df.empty or len(daily_eq_df)<2: return np.nan, np.nan
    df = daily_eq_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")
    # net flows per day: FUND_IN negative, FUND_OUT positive (convention for return calc)
    led = ledger.copy()
    if led.empty:
        led = pd.DataFrame(columns=["date","side","price"])
    else:
        led["date"] = pd.to_datetime(led["date"]).dt.date
    flow = led.groupby(["date","side"])["price"].sum().unstack(fill_value=0.0)
    flow["net"] = flow.get("FUND_OUT",0.0) - flow.get("FUND_IN",0.0)
    flow = flow["net"]
    rets=[]
    prev = None
    for d, row in df.set_index("date").iterrows():
        if prev is None:
            prev = row["equity"]; continue
        beg = prev
        net = float(flow.get(d, 0.0))
        r = (row["equity"] - beg - net) / beg if beg>0 else 0.0
        rets.append(r)
        prev = row["equity"]
    if not rets: return np.nan, np.nan
    twr_total = np.prod([1+r for r in rets]) - 1
    # Annualize using calendar years between first and last
    start = pd.to_datetime(df["date"].iloc[0])
    end   = pd.to_datetime(df["date"].iloc[-1])
    years = max((end - start).days / 365.25, 1e-9)
    cagr = (1 + twr_total)**(1/years) - 1
    return twr_total, cagr

# ----------------- UI -----------------
st.title("📈 Balanced_B Signals — NIFTY100")
st.caption("Signals use live LTP during market hours. Performance is daily mark-to-market (close-to-close).")

with st.expander("Setup help (Google Sheets + Service Account)", expanded=False):
    st.markdown("""
- Create a Google Service Account (Sheets API) and add its JSON to Streamlit Secrets as key gcp_service_account.
- Copy our Sheet template structure (the app will create tabs if missing).
- Share your Google Sheet (Editor) with the service account email shown below.
""")

# Settings (Sheet URL)
col1, col2 = st.columns([2,1])
with col1:
    sheet_url = st.text_input("Paste your Google Sheet URL (your personal state)", value=st.session_state.get("sheet_url",""))
with col2:
    if st.button("Remember URL in this session"):
        st.session_state["sheet_url"] = sheet_url
if not sheet_url:
    st.info("Paste your Google Sheet URL above to begin.")
    # Still show service account email for sharing
    try:
        _, sa_email = _service_account()
        st.write(f"Service account email (share your Sheet with this): {sa_email}")
    except Exception:
        pass
    st.stop()

# Open sheet and ensure tabs
sh, sa_email = open_sheet(sheet_url)
ensure_tabs(sh)
balances, positions, ledger, config, daily_eq = load_all(sh)

# Expose key config values (editable in Sheet->config)
cfg = DEFAULTS.copy()
if not config.empty:
    for k in ["fee","divisor","divisor_bear","take_profit","time_stop_days"]:
        if k in config.columns and pd.notna(config.iloc[0][k]):
            cfg[k] = float(config.iloc[0][k])
base_capital = float(balances.iloc[0]["base_capital"]) if not balances.empty else DEFAULTS["base_capital"]

# Load market data (history for MA/liquidity + bench)
mkt = load_market_data(DEFAULTS["lookback_days"])
tri_ser, tri_label = fetch_nifty100_tri()

# Backfill daily equity (Option B): recompute full series from inception to last completed trading day
# Universe for backfill = any symbol in ledger + current positions
universe = sorted(set(list(positions["symbol"]) + list(ledger["symbol"].dropna().unique())))
if universe:
    # Download close prices for universe (Adj Close), build trading days from ^NSEI
    start_hist = (now_ist().date() - timedelta(days=720)).strftime("%Y-%m-%d")
    end_hist   = (now_ist().date() + timedelta(days=2)).strftime("%Y-%m-%d")
    fields_hist = download_fields(universe, start_hist, end_hist, fields=("Adj Close",))
    px_hist = fields_hist["Adj Close"].ffill()
    # Align to NIFTY50 trading days
    px_hist = px_hist.reindex(mkt["bench"].index).ffill()
    # Backfill up to last completed trading day
    last_done = pd.to_datetime(daily_eq["date"]).dt.date.max() if not daily_eq.empty else None
    first_needed = (pd.to_datetime(ledger["date"]).dt.date.min() if not ledger.empty else now_ist().date())
    start_day = min(first_needed, now_ist().date()) if first_needed else now_ist().date()
    today = now_ist().date()
    # If market still open, we backfill only up to previous trading day
    if not mkt["bench"].empty:
        cutoff = mkt["bench"].index[-1].date()
    else:
        cutoff = now_ist().date()

    df_new = reconstruct_daily_equity(ledger, balances, start_day, cutoff, px_hist, cfg["fee"])

    if not df_new.empty:
        save_df(sh, "daily_equity", df_new)
        daily_eq = df_new.copy()

# Tabs
tab1, tab2 = st.tabs(["Run Signals", "My Portfolio"])

with tab1:
    open_ok, open_msg = is_market_open()
    st.markdown(f"**Market status:** {open_msg}")
    if not open_ok:
        st.warning("Run is disabled when market is closed.")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        run_click = st.button("▶️ Run scan", use_container_width=True, disabled=not open_ok)
    with colB:
        fund_amt = st.number_input("Add/Withdraw funds (₹): + add, - withdraw", value=0, step=500)
    with colC:
        if st.button("Apply funds", use_container_width=True):
            if fund_amt != 0:
                apply_trade_rows(sh, [dict(date=str(now_ist().date()), side="FUND_IN" if fund_amt>0 else "FUND_OUT",
                                           symbol="", shares=0, price=abs(float(fund_amt)), fee=0.0, reason="")], cfg["fee"])
                st.success("Funds updated.")
                balances, positions, ledger, config, daily_eq = load_all(sh)

    # Show service account email for sharing
    st.caption(f"Service account email (share your Sheet with this): {sa_email}")

    if run_click and open_ok:
        # Count sells already executed today (hard cap 4/day)
        today_d = now_ist().date()
        sells_today = 0
        if not ledger.empty:
            ld = ledger.copy(); ld["date"] = pd.to_datetime(ld["date"]).dt.date
            sells_today = int(((ld["date"]==today_d) & (ld["side"].str.upper()=="SELL")).sum())
        signals = compute_signals(cfg, mkt, positions, balances, ledger, sells_today)
        st.markdown(f"- Regime: {'BULL' if signals['regime_ok'] else 'BEAR'} | Lot cash: ₹{signals['lot_cash']:.0f} | Cash: ₹{signals['cash']:.0f}")
        # Sells
        st.subheader("Sells (cap 4/day)")
        if not signals["sells"]:
            st.info("No sells by rule.")
        else:
            st.dataframe(pd.DataFrame(signals["sells"]), use_container_width=True)
        # New buys
        st.subheader("New Buys (ranked)")
        if not signals["buys_new"]:
            st.info("No new buys qualified.")
        else:
            st.dataframe(pd.DataFrame(signals["buys_new"]), use_container_width=True)
        # Averaging
        st.subheader("Averaging candidates")
        if not signals["buys_avg"]:
            st.info("No averaging candidates.")
        else:
            st.dataframe(pd.DataFrame(signals["buys_avg"]), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Mark executed trades (partial fills OK)")
        execs=[]
        with st.form("exec_trades"):
            # Sells
            if signals["sells"]:
                st.write("Executed Sells")
                cap_left = cfg["max_sells_per_day"] - sells_today
                for i, s in enumerate(signals["sells"]):
                    c1,c2,c3,c4 = st.columns([2,1,1,2])
                    with c1: do = st.checkbox(f"SELL {s['symbol']} ({s['reason']})", key=f"sell_{i}")
                    with c2: qty = st.number_input("Qty", min_value=0, max_value=int(s["shares"]), value=int(s["shares"]), step=1, key=f"sellqty_{i}")
                    with c3: px  = st.number_input("Price", min_value=0.0, value=float(s["price"]), step=0.05, key=f"sellpx_{i}")
                    with c4: st.caption(f"TP%: {s.get('ret','')}")
                    if do and qty>0:
                        execs.append(dict(side="SELL", symbol=s["symbol"], shares=qty, price=px, reason=s["reason"]))
                # enforce cap
                if len([e for e in execs if e["side"]=="SELL"]) > cap_left:
                    st.warning(f"You can execute at most {cap_left} sells today. Extra sells will be ignored.")

            # New Buys
            if signals["buys_new"]:
                st.write("Executed New Buys")
                for i, b in enumerate(signals["buys_new"]):
                    c1,c2,c3,c4 = st.columns([2,1,1,2])
                    with c1: do = st.checkbox(f"BUY {b['symbol']}", key=f"buynew_{i}")
                    with c2: qty = st.number_input("Qty", min_value=0, value=int(b["shares"]), step=1, key=f"buynewqty_{i}")
                    with c3: px  = st.number_input("Price", min_value=0.0, value=float(b["price"]), step=0.05, key=f"buynewpx_{i}")
                    with c4: st.caption(f"Est cost: ₹{b['est_cost']}")
                    if do and qty>0:
                        execs.append(dict(side="BUY", symbol=b["symbol"], shares=qty, price=px, reason="NEW"))
            # Averaging
            if signals["buys_avg"]:
                st.write("Executed Averaging")
                for i, b in enumerate(signals["buys_avg"]):
                    c1,c2,c3,c4 = st.columns([2,1,1,2])
                    with c1: do = st.checkbox(f"AVG {b['symbol']} (z={b.get('z','')})", key=f"buyavg_{i}")
                    with c2: qty = st.number_input("Qty", min_value=0, value=int(b["shares"]), step=1, key=f"buyavgqty_{i}")
                    with c3: px  = st.number_input("Price", min_value=0.0, value=float(b["price"]), step=0.05, key=f"buyavgpx_{i}")
                    with c4: st.caption("Lower than last_buy and rule OK")
                    if do and qty>0:
                        execs.append(dict(side="BUY", symbol=b["symbol"], shares=qty, price=px, reason="AVERAGE"))
            submitted = st.form_submit_button("💾 Update portfolio")
            if submitted:
                # Trim sells to cap
                sells_today2 = 0
                if not ledger.empty:
                    ld = ledger.copy(); ld["date"] = pd.to_datetime(ld["date"]).dt.date
                    sells_today2 = int(((ld["date"]==now_ist().date()) & (ld["side"].str.upper()=="SELL")).sum())
                cap_left2 = max(0, cfg["max_sells_per_day"] - sells_today2)
                sells_exec = [e for e in execs if e["side"]=="SELL"][:cap_left2]
                buys_exec  = [e for e in execs if e["side"]=="BUY"]
                all_exec = []
                for e in sells_exec + buys_exec:
                    e["date"] = str(now_ist().date())
                    e["fee"] = round(e["shares"]*e["price"]*cfg["fee"], 2)
                    all_exec.append(e)
                if all_exec:
                    apply_trade_rows(sh, all_exec, cfg["fee"])
                    st.success("Portfolio updated.")
                else:
                    st.info("No trades to apply.")
                # Reload after update
                balances, positions, ledger, config, daily_eq = load_all(sh)

with tab2:
    st.subheader("My Portfolio")
    # Last close valuation snapshot
    last_close = mkt["prices"].iloc[-1]
    holdings_df, mv = position_snapshot(positions, last_close)
    cash = float(balances.iloc[0]["cash"]) if not balances.empty else DEFAULTS["base_capital"]
    equity_close = cash + mv
    exposure = (mv/equity_close) if equity_close>0 else 0.0
    fees_paid = float(balances.iloc[0]["fees_paid"]) if not balances.empty else 0.0
    realized = float(balances.iloc[0]["realized"]) if not balances.empty else 0.0

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Cash (₹)", f"{cash:,.0f}")
    with c2: st.metric("Equity (last close, ₹)", f"{equity_close:,.0f}")
    with c3: st.metric("Exposure", f"{exposure*100:.1f}%")
    with c4: st.metric("Realized PnL (₹)", f"{realized:,.0f}")
    with c5: st.metric("Fees paid (₹)", f"{fees_paid:,.0f}")

    st.markdown("#### Holdings (valued at last close)")
    st.dataframe(holdings_df, use_container_width=True, height=360)

    # Daily equity series (already backfilled)
    deq = daily_eq.copy()
    if not deq.empty:
        deq["date"] = pd.to_datetime(deq["date"])
        deq = deq.sort_values("date")
        # TWR and CAGR
        twr_total, twr_cagr = compute_twr_cagr(deq, ledger)
        # XIRR (money-weighted)
        flows=[]
        if not ledger.empty:
            led = ledger.copy(); led["date"]=pd.to_datetime(led["date"]).dt.date
            for _, r in led.iterrows():
                if (r["side"] or "").upper()=="FUND_IN":
                    flows.append((r["date"], -float(r["price"])))
                elif (r["side"] or "").upper()=="FUND_OUT":
                    flows.append((r["date"], +float(r["price"])))
        # Add current equity as terminal positive cash flow
        flows = sorted(flows, key=lambda x: x[0])
        if not deq.empty:
            flows.append((deq["date"].iloc[-1].date(), float(deq["equity"].iloc[-1])))
        xirr = compute_xirr(flows) if len(flows)>=2 else np.nan

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("TWR (total)", f"{(twr_total or 0)*100:.2f}%")
        with c2: st.metric("CAGR (TWR)", f"{(twr_cagr or 0)*100:.2f}%")
        with c3: st.metric("XIRR (money-weighted)", f"{(xirr or 0)*100:.2f}%")

        # Charts
        st.markdown("#### Charts")
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(deq["date"], deq["equity"]/deq["equity"].iloc[0], label="Strategy (₹ normalized)")
        # TRI
        if tri_ser is not None:
            tri_aligned = tri_ser.reindex(deq["date"]).ffill()
            ax.plot(deq["date"], tri_aligned/tri_aligned.iloc[0], label=tri_label, alpha=0.85)
        ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig, use_container_width=True)

        # Drawdown
        s = deq.set_index("date")["equity"]
        dd = s / s.cummax() - 1
        fig2, ax2 = plt.subplots(figsize=(8,2.5))
        ax2.fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.35)
        ax2.set_title("Drawdown"); ax2.grid(alpha=0.3)
        st.pyplot(fig2, use_container_width=True)

        # Exposure
        fig3, ax3 = plt.subplots(figsize=(8,2.5))
        ax3.plot(deq["date"], deq["exposure"].astype(float))
        ax3.set_ylim(0,1.1); ax3.set_title("Exposure"); ax3.grid(alpha=0.3)
        st.pyplot(fig3, use_container_width=True)

        # Rolling 1Y CAGR (YoY)
        if len(deq)>=252:
            s = deq.set_index("date")["equity"].astype(float)
            roll = (s / s.shift(252) - 1).dropna()
            fig4, ax4 = plt.subplots(figsize=(8,2.5))
            ax4.plot(roll.index, roll.values)
            ax4.set_title("Rolling 1Y return (approx CAGR)"); ax4.grid(alpha=0.3)
            st.pyplot(fig4, use_container_width=True)

        # Downloads
        st.markdown("#### Downloads")
        st.download_button("Download trade_ledger.csv", data=ledger.to_csv(index=False), file_name="trade_ledger.csv", mime="text/csv")
        st.download_button("Download holdings.csv", data=holdings_df.to_csv(index=False), file_name="holdings.csv", mime="text/csv")
        # daily_summary = daily_equity joined with NIFTY100 TRI (if available)
        daily_summary = deq.copy()
        if tri_ser is not None:
            tri_df = tri_ser.rename("NIFTY100_fallback").to_frame()
            tri_df.index = pd.to_datetime(tri_df.index)
            j = daily_summary.set_index("date").join(tri_df, how="left").reset_index()
        else:
            j = daily_summary
        st.download_button("Download daily_summary.csv", data=j.to_csv(index=False), file_name="daily_summary.csv", mime="text/csv")
        st.download_button("Download equity_series.csv", data=deq[["date","equity"]].to_csv(index=False), file_name="equity_series.csv", mime="text/csv")
    else:
        st.info("No daily equity yet. Execute a trade or add funds to start the series.")


