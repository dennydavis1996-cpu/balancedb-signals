# ==================================================
# Balanced_B Signals — NIFTY100
# Streamlit App with Google Sheets Integration
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
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

NIFTY_INDEX = "^NSEI"
N100_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"

# ----------------- GOOGLE SHEETS HELPERS -----------------
def _service_account():
    """ Returns authenticated gspread client using Streamlit secrets (not file). """
    creds_dict = st.secrets["gcp_service_account"]
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    return gspread.authorize(creds)

def open_sheet(url):
    return _service_account().open_by_url(url)

def ensure_tabs(sh):
    """Ensure required tabs exist and headers set"""
    schemas = {
        "balances": ["cash","base_capital","realized","fees_paid","last_update"],
        "positions": ["symbol","shares","avg_cost","last_buy","open_date"],
        "ledger": ["date","side","symbol","shares","price","fee","reason","realized_pnl"],
        "config": ["param","value"],
        "daily_equity": ["date","equity","cash","invested","exposure","source"],
    }
    for tab, cols in schemas.items():
        try:
            sh.worksheet(tab)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=tab, rows=1000, cols=len(cols))
            ws.append_row(cols)

def read_tab(sh, tab):
    ws = sh.worksheet(tab)
    df = pd.DataFrame(ws.get_all_records())
    return df

def save_df(sh, tab, df):
    ws = sh.worksheet(tab)
    ws.clear()
    if not df.empty:
        ws.update([df.columns.values.tolist()] + df.values.tolist())
    else:
        ws.update([[]])

# ----------------- TRADE RECORDING -----------------
def apply_trade_rows(sh, trades, balances, positions):
    ledger_df = read_tab(sh, "ledger")
    if ledger_df.empty:
        ledger_df = pd.DataFrame(columns=["date","side","symbol","shares","price","fee","reason","realized_pnl"])
    else:
        ledger_df = pd.DataFrame(ledger_df)

    for trade in trades:
        cash = float(balances.at[0, "cash"]) if "cash" in balances else DEFAULTS["base_capital"]
        realized = float(balances.at[0, "realized"]) if "realized" in balances else 0
        fees = float(balances.at[0, "fees_paid"]) if "fees_paid" in balances else 0

        shares = int(trade["shares"])
        fee_amt = float(trade["fee"])
        cost = trade["price"]*shares

        if trade["side"]=="BUY":
            cash -= cost+fee_amt
            if trade["symbol"] in positions.index:
                pos = positions.loc[trade["symbol"]]
                new_shares = pos["shares"]+shares
                new_avg = (pos["avg_cost"]*pos["shares"] + cost)/new_shares
                positions.loc[trade["symbol"], "shares"] = new_shares
                positions.loc[trade["symbol"], "avg_cost"] = new_avg
                positions.loc[trade["symbol"], "last_buy"] = trade["price"]
            else:
                positions.loc[trade["symbol"]] = [shares, trade["price"], trade["price"], trade["date"]]
        elif trade["side"]=="SELL":
            if trade["symbol"] in positions.index:
                pos = positions.loc[trade["symbol"]]
                cash += cost - fee_amt
                pnl = (trade["price"] - pos["avg_cost"])*shares - fee_amt
                realized += pnl
                positions.loc[trade["symbol"], "shares"] -= shares
                if positions.loc[trade["symbol"], "shares"]<=0:
                    positions.drop(trade["symbol"], inplace=True)

        fees += fee_amt
        balances.at[0,"cash"] = cash
        balances.at[0,"realized"] = realized
        balances.at[0,"fees_paid"] = fees
        balances.at[0,"last_update"] = str(trade["date"])
        ledger_df = pd.concat([ledger_df, pd.DataFrame([trade])], ignore_index=True)
    
    save_df(sh, "balances", balances)
    save_df(sh, "positions", positions.reset_index().rename(columns={"index":"symbol"}))
    save_df(sh, "ledger", ledger_df)

# ----------------- MARKET DATA -----------------
@st.cache_data(ttl=300)
def safe_yf_download(symbols, start, end):
    try:
        data = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=False)
        return data
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance error: {e}")
        return pd.DataFrame()

def fetch_nifty100_symbols():
    try:
        df = pd.read_csv(N100_URL)
        symbols = sorted(df["Symbol"].dropna().astype(str).str.upper().tolist())
        return [s+".NS" for s in symbols]
    except Exception:
        st.error("Could not fetch NIFTY100 list. Returning empty list.")
        return []

# ----------------- SIGNAL COMPUTATION -----------------
def compute_signals(prices, config, positions):
    ma = prices.rolling(config["ma"]).mean()
    std = prices.rolling(config["ma"]).std()
    last = prices.iloc[-1]
    last_ma = ma.iloc[-1]
    last_std = std.iloc[-1]
    signals = {"SELL":[],"NEW":[],"AVERAGE":[]}
    for sym, pos in positions.iterrows():
        if sym not in last: continue
        price = last[sym]
        if price/pos["avg_cost"]-1 >= config["take_profit"]:
            signals["SELL"].append((sym, price, "TP"))
    elig = [c for c in prices.columns if (c in last_ma) and (last[c]<last_ma[c]) and not np.isnan(last_ma[c])]
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

# ----------------- SNAPSHOT & METRICS -----------------
def position_snapshot(prices, positions):
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

def compute_drawdown(equity):
    return equity/equity.cummax()-1

def compute_cagr(equity):
    start_val, end_val = equity.iloc[0], equity.iloc[-1]
    years = (equity.index[-1]-equity.index[0]).days/365.25
    return (end_val/start_val)**(1/years)-1 if years>0 else np.nan

def compute_sharpe(returns):
    mean, std = returns.mean(), returns.std()
    return np.sqrt(252)*mean/std if std>0 else np.nan

def reconstruct_daily_equity(prices, balances, positions, ledger):
    if prices.empty:
        return pd.Series()
    equity=[]
    for d in prices.index:
        mv = 0.0
        for sym,pos in positions.iterrows():
            if sym in prices.columns and not np.isnan(prices.loc[d,sym]):
                mv += prices.loc[d,sym]*pos["shares"]
        cash = float(balances.at[0, "cash"]) if "cash" in balances else DEFAULTS["base_capital"]
        realized = float(balances.at[0, "realized"]) if "realized" in balances else 0
        equity.append(mv + cash + realized)
    return pd.Series(equity, index=prices.index, name="Equity")

# ----------------- STREAMLIT APP -----------------
with st.sidebar:
    st.header("⚙️ Settings")
    sheet_url = st.text_input("Google Sheet URL", value="")
    run_button = st.button("🔄 Run Scan")

if sheet_url:
    sh = open_sheet(sheet_url)
    ensure_tabs(sh)

    balances = read_tab(sh,"balances")
    if balances.empty:
        balances = pd.DataFrame([{"cash":DEFAULTS["base_capital"],"base_capital":DEFAULTS["base_capital"],"realized":0,"fees_paid":0,"last_update":str(datetime.today().date())}])
        save_df(sh,"balances",balances)

    positions = read_tab(sh,"positions").set_index("symbol") if not read_tab(sh,"positions").empty else pd.DataFrame(columns=["shares","avg_cost","last_buy","open_date"]).set_index(pd.Index([]))
    config_df = read_tab(sh,"config")
    config = DEFAULTS.copy()
    for _,row in config_df.iterrows():
        if row["param"] in config: config[row["param"]] = float(row["value"])

    tickers = fetch_nifty100_symbols()
    end = datetime.today().date()
    start = end - timedelta(days=int(config["lookback_days"]))
    data = safe_yf_download(tickers, start, end)
    prices = data["Adj Close"] if not data.empty else pd.DataFrame()

    tabs = st.tabs(["📊 Run Signals", "💼 My Portfolio", "📑 Reports & Analytics"])

    with tabs[0]:
        st.subheader("📊 Trade Signals")
        if run_button and not prices.empty:
            sigs = compute_signals(prices, config, positions)
            st.success("Signals generated ✅")
            st.markdown("### 🔴 SELL Signals")
            st.write(pd.DataFrame(sigs["SELL"], columns=["symbol","price","reason"]))
            st.markdown("### 🟢 NEW BUY Signals")
            st.write(pd.DataFrame(sigs["NEW"], columns=["symbol","price","reason"]))
            st.markdown("### 🔵 AVERAGING Signals")
            st.write(pd.DataFrame(sigs["AVERAGE"], columns=["symbol","price","reason"]))

        st.markdown("### 📝 Record Trades")
        with st.form("trade_form"):
            trade_side = st.selectbox("Side", ["BUY","SELL"])
            trade_symbol = st.text_input("Symbol (Yahoo format e.g. RELIANCE.NS)")
            trade_shares = st.number_input("Shares", value=0, step=1)
            trade_price = st.number_input("Price", value=0.0)
            trade_reason = st.text_input("Reason", "")
            submitted = st.form_submit_button("Record Trade")
            if submitted and trade_symbol and trade_shares>0:
                trade = dict(
                    date=str(datetime.today().date()),
                    side=trade_side,symbol=trade_symbol,shares=int(trade_shares),
                    price=float(trade_price),
                    fee=float(trade_price*trade_shares*config["fee"]),
                    reason=trade_reason,
                    realized_pnl=0.0
                )
                apply_trade_rows(sh,[trade], balances, positions)
                st.success("✅ Trade recorded")

    with tabs[1]:
        st.subheader("💼 Portfolio Snapshot")
        snaps = position_snapshot(prices, positions) if not prices.empty else pd.DataFrame()
        if not snaps.empty:
            st.dataframe(snaps.style.bar(subset=["unrealized_pct"], align="mid", color=["red","green"]))
            ledger = read_tab(sh,"ledger")
            equity_series = reconstruct_daily_equity(prices, balances, positions, ledger)
            if not equity_series.empty:
                st.markdown("### 📈 Equity Curve")
                st.line_chart(equity_series)
                st.markdown("### 📉 Drawdown")
                st.area_chart(compute_drawdown(equity_series))
                rets = equity_series.pct_change().dropna()
                cagr = compute_cagr(equity_series)
                sharpe = compute_sharpe(rets)
                st.write(f"**CAGR:** {cagr:.2%} | **Sharpe:** {sharpe:.2f}")
                st.download_button("⬇️ Download Ledger CSV", ledger.to_csv(index=False), "ledger.csv")
                st.download_button("⬇️ Download Holdings CSV", snaps.to_csv(), "holdings.csv")
                st.download_button("⬇️ Download Equity CSV", equity_series.to_csv(), "equity.csv")
        else:
            st.info("No active positions yet")

    with tabs[2]:
        st.subheader("📑 Reports & Analytics")
        ledger = read_tab(sh,"ledger")
        equity_series = reconstruct_daily_equity(prices, balances, positions, ledger)

        if not equity_series.empty and not ledger.empty:
            rets = equity_series.pct_change().dropna()
            cagr = compute_cagr(equity_series)
            sharpe = compute_sharpe(rets)
            dd = compute_drawdown(equity_series).min()
            fees_paid = float(balances.at[0,"fees_paid"]) if "fees_paid" in balances else 0
            st.markdown("### 📋 Performance Summary")
            st.write(pd.DataFrame({
                "CAGR":[f"{cagr:.2%}"],
                "Sharpe":[f"{sharpe:.2f}"],
                "Max Drawdown":[f"{dd:.2%}"],
                "Fees Paid":[f"₹{fees_paid:,.0f}"]
            }))

            st.markdown("### 🎯 Realized PnL % Histogram")
            sells = ledger[ledger["side"]=="SELL"].copy()
            if not sells.empty and "realized_pnl" in sells:
                sells["pnl_pct"] = sells["realized_pnl"]/(sells["price"]*sells["shares"]).replace(0,np.nan)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.hist(sells["pnl_pct"].dropna()*100, bins=40, color="skyblue", edgecolor="black")
                ax.set_xlabel("PnL (%)"); ax.set_ylabel("Count")
                ax.set_title("Trade Return Distribution")
                st.pyplot(fig)

            st.markdown("### 📊 Exposure Over Time")
            exposure=[]
            for d in prices.index:
                mv_total=0.0
                for sym,pos in positions.iterrows():
                    if sym in prices.columns and not np.isnan(prices.loc[d,sym]):
                        mv_total += prices.loc[d,sym]*pos["shares"]
                eq = mv_total+float(balances.at[0,"cash"])+float(balances.at[0,"realized"])
                exp = mv_total/eq if eq>0 else 0
                exposure.append(exp)
            st.line_chart(pd.Series(exposure,index=prices.index,name="Exposure"))

            st.markdown("### 🔄 Rolling 1Y CAGR")
            if len(equity_series)>252:
                roll_cagr = (equity_series/equity_series.shift(252))-1
                st.line_chart(roll_cagr.dropna())
        else:
            st.info("Not enough data for reports yet.")

