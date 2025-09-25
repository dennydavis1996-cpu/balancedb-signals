# ==================================================
# Balanced_B Signals — NIFTY100
# Streamlit App with Google Sheets Integration
# (Full Strategy Logic Ported from Backtest)
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
    ma=3,
    bottom_n=16,
    max_new_buys=3,
    avg_dd=0.035,
    take_profit=0.09,
    max_sells_per_day=4,
    time_stop_days=140,
    regime_filter_ma=60,
    regime_buffer=0.003,
    divisor=30,
    divisor_bear=38,
    lookback_days=420,
    min_turnover_cr=0.0,   # 20D median turnover filter in ₹ Cr
    turnover_window=20,
    avg_in_bear_z_thresh=-1.8
)

NIFTY_INDEX = "^NSEI"
N100_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"

# ----------------- GOOGLE SHEETS HELPERS -----------------
def _service_account():
    creds_dict = st.secrets["gcp_service_account"]
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    return gspread.authorize(creds)

def open_sheet(url): return _service_account().open_by_url(url)

def ensure_tabs(sh):
    schemas = {
        "balances":["cash","base_capital","realized","fees_paid","last_update"],
        "positions":["symbol","shares","avg_cost","last_buy","open_date"],
        "ledger":["date","side","symbol","shares","price","fee","reason","realized_pnl"],
        "config":["param","value"],
        "daily_equity":["date","equity","cash","invested","exposure","source"],
    }
    for tab,cols in schemas.items():
        try: sh.worksheet(tab)
        except gspread.exceptions.WorksheetNotFound:
            ws=sh.add_worksheet(title=tab,rows=1000,cols=len(cols))
            ws.append_row(cols)

def read_tab(sh,tab): return pd.DataFrame(sh.worksheet(tab).get_all_records())
def save_df(sh,tab,df):
    ws=sh.worksheet(tab); ws.clear()
    if not df.empty: ws.update([df.columns.values.tolist()]+df.values.tolist())
    else: ws.update([[]])

# ----------------- MARKET DATA -----------------
@st.cache_data(ttl=300)
def safe_yf_download(symbols,start,end):
    try: return yf.download(symbols,start=start,end=end,progress=False,auto_adjust=False)
    except Exception as e: st.warning(f"⚠️ Yahoo Finance error: {e}"); return pd.DataFrame()

def fetch_nifty100_symbols():
    try:
        df=pd.read_csv(N100_URL)
        syms=sorted(df["Symbol"].dropna().astype(str).str.upper().tolist())
        return [s+".NS" for s in syms]
    except Exception: st.error("NIFTY100 list fetch failed."); return []

# ----------------- SIGNAL LOGIC -----------------
def compute_signals(prices, vols, bench, config, positions):
    """
    Full Balanced_B signal logic:
    - Regime filter (bull/bear)
    - Liquidity filter
    - Profit-taking / Time stops
    - New buys ranked by z-score
    - Averaging rules
    """
    ma = prices.rolling(config["ma"]).mean()
    std = prices.rolling(config["ma"]).std()

    bench_ma = bench.rolling(config["regime_filter_ma"]).mean()
    regime_ok = bench.iloc[-1] >= bench_ma.iloc[-1]*(1+config["regime_buffer"])

    turnover = (prices*vols)/1e7  # ₹ Cr
    med_turnover = turnover.rolling(config["turnover_window"]).median()

    last = prices.iloc[-1]; last_ma = ma.iloc[-1]; last_std = std.iloc[-1]
    liq_today = med_turnover.iloc[-1]

    signals={"SELL":[],"TIME_STOP":[],"NEW":[],"AVERAGE":[]}

    # ---- SELL logic (TP + Time stops) ----
    for sym,pos in positions.iterrows():
        if sym not in last: continue
        price=last[sym]
        if pd.isna(price): continue
        r = price/pos["avg_cost"]-1
        age=(datetime.today().date()-pd.to_datetime(pos["open_date"]).date()).days
        if r>=config["take_profit"]:
            signals["SELL"].append((sym,price,"TP"))
        elif age>=config["time_stop_days"]:
            signals["TIME_STOP"].append((sym,price,"TIME_STOP"))

    # ---- BUY logic ----
    def liq_ok(sym): return liq_today.get(sym,np.nan)>=config["min_turnover_cr"]
    elig=[c for c in prices.columns if pd.notna(last_ma.get(c)) and pd.notna(last[c]) and last[c]<last_ma[c] and liq_ok(c)]

    if config.get("use_zscore",True):
        zmap={c:(last[c]-last_ma[c])/last_std[c] for c in elig if pd.notna(last_std.get(c)) and last_std[c]>0}
        ranked=sorted(zmap,key=zmap.get)[:config["bottom_n"]]
    else:
        dist=last/last_ma-1
        ranked=sorted(elig,key=lambda c:dist[c])[:config["bottom_n"]]

    # New buys capped
    new_buys=[c for c in ranked if c not in positions.index][:config["max_new_buys"]]
    if regime_ok:
        for sym in new_buys: signals["NEW"].append((sym,last[sym],"NEW"))

    # Averaging logic
    for sym,pos in positions.iterrows():
        if sym not in last: continue
        price=last[sym]
        if pd.isna(price): continue
        if not liq_ok(sym): continue
        if regime_ok:
            if price<=pos["last_buy"]*(1-config["avg_dd"]):
                signals["AVERAGE"].append((sym,price,"AVG"))
        else:
            m=last_ma.get(sym); s=last_std.get(sym)
            if pd.notna(m) and pd.notna(s) and s>0:
                z=(price-m)/s
                if z<=config["avg_in_bear_z_thresh"] and price<=pos["last_buy"]*(1-config["avg_dd"]):
                    signals["AVERAGE"].append((sym,price,"AVG_BEAR"))
    return signals,regime_ok

# ----------------- METRICS -----------------
def position_snapshot(prices,positions):
    last=prices.iloc[-1]; snaps=[]
    for sym,pos in positions.iterrows():
        if sym not in last: continue; price=last[sym]; 
        price=last[sym]
        mv=price*pos["shares"]; unr=(price-pos["avg_cost"])*pos["shares"]
        unrpct=(price/pos["avg_cost"]-1)*100
        snaps.append([sym,pos["shares"],pos["avg_cost"],price,mv,unr,unrpct])
    return pd.DataFrame(snaps,columns=["symbol","shares","avg_cost","last_price","market_value","unrealized_pnl","unrealized_pct"]).set_index("symbol") if snaps else pd.DataFrame(columns=["symbol","shares","avg_cost","last_price","market_value","unrealized_pnl","unrealized_pct"])

def compute_drawdown(eq): return eq/eq.cummax()-1
def compute_cagr(eq): return (eq.iloc[-1]/eq.iloc[0])**(365.25/((eq.index[-1]-eq.index[0]).days))-1 if len(eq)>1 else np.nan
def compute_sharpe(rets): return (np.sqrt(252)*rets.mean()/rets.std()) if rets.std()>0 else np.nan

def reconstruct_daily_equity(prices,balances,positions):
    eq=[]
    for d in prices.index:
        mv=0
        for sym,pos in positions.iterrows():
            if sym in prices.columns and not pd.isna(prices.loc[d,sym]):
                mv+=prices.loc[d,sym]*pos["shares"]
        cash=float(balances.at[0,"cash"]); realized=float(balances.at[0,"realized"])
        eq.append(mv+cash+realized)
    return pd.Series(eq,index=prices.index,name="Equity")

# ----------------- UI -----------------
with st.sidebar:
    st.header("⚙️ Settings")
    sheet_url=st.text_input("Google Sheet URL",value="")
    run_button=st.button("🔄 Run Scan")

if sheet_url:
    sh=open_sheet(sheet_url); ensure_tabs(sh)

    # ---- Load tabs ----
    balances=read_tab(sh,"balances")
    if balances.empty:
        balances=pd.DataFrame([{"cash":DEFAULTS["base_capital"],"base_capital":DEFAULTS["base_capital"],"realized":0,"fees_paid":0,"last_update":str(datetime.today().date())}])
        save_df(sh,"balances",balances)
    positions=read_tab(sh,"positions").set_index("symbol") if not read_tab(sh,"positions").empty else pd.DataFrame(columns=["shares","avg_cost","last_buy","open_date"]).set_index(pd.Index([]))
    config_df=read_tab(sh,"config"); config=DEFAULTS.copy()
    for _,r in config_df.iterrows():
        if r["param"] in config: config[r["param"]]=float(r["value"])

    # ---- Market data ----
    tickers=fetch_nifty100_symbols()
    end=datetime.today().date(); start=end-timedelta(days=int(config["lookback_days"]))
    data=safe_yf_download(tickers+[NIFTY_INDEX],start,end)
    prices=data["Adj Close"].dropna(axis=1,how="all") if not data.empty else pd.DataFrame()
    vols=data["Volume"] if "Volume" in data else pd.DataFrame()
    bench=prices[NIFTY_INDEX] if NIFTY_INDEX in prices else pd.Series()

    tabs=st.tabs(["📊 Run Signals","💼 My Portfolio","📑 Reports & Analytics"])

    with tabs[0]:
        st.subheader("📊 Signals")
        if run_button and not prices.empty:
            sigs,regime_ok=compute_signals(prices,vols,bench,config,positions)
            st.write("📈 Regime:", "Bullish ✅" if regime_ok else "Bearish ⚠️")
            st.markdown("### 🔴 SELL / TIME STOP Signals")
            st.write(pd.DataFrame(sigs["SELL"]+sigs["TIME_STOP"],columns=["symbol","price","reason"]))
            st.markdown("### 🟢 NEW BUY")
            st.write(pd.DataFrame(sigs["NEW"],columns=["symbol","price","reason"]))
            st.markdown("### 🔵 AVERAGE")
            st.write(pd.DataFrame(sigs["AVERAGE"],columns=["symbol","price","reason"]))

    with tabs[1]:
        st.subheader("💼 Portfolio Snapshot")
        snaps=position_snapshot(prices,positions) if not prices.empty else pd.DataFrame()
        if not snaps.empty:
            st.dataframe(snaps.style.bar(subset=["unrealized_pct"],align="mid",color=["red","green"]))
            equity=reconstruct_daily_equity(prices,balances,positions)
            bench_norm=(bench/bench.iloc[0])*equity.iloc[0] if not bench.empty else None
            st.line_chart(pd.DataFrame({"Equity":equity,"NIFTY50":bench_norm}))
            st.area_chart(compute_drawdown(equity))
            rets=equity.pct_change().dropna()
            st.write(f"**CAGR:** {compute_cagr(equity):.2%} | **Sharpe:** {compute_sharpe(rets):.2f} | **MaxDD:** {compute_drawdown(equity).min():.2%}")
        else: st.info("No positions yet")

    with tabs[2]:
        st.subheader("📑 Reports")
        ledger=read_tab(sh,"ledger"); equity=reconstruct_daily_equity(prices,balances,positions)
        if not equity.empty:
            rets=equity.pct_change().dropna()
            st.write("Performance Summary",pd.DataFrame({
                "CAGR":[f"{compute_cagr(equity):.2%}"],
                "Sharpe":[f"{compute_sharpe(rets):.2f}"],
                "MaxDD":[f"{compute_drawdown(equity).min():.2%}"],
                "Fees Paid":[balances.at[0,"fees_paid"]]
            }))
            # exposure
            st.line_chart((equity-(float(balances.at[0,"cash"])+float(balances.at[0,"realized"])))/equity)
            if not ledger.empty:
                sells=ledger[ledger["side"]=="SELL"]
                if not sells.empty and "realized_pnl" in sells:
                    sells["pnl_pct"]=sells["realized_pnl"]/(sells["price"]*sells["shares"]).replace(0,np.nan)
                    fig,ax=plt.subplots(figsize=(6,3))
                    ax.hist(sells["pnl_pct"].dropna()*100,bins=40,color="tab:blue",alpha=0.7)
                    ax.set_title("Realized PnL Distribution (%)"); st.pyplot(fig)
                if len(equity)>252: st.line_chart(((equity/equity.shift(252))-1).dropna())

