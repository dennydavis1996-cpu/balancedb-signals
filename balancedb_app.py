# ==================================================
# Balanced_B Signals — NIFTY100
# Streamlit App with Google Sheets Integration
# FULL STRATEGY LOGIC (identical to backtest engine)
# ==================================================

import streamlit as st
import pandas as pd, numpy as np
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

# ----------------- DEFAULT PARAMS -----------------
DEFAULTS = dict(
    base_capital=500000,
    fee=0.0011,
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
    divisor=30,
    divisor_bear=38,
    lookback_days=420,
    min_turnover_cr=8.0,
    turnover_window=20,
    apply_turnover_to_averaging=True
)

NIFTY_INDEX = "^NSEI"
N100_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"

# ----------------- GOOGLE SHEETS -----------------
def _service_account():
    creds_dict = st.secrets["gcp_service_account"]
    scope=["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds=ServiceAccountCredentials.from_json_keyfile_dict(creds_dict,scope)
    return gspread.authorize(creds)

def open_sheet(url): return _service_account().open_by_url(url)

def ensure_tabs(sh):
    schemas={
        "balances":["cash","base_capital","realized","fees_paid","last_update"],
        "positions":["symbol","shares","avg_cost","last_buy","open_date","open_idx"],
        "ledger":["date","side","symbol","shares","price","gross","fee","cash_before","cash_after","realized_pnl","holding_days","type"],
        "config":["param","value"],
        "daily_equity":["date","equity","cash","invested","exposure","positions","fees_cum","benchmark"],
    }
    for tab,cols in schemas.items():
        try: sh.worksheet(tab)
        except gspread.exceptions.WorksheetNotFound:
            ws=sh.add_worksheet(title=tab,rows=1000,cols=len(cols)); ws.append_row(cols)

def read_tab(sh,tab): return pd.DataFrame(sh.worksheet(tab).get_all_records())
def save_df(sh,tab,df):
    ws=sh.worksheet(tab); ws.clear()
    if not df.empty: ws.update([df.columns.values.tolist()]+df.astype(str).values.tolist())
    else: ws.update([[]])

# ----------------- MARKET DATA -----------------
@st.cache_data(ttl=300)
def safe_yf_download(symbols,start,end):
    try: return yf.download(symbols,start=start,end=end,progress=False,auto_adjust=False)
    except Exception as e: st.warning(f"⚠️ Yahoo error: {e}"); return pd.DataFrame()

def fetch_nifty100_symbols():
    try: df=pd.read_csv(N100_URL); return [s+".NS" for s in df["Symbol"].dropna()]
    except Exception: st.error("NIFTY100 list fetch failed."); return []

# ----------------- SIGNAL GENERATION -----------------
def compute_signals(prices, vols, bench, config, positions, balances):
    ma=prices.rolling(config["ma"]).mean()
    std=prices.rolling(config["ma"]).std()
    bench_ma=bench.rolling(config["regime_filter_ma"]).mean()
    regime_ok = bench.iloc[-1] >= bench_ma.iloc[-1]*(1+config["regime_buffer"])

    turnover=(prices*vols)/1e7
    med_turn=turnover.rolling(config["turnover_window"]).median()

    last=prices.iloc[-1]; ma_last=ma.iloc[-1]; std_last=std.iloc[-1]; liq_today=med_turn.iloc[-1]

    signals={"SELL":[],"TIME_STOP":[],"NEW":[],"AVERAGE":[]}

    # --- SELL (TP + time stop) obeying max_sells_per_day ---
    sells=[]
    for sym,pos in positions.iterrows():
        price=last.get(sym,np.nan)
        if pd.isna(price): continue
        r=price/pos["avg_cost"]-1
        age=(datetime.today().date()-pd.to_datetime(pos["open_date"]).date()).days
        if r>=config["take_profit"]: sells.append((r,sym,price,"TP",age))
        elif age>=config["time_stop_days"]: sells.append((r,sym,price,"TIME_STOP",age))
    sells=sorted(sells,key=lambda x:x[0],reverse=True)[:config["max_sells_per_day"]]
    for r,sym,price,typ,age in sells: signals[typ if typ=="TIME_STOP" else "SELL"].append((sym,price,typ,age))

    # --- BUY eligibility ---
    def liq_ok(sym): return liq_today.get(sym,np.nan)>=config["min_turnover_cr"]
    elig=[c for c in prices.columns if pd.notna(ma_last.get(c)) and pd.notna(last[c]) and last[c]<ma_last[c] and liq_ok(c)]

    if config["use_zscore"]:
        zmap={c:(last[c]-ma_last[c])/std_last[c] for c in elig if pd.notna(std_last.get(c)) and std_last[c]>0}
        ranked=sorted(zmap,key=zmap.get)[:config["bottom_n"]]
    else:
        dist=last/ma_last-1; ranked=sorted(elig,key=lambda c:dist[c])[:config["bottom_n"]]

    if regime_ok: # new buys allowed only if bull regime
        new_buys=[c for c in ranked if c not in positions.index][:config["max_new_buys"]]
        for sym in new_buys: signals["NEW"].append((sym,last[sym],"NEW"))

    # --- Averaging ---
    for sym,pos in positions.iterrows():
        price=last.get(sym,np.nan)
        if pd.isna(price): continue
        if config["apply_turnover_to_averaging"] and not liq_ok(sym): continue
        if regime_ok or (not config["averaging_requires_regime"]):
            if price<=pos["last_buy"]*(1-config["avg_dd"]):
                if not regime_ok:
                    m=ma_last.get(sym); s=std_last.get(sym)
                    if pd.notna(m) and pd.notna(s) and s>0 and (price-m)/s <= config["avg_in_bear_z_thresh"]:
                        signals["AVERAGE"].append((sym,price,"AVERAGE_BEAR"))
                else:
                    signals["AVERAGE"].append((sym,price,"AVERAGE"))

    # --- Lot size from divisor ---
    cap=float(balances.at[0,"base_capital"])+float(balances.at[0,"realized"])
    lot_cash=cap/(config["divisor"] if regime_ok else config["divisor_bear"])

    return signals,regime_ok,lot_cash

# ----------------- PORTFOLIO VALUATION -----------------
def rebuild_equity(prices,bench,positions,balances,ledger):
    equity=[]; cash=balances.at[0,"cash"]; realized=balances.at[0,"realized"]; fees=balances.at[0,"fees_paid"]
    daily=[]
    for i,d in enumerate(prices.index):
        row=prices.loc[d]; mv=0
        for sym,pos in positions.iterrows():
            if sym in row and not pd.isna(row[sym]):
                mv+=row[sym]*pos["shares"]
        eq=cash+realized+mv
        equity.append(eq)
        exposure=mv/eq if eq>0 else 0
        daily.append([str(d.date()),eq,cash,mv,exposure,len(positions),fees,bench.get(d,np.nan) if bench is not None else np.nan])
    return pd.DataFrame(daily,columns=["date","equity","cash","invested","exposure","positions","fees_cum","benchmark"]).set_index("date")

def compute_drawdown(eq): return eq/eq.cummax()-1
def compute_cagr(eq):
    start,end=eq.iloc[0],eq.iloc[-1]; years=(len(eq)/252)
    return (end/start)**(1/years)-1 if years>0 else np.nan
def compute_sharpe(rets): return (rets.mean()*252)/(rets.std()*np.sqrt(252)) if rets.std()>0 else np.nan

# ----------------- UI -----------------
with st.sidebar:
    st.header("⚙️ Settings")
    sheet_url=st.text_input("Google Sheet URL","")
    run_button=st.button("🔄 Run Scan")

if sheet_url:
    sh=open_sheet(sheet_url); ensure_tabs(sh)

    balances=read_tab(sh,"balances")
    if balances.empty:
        balances=pd.DataFrame([{"cash":DEFAULTS["base_capital"],"base_capital":DEFAULTS["base_capital"],"realized":0,"fees_paid":0,"last_update":str(datetime.today().date())}])
        save_df(sh,"balances",balances)

    positions=read_tab(sh,"positions").set_index("symbol") if not read_tab(sh,"positions").empty else pd.DataFrame(columns=["shares","avg_cost","last_buy","open_date","open_idx"]).set_index(pd.Index([]))
    config_df=read_tab(sh,"config"); config=DEFAULTS.copy()
    for _,r in config_df.iterrows():
        if r["param"] in config: config[r["param"]]=float(r["value"])

    tickers=fetch_nifty100_symbols(); end=datetime.today().date(); start=end-timedelta(days=int(config["lookback_days"]))
    data=safe_yf_download(tickers+[NIFTY_INDEX],start,end)
    prices=data["Adj Close"].dropna(axis=1,how="all") if not data.empty else pd.DataFrame()
    vols=data["Volume"] if "Volume" in data else pd.DataFrame()
    bench=prices[NIFTY_INDEX] if NIFTY_INDEX in prices else pd.Series()

    tabs=st.tabs(["📊 Run Signals","💼 Portfolio","📑 Reports"])

    with tabs[0]:
        st.subheader("Signals")
        if run_button and not prices.empty:
            sigs,regime_ok,lot_cash=compute_signals(prices,vols,bench,config,positions,balances)
            st.write("Regime:", "Bullish ✅" if regime_ok else "Bearish ⚠️")
            st.write("Lot Size (₹)",round(lot_cash,2))
            st.write("SELL/TS",pd.DataFrame(sigs["SELL"]+sigs["TIME_STOP"],columns=["symbol","price","reason","age"]))
            st.write("NEW BUY",pd.DataFrame(sigs["NEW"],columns=["symbol","price","reason"]))
            st.write("AVERAGE",pd.DataFrame(sigs["AVERAGE"],columns=["symbol","price","reason"]))

    with tabs[1]:
        st.subheader("Portfolio")
        if not positions.empty and not prices.empty:
            snaps=[]; last=prices.iloc[-1]
            for sym,pos in positions.iterrows():
                price=last.get(sym,np.nan)
                if pd.isna(price): continue
                mv=price*pos["shares"]; unr=(price-pos["avg_cost"])*pos["shares"]
                snaps.append([sym,pos["shares"],pos["avg_cost"],price,mv,unr,(price/pos["avg_cost"]-1)*100])
            df=pd.DataFrame(snaps,columns=["symbol","shares","avg_cost","last_price","market_value","unrealized_pnl","unrealized_pct"])
            st.dataframe(df.set_index("symbol"))
            eqdf=rebuild_equity(prices,bench,positions,balances,read_tab(sh,"ledger"))
            st.line_chart(eqdf[["equity","benchmark"]].dropna())
            st.area_chart(compute_drawdown(eqdf["equity"]))
            rets=eqdf["equity"].pct_change().dropna()
            st.write(f"CAGR: {compute_cagr(eqdf['equity']):.2%}, Sharpe: {compute_sharpe(rets):.2f}, MaxDD: {compute_drawdown(eqdf['equity']).min():.2%}")

    with tabs[2]:
        st.subheader("Reports")
        ledger=read_tab(sh,"ledger")
        if not ledger.empty:
            sells=ledger[ledger["side"]=="SELL"]
            if not sells.empty:
                sells["pnl_pct"]=sells["realized_pnl"]/(sells["gross"]).replace(0,np.nan)
                fig,ax=plt.subplots(figsize=(6,3))
                ax.hist(sells["pnl_pct"].dropna()*100,bins=40,color="tab:blue",alpha=0.7)
                ax.set_title("Realized Trade PnL %"); st.pyplot(fig)
