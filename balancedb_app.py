# balancedb_app.py
# Balanced_B mobile signal terminal (NIFTY100, daily MTM, Google Sheets storage)

import os, re, json
from io import StringIO
from datetime import datetime, timedelta, date
import zoneinfo
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------- Safe Yahoo downloader ----------------
def safe_yf_download(tickers, **kwargs):
    try:
        df = yf.download(tickers, **kwargs)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        try:
            st.warning(f"Yahoo download failed for {tickers}: {e}")
        except Exception:
            print(f"Yahoo download failed for {tickers}: {e}")
        return pd.DataFrame()

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

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
    divisor=30,        
    divisor_bear=38,   
    lookback_days=420,
)

st.set_page_config(page_title="Balanced_B Signals (NIFTY100)", page_icon="📈", layout="wide")

# ---------------- Utilities ----------------
def now_ist(): return datetime.now(IST)
def is_weekday(d): return d.weekday() < 5
def is_time_between(t, start_hm=(9,15), end_hm=(15,30)):
    sh, sm = start_hm; eh, em = end_hm
    return (t.hour,t.minute) >= (sh,sm) and (t.hour,t.minute) <= (eh,em)

def is_market_open():
    t=now_ist()
    if not is_weekday(t): return False,"Weekend"
    if not is_time_between(t.time(), (9,15), (15,30)):
        return False,"Outside trading hours (IST 09:15–15:30)"
    try:
        s = requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0"})
        r = s.get("https://www.nseindia.com/api/marketStatus", timeout=8)
        if r.ok:
            js = r.json()
            for seg in js.get("marketState", []):
                if "Equity" in seg.get("market","") and seg.get("marketStatus")=="Open":
                    return True,"Market Open"
    except Exception: pass
    return True,"Market Open (holiday check not confirmed)"

def _clean_symbol_keep_punct(s): return re.sub(r'[^A-Za-z0-9\-\&\.]+','',str(s)).upper()

def read_html_tables(url):
    headers={"User-Agent":"Mozilla/5.0"}
    resp=requests.get(url,headers=headers,timeout=30)
    resp.raise_for_status()
    html=resp.text
    for kw in ({},{"flavor":"bs4"},{"flavor":"html5lib"}):
        try: return pd.read_html(StringIO(html),**kw)
        except: continue
    raise RuntimeError("Install parsers: pip install beautifulsoup4 html5lib")

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_nifty100_symbols():
    try:
        url="https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
        r=requests.get(url, headers={"User-Agent":"Mozilla/5.0"},timeout=30)
        r.raise_for_status()
        df=pd.read_csv(StringIO(r.content.decode("utf-8")))
        sym_col=[c for c in df.columns if re.search(r'(symbol|ticker|nse)', c, re.I)][0]
        df[sym_col]=df[sym_col].apply(_clean_symbol_keep_punct)
        syms=sorted(df[sym_col].dropna().astype(str).tolist())
        return [s+".NS" for s in syms]
    except Exception:
        tables=read_html_tables("https://en.wikipedia.org/wiki/NIFTY_100")
        cons=None
        for t in tables:
            cols=[c.lower() for c in t.columns]
            if any('symbol' in c or 'ticker' in c or 'nse' in c for c in cols) and any('company' in c or 'name' in c for c in cols) and len(t)>=40:
                cons=t.copy();break
        sym_col=[c for c in cons.columns if re.search(r'(symbol|ticker|code|nse)',c,re.I)][0]
        cons[sym_col]=cons[sym_col].apply(_clean_symbol_keep_punct)
        syms=sorted(cons[sym_col].tolist())
        return [s+".NS" for s in syms]

# --- Market data ---
def download_fields(tickers,start,end,fields=("Adj Close","Volume"),chunk=50):
    tickers=list(dict.fromkeys([t for t in tickers if isinstance(t,str)]))
    out={f:[] for f in fields}
    for i in range(0,len(tickers),chunk):
        t2=tickers[i:i+chunk]
        data=safe_yf_download(t2,start=start,end=end,progress=False,threads=True)
        if not isinstance(data,pd.DataFrame): continue
        if isinstance(data.columns,pd.MultiIndex):
            for f in fields:
                sub=data.get(f)
                if sub is not None: out[f].append(sub.copy())
        else:
            for f in fields:
                sub=data.get(f)
                if sub is None: continue
                if isinstance(sub,pd.Series): sub=sub.to_frame()
                out[f].append(sub.copy())
    for f in fields:
        if out[f]:
            df=pd.concat(out[f],axis=1)
            df=df.loc[:,~df.columns.duplicated()].sort_index()
            out[f]=df
        else: out[f]=pd.DataFrame()
    return out

@st.cache_data(ttl=2*3600, show_spinner=True)
def load_market_data(lookback_days=420):
    today=now_ist().date()
    start=(today-timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end=(today+timedelta(days=2)).strftime("%Y-%m-%d")
    tickers=fetch_nifty100_symbols()
    b=safe_yf_download("^NSEI",start=start,end=end,progress=False)
    if b is not None and not b.empty:
        ser=(b["Adj Close"] if "Adj Close" in b else b["Close"]).dropna()
        if isinstance(ser,pd.DataFrame): ser=ser.squeeze("columns")
        bench=ser.rename("NIFTY50")
    else: bench=pd.Series(dtype=float,name="NIFTY50")
    fields=download_fields(tickers,start,end,fields=("Adj Close","Volume"))
    prices=fields["Adj Close"].reindex(bench.index).ffill()
    vols=fields["Volume"].reindex(bench.index).ffill()
    turnover_cr=(prices*vols)/1e7
    med_turnover=turnover_cr.rolling(20,min_periods=20).median()
    return dict(bench=bench,prices=prices,vols=vols,
                med_turnover=med_turnover,tickers=list(prices.columns))

# ----------------- Compute Signals -----------------
def shares_from_lot(price, lot_cash, fee):
    per_share = price * (1 + fee)
    if per_share <= 0: return 0
    return max(int(lot_cash // per_share), 0)

def compute_signals(params, mkt, positions_df, balances_df, ledger_df, sells_done_today):
    p=params
    bench=mkt["bench"]; prices=mkt["prices"]; vols=mkt["vols"]; med_turnover=mkt["med_turnover"]
    tickers=mkt["tickers"]

    # --- Try get live prices
    live_prices = {}
    try:
        data = yf.download(tickers + ["^NSEI"], period="5d", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close=data["Close"].ffill().iloc[-1]
            live_prices=close.to_dict()
    except Exception as e:
        st.warning(f"⚠️ Live price fetch failed: {e}. Using yesterday’s close for signals.")

    if not live_prices:
        st.warning("⚠️ No live prices available — using last cached daily close instead.")
    
    today=bench.index[-1]
    bench_ma60=bench.rolling(p["regime_filter_ma"], min_periods=p["regime_filter_ma"]).mean().iloc[-1]
    bench_live=live_prices.get("^NSEI", float(bench.iloc[-1]))
    regime_ok=bool(bench_live >= float(bench_ma60)*(1+p["regime_buffer"])) if pd.notna(bench_ma60) else True
    div_today=p["divisor"] if regime_ok else p["divisor_bear"]

    row_live=prices.iloc[-1].copy()
    for sym in row_live.index:
        if sym in live_prices: row_live[sym]=live_prices[sym]

    ma=prices.rolling(p["ma"],min_periods=p["ma"]).mean().iloc[-1]
    std=prices.rolling(p["ma"],min_periods=p["ma"]).std().iloc[-1]
    liq_today=med_turnover.iloc[-1]

    base_capital=float(balances_df.iloc[0]["base_capital"]) if not balances_df.empty else DEFAULTS["base_capital"]
    realized=float(balances_df.iloc[0]["realized"]) if not balances_df.empty else 0.0
    cash=float(balances_df.iloc[0]["cash"]) if not balances_df.empty else base_capital
    lot_cash=(base_capital+realized)/div_today

    positions={}
    if not positions_df.empty:
        for _, r in positions_df.iterrows():
            positions[str(r["symbol"])] = dict(
                shares=int(r["shares"]), avg_cost=float(r["avg_cost"]),
                last_buy=float(r.get("last_buy", r["avg_cost"])),
                open_date=str(r.get("open_date") or "")
            )

    sells=[]; left=max(0, p["max_sells_per_day"]-sells_done_today)
    for sym,pos in positions.items():
        px=float(row_live.get(sym,np.nan))
        if pd.isna(px) or pos["shares"]<=0: continue
        r=px/pos["avg_cost"]-1
        if r>=p["take_profit"]:
            sells.append(dict(symbol=sym,price=round(px,2),shares=pos["shares"],
                              reason="TP",ret=round(r*100,2)))
    sells.sort(key=lambda x:x.get("ret",0),reverse=True)
    sells=sells[:left]

    elig=[c for c in tickers if pd.notna(ma.get(c)) and pd.notna(row_live.get(c))
          and row_live[c]<ma[c] and (liq_today.get(c,np.nan) >= p["min_turnover_cr"])]
    ranked_syms=[]
    if p["use_zscore"]:
        zlist=[]
        for c in elig:
            s=std.get(c)
            if pd.isna(s) or s<=0: continue
            z=(row_live[c]-ma[c])/s
            zlist.append((z,c))
        zlist.sort(key=lambda x:x[0])
        ranked_syms=[c for _,c in zlist[:p["bottom_n"]]]
    else:
        dist=(row_live/ma-1).dropna()
        ranked_syms=list(dist.sort_values().index[:p["bottom_n"]])

    buys_new=[]
    for sym in ranked_syms:
        if sym in positions: continue
        px=float(row_live.get(sym,np.nan))
        if pd.isna(px): continue
        sh=shares_from_lot(px,lot_cash,p["fee"])
        if sh<=0: continue
        cost=sh*px*(1+p["fee"])
        buys_new.append(dict(symbol=sym,price=round(px,2),shares=sh,
                             reason="NEW",est_cost=round(cost,2)))
        if len(buys_new)>=p["max_new_buys"]: break

    buys_avg=[]
    for sym,pos in positions.items():
        px=float(row_live.get(sym,np.nan))
        if pd.isna(px): continue
        m=ma.get(sym); s=std.get(sym)
        if pd.isna(m) or pd.isna(s) or s<=0: continue
        z=(px-m)/s
        price_ok=px <= pos["last_buy"]*(1-p["avg_dd"])
        regime_gate=regime_ok or (not p["averaging_requires_regime"] and z <= p["avg_in_bear_z_thresh"])
        if price_ok and regime_gate:
            sh=shares_from_lot(px,lot_cash,p["fee"])
            if sh>0:
                buys_avg.append(dict(symbol=sym,price=round(px,2),shares=sh,
                                     reason="AVERAGE",z=round(float(z),2)))
    buys_avg.sort(key=lambda x:x.get("z",0.0))

    return dict(
        regime_ok=regime_ok, bench_live=bench_live, lot_cash=lot_cash,
        sells=sells, buys_new=buys_new, buys_avg=buys_avg, cash=cash
    )
# ----------------- Apply executed trades -----------------
def apply_trade_rows(sh, trades, fee_rate):
    balances, positions, ledger, config, daily_eq = load_all(sh)
    if balances.empty:
        balances=pd.DataFrame([dict(cash=DEFAULTS["base_capital"],
            base_capital=DEFAULTS["base_capital"],realized=0.0,fees_paid=0.0,
            last_update=str(date.today()))])
    cash=float(balances.iloc[0]["cash"]);base_capital=float(balances.iloc[0]["base_capital"])
    realized=float(balances.iloc[0]["realized"]);fees_paid=float(balances.iloc[0]["fees_paid"])
    pos_map={};
    for _, r in positions.iterrows():
        pos_map[str(r["symbol"])]=dict(shares=int(r["shares"]),
            avg_cost=float(r["avg_cost"]), last_buy=float(r.get("last_buy", r["avg_cost"])),
            open_date=str(r.get("open_date") or ""))

    for tr in trades:
        side=tr["side"].upper()
        sym=tr.get("symbol","").strip()
        qty=int(tr.get("shares",0))
        px=float(tr.get("price",0.0))
        fee=float(tr.get("fee", px*qty*fee_rate))
        reason=tr.get("reason","")
        dt=tr.get("date", str(now_ist().date()))
        if side=="FUND_IN":
            cash += px; base_capital += px
            ledger=pd.concat([ledger,pd.DataFrame([dict(date=dt,side=side,symbol="",shares=0,price=px,fee=0.0,reason="FUND_IN",realized_pnl=0.0)])],ignore_index=True)
        elif side=="FUND_OUT":
            cash -= px; base_capital -= px
            ledger=pd.concat([ledger,pd.DataFrame([dict(date=dt,side=side,symbol="",shares=0,price=px,fee=0.0,reason="FUND_OUT",realized_pnl=0.0)])],ignore_index=True)
        elif side=="BUY" and sym and qty>0 and px>0:
            gross=qty*px; total=gross+fee
            if cash<total: continue
            cash-=total; fees_paid+=fee
            if sym in pos_map:
                pos=pos_map[sym]
                tot_cost=pos["avg_cost"]*pos["shares"]+gross
                pos["shares"]+=qty
                pos["avg_cost"]=(tot_cost/pos["shares"]) if pos["shares"]>0 else pos["avg_cost"]
                pos["last_buy"]=px
            else:
                pos_map[sym]=dict(shares=qty, avg_cost=px, last_buy=px, open_date=dt)
            ledger=pd.concat([ledger,pd.DataFrame([dict(date=dt,side="BUY",symbol=sym,shares=qty,price=px,fee=round(fee,2),reason=reason,realized_pnl=0.0)])],ignore_index=True)
        elif side=="SELL" and sym and qty>0 and px>0:
            if sym not in pos_map: continue
            pos=pos_map[sym]
            qty=min(qty,pos["shares"])
            gross=qty*px
            proceeds=gross-fee
            pnl=proceeds-qty*pos["avg_cost"]
            cash+=proceeds; fees_paid+=fee; realized+=pnl
            pos["shares"]-=qty
            if pos["shares"]<=0: del pos_map[sym]
            ledger=pd.concat([ledger,pd.DataFrame([dict(date=dt,side="SELL",symbol=sym,shares=qty,price=px,fee=round(fee,2),reason=reason,realized_pnl=round(pnl,2))])],ignore_index=True)

    pos_rows=[dict(symbol=sym, shares=pos["shares"], avg_cost=round(pos["avg_cost"],2),
                   last_buy=round(pos.get("last_buy",pos["avg_cost"]),2), open_date=pos.get("open_date",""))
              for sym,pos in pos_map.items()]
    positions_out=pd.DataFrame(pos_rows).sort_values("symbol") if pos_rows else pd.DataFrame(columns=["symbol","shares","avg_cost","last_buy","open_date"])
    balances_out=pd.DataFrame([dict(cash=round(cash,2),base_capital=round(base_capital,2),
                                   realized=round(realized,2),fees_paid=round(fees_paid,2),
                                   last_update=str(now_ist().date()))])
    save_df(sh,"positions",positions_out)
    save_df(sh,"ledger",ledger.sort_values(["date","side","symbol"]))
    save_df(sh,"balances",balances_out)
# ----------------- Performance metrics -----------------
def compute_xirr(flows):
    """flows: list of (date, amount), FUND_IN negative, FUND_OUT positive, final equity positive."""
    if not flows or len(flows)<2: return np.nan
    dates = [pd.to_datetime(d).to_pydatetime() for d,_ in flows]
    amounts=[float(a) for _,a in flows]
    t0=dates[0]
    days=np.array([(d-t0).days for d in dates],dtype=float)
    def xnpv(rate):
        return np.sum([amt/((1+rate)**(dd/365.0)) for amt,dd in zip(amounts,days)])
    low,high=-0.999,10.0
    for _ in range(100):
        mid=(low+high)/2
        v=xnpv(mid)
        if abs(v)<1e-6: return mid
        v_low=xnpv(low)
        if v_low*v<0: high=mid
        else: low=mid
    return np.nan

def compute_twr_cagr(daily_eq_df, ledger):
    if daily_eq_df.empty or len(daily_eq_df)<2: return np.nan, np.nan
    df=daily_eq_df.copy()
    df["date"]=pd.to_datetime(df["date"]).dt.date
    df=df.sort_values("date")
    led=ledger.copy()
    if led.empty: led=pd.DataFrame(columns=["date","side","price"])
    else: led["date"]=pd.to_datetime(led["date"]).dt.date
    flow = led.groupby(["date","side"])["price"].sum().unstack(fill_value=0.0)
    flow["net"] = flow.get("FUND_OUT",0.0)-flow.get("FUND_IN",0.0)
    flow=flow["net"]
    rets=[]; prev=None
    for d,row in df.set_index("date").iterrows():
        if prev is None: prev=row["equity"]; continue
        beg=prev; net=float(flow.get(d,0.0))
        r=(row["equity"]-beg-net)/beg if beg>0 else 0.0
        rets.append(r); prev=row["equity"]
    if not rets: return np.nan, np.nan
    twr_total=np.prod([1+r for r in rets])-1
    start=pd.to_datetime(df["date"].iloc[0])
    end=pd.to_datetime(df["date"].iloc[-1])
    years=max((end-start).days/365.25,1e-9)
    cagr=(1+twr_total)**(1/years)-1
    return twr_total,cagr

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
    try:
        _, sa_email = _service_account()
        st.write(f"Service account email (share your Sheet with this): {sa_email}")
    except Exception: pass
    st.stop()

# Open sheet and ensure tabs
sh, sa_email = open_sheet(sheet_url)
ensure_tabs(sh)
balances, positions, ledger, config, daily_eq = load_all(sh)

# Expose key config values (editable in Sheet->config)
cfg = DEFAULTS.copy()
if not config.empty:
    for k in cfg:
        if k in config.columns:
            val=config.iloc[0].get(k)
            if pd.notna(val):
                try: cfg[k]=float(val)
                except: pass
base_capital=float(balances.iloc[0]["base_capital"]) if not balances.empty else DEFAULTS["base_capital"]

# Load market data
mkt=load_market_data(DEFAULTS["lookback_days"])
# Backfill daily equity
universe = sorted(set(list(positions["symbol"]) + list(ledger["symbol"].dropna().unique())))
if universe:
    start_hist=(now_ist().date()-timedelta(days=720)).strftime("%Y-%m-%d")
    end_hist=(now_ist().date()+timedelta(days=2)).strftime("%Y-%m-%d")
    fields_hist=download_fields(universe,start_hist,end_hist,fields=("Adj Close",))
    px_hist=fields_hist["Adj Close"].ffill()
    px_hist=px_hist.reindex(mkt["bench"].index).ffill()
    first_needed=(pd.to_datetime(ledger["date"]).dt.date.min() if not ledger.empty else now_ist().date())
    start_day=min(first_needed,now_ist().date()) if first_needed else now_ist().date()
    if not mkt["bench"].empty:
        cutoff=mkt["bench"].index[-1].date()
    else:
        cutoff=now_ist().date()
    df_new=reconstruct_daily_equity(ledger,balances,start_day,cutoff,px_hist,cfg["fee"])
    if not df_new.empty:
        save_df(sh,"daily_equity",df_new)
        daily_eq=df_new.copy()

# Tabs
tab1, tab2 = st.tabs(["Run Signals","My Portfolio"])

# --- Tab 1: Run Signals ---
with tab1:
    open_ok, open_msg = is_market_open()
    st.markdown(f"**Market status:** {open_msg}")
    if not open_ok: st.warning("Run is disabled when market is closed.")
    colA,colB,colC = st.columns([1,1,1])
    with colA: run_click=st.button("▶️ Run scan",use_container_width=True,disabled=not open_ok)
    with colB: fund_amt=st.number_input("Add/Withdraw funds (₹): + add, - withdraw",value=0,step=500)
    with colC:
        if st.button("Apply funds",use_container_width=True):
            if fund_amt!=0:
                apply_trade_rows(sh,[dict(date=str(now_ist().date()),side="FUND_IN" if fund_amt>0 else "FUND_OUT",
                                          symbol="",shares=0,price=abs(float(fund_amt)),fee=0.0,reason="")],cfg["fee"])
                st.success("Funds updated.")
                balances,positions,ledger,config,daily_eq=load_all(sh)
    st.caption(f"Service account email: {sa_email}")

    if run_click and open_ok:
        today_d=now_ist().date()
        sells_today=0
        if not ledger.empty:
            ld=ledger.copy(); ld["date"]=pd.to_datetime(ld["date"]).dt.date
            sells_today=int(((ld["date"]==today_d) & (ld["side"].str.upper()=="SELL")).sum())
        signals=compute_signals(cfg,mkt,positions,balances,ledger,sells_today)
        st.markdown(f"- Regime: {'BULL' if signals['regime_ok'] else 'BEAR'} | Lot cash: ₹{signals['lot_cash']:.0f} | Cash: ₹{signals['cash']:.0f}")
        st.subheader("Sells (cap 4/day)")
        if not signals["sells"]: st.info("No sells by rule.")
        else: st.dataframe(pd.DataFrame(signals["sells"]),use_container_width=True)
        st.subheader("New Buys (ranked)")
        if not signals["buys_new"]: st.info("No new buys qualified.")
        else: st.dataframe(pd.DataFrame(signals["buys_new"]),use_container_width=True)
        st.subheader("Averaging candidates")
        if not signals["buys_avg"]: st.info("No averaging candidates.")
        else: st.dataframe(pd.DataFrame(signals["buys_avg"]),use_container_width=True)

        st.markdown("---")
        st.markdown("#### Mark executed trades (partial fills OK)")
        execs=[]
        with st.form("exec_trades"):
            if signals["sells"]:
                st.write("Executed Sells")
                cap_left=cfg["max_sells_per_day"]-sells_today
                for i,s in enumerate(signals["sells"]):
                    c1,c2,c3,c4=st.columns([2,1,1,2])
                    with c1: do=st.checkbox(f"SELL {s['symbol']} ({s['reason']})",key=f"sell_{i}")
                    with c2: qty=st.number_input("Qty",min_value=0,max_value=int(s["shares"]),value=int(s["shares"]),step=1,key=f"sellqty_{i}")
                    with c3: px=st.number_input("Price",min_value=0.0,value=float(s["price"]),step=0.05,key=f"sellpx_{i}")
                    with c4: st.caption(f"TP%: {s.get('ret','')}")
                    if do and qty>0:
                        execs.append(dict(side="SELL",symbol=s["symbol"],shares=qty,price=px,reason=s["reason"]))
                if len([e for e in execs if e["side"]=="SELL"])>cap_left:
                    st.warning(f"You can execute at most {cap_left} sells today.")

            if signals["buys_new"]:
                st.write("Executed New Buys")
                for i,b in enumerate(signals["buys_new"]):
                    c1,c2,c3,c4=st.columns([2,1,1,2])
                    with c1: do=st.checkbox(f"BUY {b['symbol']}",key=f"buynew_{i}")
                    with c2: qty=st.number_input("Qty",min_value=0,value=int(b["shares"]),step=1,key=f"buynewqty_{i}")
                    with c3: px=st.number_input("Price",min_value=0.0,value=float(b["price"]),step=0.05,key=f"buynewpx_{i}")
                    with c4: st.caption(f"Est cost: ₹{b['est_cost']}")
                    if do and qty>0:
                        execs.append(dict(side="BUY",symbol=b["symbol"],shares=qty,price=px,reason="NEW"))

            if signals["buys_avg"]:
                st.write("Executed Averaging")
                for i,b in enumerate(signals["buys_avg"]):
                    c1,c2,c3,c4=st.columns([2,1,1,2])
                    with c1: do=st.checkbox(f"AVG {b['symbol']} (z={b.get('z','')})",key=f"buyavg_{i}")
                    with c2: qty=st.number_input("Qty",min_value=0,value=int(b["shares"]),step=1,key=f"buyavgqty_{i}")
                    with c3: px=st.number_input("Price",min_value=0.0,value=float(b["price"]),step=0.05,key=f"buyavgpx_{i}")
                    with c4: st.caption("Lower than last_buy and rule OK")
                    if do and qty>0:
                        execs.append(dict(side="BUY",symbol=b["symbol"],shares=qty,price=px,reason="AVERAGE"))
            submitted=st.form_submit_button("💾 Update portfolio")
            if submitted:
                sells_today2=0
                if not ledger.empty:
                    ld=ledger.copy(); ld["date"]=pd.to_datetime(ld["date"]).dt.date
                    sells_today2=int(((ld["date"]==now_ist().date()) & (ld["side"].str.upper()=="SELL")).sum())
                cap_left2=max(0,cfg["max_sells_per_day"]-sells_today2)
                sells_exec=[e for e in execs if e["side"]=="SELL"][:cap_left2]
                buys_exec=[e for e in execs if e["side"]=="BUY"]
                all_exec=[]
                for e in sells_exec+buys_exec:
                    e["date"]=str(now_ist().date())
                    e["fee"]=round(e["shares"]*e["price"]*cfg["fee"],2)
                    all_exec.append(e)
                if all_exec:
                    apply_trade_rows(sh,all_exec,cfg["fee"]); st.success("Portfolio updated.")
                else: st.info("No trades to apply.")
                balances,positions,ledger,config,daily_eq=load_all(sh)

# --- Tab 2: My Portfolio ---
with tab2:
    st.subheader("My Portfolio")

    if not mkt["prices"].empty:
        # Try to fetch the latest intraday close (1‑minute interval)
        live_prices = yf.download(list(mkt["prices"].columns), period="1d", interval="1m", progress=False)

        if not live_prices.empty and ("Close" in live_prices):
            last_close_row = live_prices["Close"].iloc[-1]
        else:
            # Fallback: use previous daily close from cache
            last_close_row = mkt["prices"].iloc[-1]

        # If only a single value (scalar), wrap in Series for consistency
        if np.isscalar(last_close_row):
            col = mkt["prices"].columns[0]
            last_close_row = pd.Series({col: float(last_close_row)})
    else:
        last_close_row = pd.Series(dtype=float)

    holdings_df,mv=position_snapshot(positions,last_close_row)
    cash=float(balances.iloc[0]["cash"]) if not balances.empty else DEFAULTS["base_capital"]
    equity_close=cash+mv
    exposure=(mv/equity_close) if equity_close>0 else 0.0
    fees_paid=float(balances.iloc[0]["fees_paid"]) if not balances.empty else 0.0
    realized=float(balances.iloc[0]["realized"]) if not balances.empty else 0.0

    c1,c2,c3,c4,c5=st.columns(5)
    with c1: st.metric("Cash (₹)",f"{cash:,.0f}")
    with c2: st.metric("Equity (last close, ₹)",f"{equity_close:,.0f}")
    with c3: st.metric("Exposure",f"{exposure*100:.1f}%")
    with c4: st.metric("Realized PnL (₹)",f"{realized:,.0f}")
    with c5: st.metric("Fees paid (₹)",f"{fees_paid:,.0f}")

    st.markdown("#### Holdings (valued at last close)")
    st.dataframe(holdings_df,use_container_width=True,height=360)

    deq=daily_eq.copy()
    if not deq.empty:
        deq["date"]=pd.to_datetime(deq["date"])
        deq=deq.sort_values("date")
        deq["equity"]=pd.to_numeric(deq["equity"],errors="coerce")
        twr_total,twr_cagr=compute_twr_cagr(deq,ledger)
        flows=[]
        if not ledger.empty:
            led=ledger.copy(); led["date"]=pd.to_datetime(led["date"]).dt.date
            for _,r in led.iterrows():
                if (r["side"] or "").upper()=="FUND_IN": flows.append((r["date"],-float(r["price"])))
                elif (r["side"] or "").upper()=="FUND_OUT": flows.append((r["date"],+float(r["price"])))
        flows=sorted(flows,key=lambda x:x[0])
        if not deq.empty: flows.append((deq["date"].iloc[-1].date(),float(deq["equity"].iloc[-1])))
        xirr=compute_xirr(flows) if len(flows)>=2 else np.nan

        c1,c2,c3=st.columns(3)
        with c1: st.metric("TWR (total)",f"{(twr_total or 0)*100:.2f}%")
        with c2: st.metric("CAGR (TWR)",f"{(twr_cagr or 0)*100:.2f}%")
        with c3: st.metric("XIRR",f"{(xirr or 0)*100:.2f}%")

        st.markdown("#### Charts")
        fig,ax=plt.subplots(figsize=(8,3))
        ax.plot(deq["date"], deq["equity"]/deq["equity"].iloc[0],label="Strategy (₹ normalized)")
        ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig,use_container_width=True)

        dd=deq.set_index("date")["equity"]/deq.set_index("date")["equity"].cummax()-1
        fig2,ax2=plt.subplots(figsize=(8,2.5))
        ax2.fill_between(dd.index,dd.values,0,color="tab:red",alpha=0.35)
        ax2.set_title("Drawdown"); ax2.grid(alpha=0.3)
        st.pyplot(fig2,use_container_width=True)

        fig3,ax3=plt.subplots(figsize=(8,2.5))
        ax3.plot(deq["date"],deq["exposure"].astype(float))
        ax3.set_ylim(0,1.1); ax3.set_title("Exposure"); ax3.grid(alpha=0.3)
        st.pyplot(fig3,use_container_width=True)

        if len(deq)>=252:
            s=deq.set_index("date")["equity"].astype(float)
            roll=(s/s.shift(252)-1).dropna()
            fig4,ax4=plt.subplots(figsize=(8,2.5))
            ax4.plot(roll.index,roll.values)
            ax4.set_title("Rolling 1Y return"); ax4.grid(alpha=0.3)
            st.pyplot(fig4,use_container_width=True)

        st.markdown("#### Downloads")
        st.download_button("Download trade_ledger.csv",data=ledger.to_csv(index=False),file_name="trade_ledger.csv",mime="text/csv")
        st.download_button("Download holdings.csv",data=holdings_df.to_csv(index=False),file_name="holdings.csv",mime="text/csv")
        st.download_button("Download daily_summary.csv",data=deq.to_csv(index=False),file_name="daily_summary.csv",mime="text/csv")
        st.download_button("Download equity_series.csv",data=deq[["date","equity"]].to_csv(index=False),file_name="equity_series.csv",mime="text/csv")
    else:
        st.info("No daily equity yet. Execute a trade or add funds to start the series.")



