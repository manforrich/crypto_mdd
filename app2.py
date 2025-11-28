import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import time
from datetime import datetime, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="自訂資金配比回測", layout="wide")
st.title("💰 自訂資金配比策略：MA 趨勢 + 暴跌加碼")

# --- 1. 側邊欄設定 ---
st.sidebar.header("1. 數據設定")
common_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BTC/USD', 'ETH/USD', 'DOGE/USDT', 'XRP/USDT']
selected_symbol = st.sidebar.selectbox("交易對", common_pairs)
custom_symbol = st.sidebar.text_input("自定義 (如 BNB/USDT)", "").upper()
if custom_symbol: selected_symbol = custom_symbol

timeframe = st.sidebar.selectbox("K線週期", ["15m", "1h", "4h", "1d", "1w"], index=3)

st.sidebar.markdown("### 選擇日期範圍")
default_start = datetime.now() - timedelta(days=365)
default_end = datetime.now()
col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("開始日期", default_start)
end_date = col_d2.date_input("結束日期", default_end)

initial_capital = st.sidebar.number_input("初始本金 (USDT)", value=10000)

st.sidebar.markdown("---")

# --- 2. 策略設定 ---
ma_options = ["SMA (簡單)", "EMA (指數)", "HMA (赫爾)"]

st.sidebar.subheader("🔵 策略 A")
ma_type_a = st.sidebar.selectbox("種類 A", ma_options, key='type_a', index=1)
short_a = st.sidebar.number_input("短 A", value=5, key='short_a')
long_a = st.sidebar.number_input("長 A", value=20, key='long_a')

st.sidebar.subheader("🟠 策略 B")
ma_type_b = st.sidebar.selectbox("種類 B", ma_options, key='type_b', index=2)
short_b = st.sidebar.number_input("短 B", value=10, key='short_b')
long_b = st.sidebar.number_input("長 B", value=60, key='long_b')

st.sidebar.markdown("---")

# --- 新增：資金與加碼控制 ---
st.sidebar.subheader("💸 資金與加碼設定")

# 1. 首單資金
initial_entry_pct = st.sidebar.slider("1️⃣ 首單資金占比 (% of 總現金)", min_value=10, max_value=100, value=50, step=10)
st.sidebar.caption(f"當 MA 黃金交叉時，投入當下現金的 {initial_entry_pct}%。")

# 2. 加碼設定
enable_dip_buy = st.sidebar.checkbox("✅ 啟用「暴跌加碼」機制", value=True)
if enable_dip_buy:
    dip_threshold = st.sidebar.slider("📉 觸發加碼的跌幅 (% From High)", min_value=10, max_value=90, value=50, step=5)
    dip_entry_pct = st.sidebar.slider("2️⃣ 加碼單資金占比 (% of 剩餘現金)", min_value=10, max_value=100, value=100, step=10)
    st.sidebar.caption(f"當價格從高點回撤 {dip_threshold}% 時，投入剩餘現金的 {dip_entry_pct}%。")
else:
    dip_threshold = 0
    dip_entry_pct = 0

# --- 核心函數：抓取數據 ---
@st.cache_data(ttl=3600)
def get_data_by_date_range(symbol, timeframe, start_date, end_date):
    exchanges_list = [('Binance', ccxt.binance()), ('Binance US', ccxt.binanceus()), ('Kraken', ccxt.kraken())]
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for exchange_name, exchange in exchanges_list:
        try:
            if not exchange.fetch_ohlcv(symbol, timeframe, limit=1): continue 
            status_text.text(f"正在從 {exchange_name} 下載數據...")
            since = exchange.parse8601(f"{start_date}T00:00:00Z")
            end_timestamp = exchange.parse8601(f"{end_date}T23:59:59Z")
            all_ohlcv = []
            limit = 1000 
            
            while since < end_timestamp:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not ohlcv: break
                all_ohlcv += ohlcv
                last_timestamp = ohlcv[-1][0]
                if last_timestamp >= end_timestamp: break
                since = last_timestamp + 1 
                total = end_timestamp - exchange.parse8601(f"{start_date}T00:00:00Z")
                curr = last_timestamp - exchange.parse8601(f"{start_date}T00:00:00Z")
                progress_bar.progress(min(curr / total, 1.0))
                time.sleep(exchange.rateLimit / 1000 if exchange.rateLimit else 0.1)

            if not all_ohlcv: continue
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            mask = (df['timestamp'] >= pd.to_datetime(start_date)) & (df['timestamp'] <= pd.to_datetime(end_date) + timedelta(days=1))
            df = df.loc[mask]
            progress_bar.progress(1.0)
            status_text.empty()
            return df, exchange_name
        except: continue
    progress_bar.empty()
    return None, "Fail"

# --- 數學計算 ---
def calculate_wma(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hma(series, window):
    half_window = int(window / 2)
    sqrt_window = int(np.sqrt(window))
    wma_half = calculate_wma(series, half_window)
    wma_full = calculate_wma(series, window)
    raw_hma = 2 * wma_half - wma_full
    return calculate_wma(raw_hma, sqrt_window)

def calculate_ma(series, window, ma_type):
    if "EMA" in ma_type: return series.ewm(span=window, adjust=False).mean()
    elif "HMA" in ma_type: return calculate_hma(series, window)
    else: return series.rolling(window).mean()

def calculate_mdd(equity_series):
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    return drawdown.min() * 100 

def analyze_drawdown_depth(df):
    df = df.copy()
    df['Rolling_Max'] = df['close'].cummax()
    df['Drawdown'] = (df['close'] / df['Rolling_Max']) - 1
    max_dd = df['Drawdown'].min()
    current_dd = df['Drawdown'].iloc[-1]
    return df, max_dd, current_dd

# --- 修正後的策略執行 (含自訂資金配比) ---
def run_strategy(df_input, short_w, long_w, ma_type, capital, enable_dip, dip_thresh, init_pct, dip_pct):
    df = df_input.copy()
    col_s, col_l = f'MA_{short_w}', f'MA_{long_w}'
    
    df[col_s] = calculate_ma(df['close'], short_w, ma_type)
    df[col_l] = calculate_ma(df['close'], long_w, ma_type)
    
    # 計算價格的 Drawdown (用來觸發加碼)
    df['Price_Rolling_Max'] = df['close'].cummax()
    df['Price_DD'] = (df['close'] / df['Price_Rolling_Max']) - 1
    
    df['MA_Signal'] = 0
    df.loc[(df[col_s] > df[col_l]) & (df[col_s].shift(1) <= df[col_l].shift(1)), 'MA_Signal'] = 1
    df.loc[(df[col_s] < df[col_l]) & (df[col_s].shift(1) >= df[col_l].shift(1)), 'MA_Signal'] = -1
    
    balance = capital
    position = 0 
    equity = []
    trades = 0
    trade_log = [] 
    avg_cost = 0 
    
    buy_signals = []
    add_signals = [] 
    sell_signals = []
    
    # 跌幅閾值 (e.g. 50% -> -0.5)
    dip_limit = - (dip_thresh / 100.0)

    for i, row in df.iterrows():
        price = row['close']
        time = row['timestamp']
        
        # A. 賣出 (死亡交叉)
        if row['MA_Signal'] == -1 and position > 0:
            pnl = (price - avg_cost) / avg_cost * 100
            balance = position * price
            position = 0
            avg_cost = 0
            trades += 1
            sell_signals.append((time, price))
            trade_log.append({"動作": "賣出 (Sell)", "時間": time, "價格": price, "獲利 (%)": pnl})
            
        # B. 買進 (黃金交叉)
        elif row['MA_Signal'] == 1:
            if position == 0:
                # 使用使用者設定的比例
                invest_amt = balance * (init_pct / 100.0)
                
                # 至少要有 10U 才能交易
                if invest_amt > 10:
                    new_units = invest_amt / price
                    position += new_units
                    balance -= invest_amt
                    avg_cost = price 
                    trades += 1
                    buy_signals.append((time, price))
                    trade_log.append({"動作": f"首單 ({init_pct}%)", "時間": time, "價格": price, "獲利 (%)": 0})
        
        # C. 逆勢加碼
        # 條件：啟用 + 有現金 + 跌破閾值
        if enable_dip and balance > 10 and (row['Price_DD'] <= dip_limit):
            # 使用使用者設定的「剩餘現金比例」
            invest_amt = balance * (dip_pct / 100.0)
            
            if invest_amt > 10:
                new_units = invest_amt / price
                total_value_cost = (position * avg_cost) + invest_amt
                position += new_units
                balance -= invest_amt
                avg_cost = total_value_cost / position
                trades += 1
                add_signals.append((time, price))
                trade_log.append({"動作": f"加碼 ({dip_pct}%)", "時間": time, "價格": price, "獲利 (%)": 0})

        current_equity = balance + (position * price)
        equity.append(current_equity)
        
    df['Equity'] = equity
    final_equity = equity[-1]
    roi = ((final_equity - capital) / capital) * 100
    mdd = calculate_mdd(pd.Series(equity))
    
    return {
        "final_equity": final_equity, 
        "roi": roi, 
        "trades": trades, 
        "mdd": mdd, 
        "df": df, 
        "buys": buy_signals, 
        "adds": add_signals,
        "sells": sell_signals, 
        "trade_log": pd.DataFrame(trade_log)
    }

# --- 主程式 ---

if start_date > end_date:
    st.error("❌ 日期設定錯誤")
else:
    st.write(f"正在下載 **{selected_symbol}** 數據...")
    raw_data, source = get_data_by_date_range(selected_symbol, timeframe, start_date, end_date)

    if raw_data is not None and not raw_data.empty:
        st.success(f"✅ 下載完成 (來源: {source}) | 共 {len(raw_data)} 根 K 棒")
        
        with st.expander("🌊 風險深度與回撤機率分析", expanded=True):
            dd_df, dd_max, dd_curr = analyze_drawdown_depth(raw_data)
            col1, col2 = st.columns(2)
            col1.metric("期間最大回撤 (Max DD)", f"{dd_max:.2%}", delta_color="inverse")
            col2.metric("目前回撤", f"{dd_curr:.2%}", delta_color="inverse")
            st.plotly_chart(px.area(dd_df, x='timestamp', y='Drawdown', title="水下圖", color_discrete_sequence=['#EF553B']), use_container_width=True)

        st.markdown("---")

        bh_equity = initial_capital * (raw_data['close'] / raw_data['close'].iloc[0])
        bh_roi = ((bh_equity.iloc[-1] - initial_capital) / initial_capital) * 100
        bh_mdd = calculate_mdd(bh_equity)

        # 傳入新的資金參數 (init_pct, dip_pct)
        res_a = run_strategy(raw_data, short_a, long_a, ma_type_a, initial_capital, enable_dip_buy, dip_threshold, initial_entry_pct, dip_entry_pct)
        res_b = run_strategy(raw_data, short_b, long_b, ma_type_b, initial_capital, enable_dip_buy, dip_threshold, initial_entry_pct, dip_entry_pct)
        
        st.subheader("🏆 策略績效對決")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🔵 策略 A")
            st.metric("ROI", f"{res_a['roi']:.2f}%", f"{res_a['roi']-bh_roi:.2f}% vs B&H")
            st.metric("MDD", f"{res_a['mdd']:.2f}%", delta_color="inverse")
            st.write(f"交易: {res_a['trades']}")
        with col2:
            st.info(f"🟠 策略 B")
            st.metric("ROI", f"{res_b['roi']:.2f}%", f"{res_b['roi']-bh_roi:.2f}% vs B&H")
            st.metric("MDD", f"{res_b['mdd']:.2f}%", delta_color="inverse")
            st.write(f"交易: {res_b['trades']}")
        with col3:
            st.write("### 🏳️ Buy & Hold")
            st.metric("ROI", f"{bh_roi:.2f}%")
            st.metric("MDD", f"{bh_mdd:.2f}%")

        st.markdown("---")
        view_option = st.radio("選擇策略詳情：", ("策略 A", "策略 B"), horizontal=True)
        target_res = res_a if view_option == "策略 A" else res_b
        target_short = short_a if view_option == "策略 A" else short_b
        target_long = long_a if view_option == "策略 A" else long_b
        
        tab1, tab2 = st.tabs(["📈 K 線圖與加碼點", "📋 交易明細表"])

        with tab1:
            fig_k = go.Figure()
            fig_k.add_trace(go.Candlestick(x=target_res['df']['timestamp'], open=target_res['df']['open'], high=target_res['df']['high'], low=target_res['df']['low'], close=target_res['df']['close'], name='價格'))
            fig_k.add_trace(go.Scatter(x=target_res['df']['timestamp'], y=target_res['df'][f'MA_{target_short}'], line=dict(color='orange', width=1), name=f'MA {target_short}'))
            fig_k.add_trace(go.Scatter(x=target_res['df']['timestamp'], y=target_res['df'][f'MA_{target_long}'], line=dict(color='blue', width=1), name=f'MA {target_long}'))
            
            if target_res['buys']:
                bx, by = zip(*target_res['buys'])
                fig_k.add_trace(go.Scatter(x=bx, y=by, mode='markers', name='首單', marker=dict(symbol='triangle-up', size=12, color='#00CC96')))
            
            if target_res['adds']:
                ax, ay = zip(*target_res['adds'])
                fig_k.add_trace(go.Scatter(x=ax, y=ay, mode='markers', name='加碼', marker=dict(symbol='star', size=15, color='#AB63FA')))
            
            if target_res['sells']:
                sx, sy = zip(*target_res['sells'])
                fig_k.add_trace(go.Scatter(x=sx, y=sy, mode='markers', name='賣出', marker=dict(symbol='triangle-down', size=12, color='#EF553B')))
            
            fig_k.update_layout(template='plotly_dark', height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_k, use_container_width=True)

        with tab2:
            if not target_res['trade_log'].empty:
                st.dataframe(target_res['trade_log'], use_container_width=True)
            else:
                st.warning("無交易紀錄")

    else:
        st.error("無法獲取數據")
