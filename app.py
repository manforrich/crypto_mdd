import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import time
from datetime import datetime, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="全能回測 + 蒙地卡羅", layout="wide")
st.title("🎲 全能策略回測系統 (含蒙地卡羅分析)")

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

# --- 核心函數：分批抓取數據 (抗封鎖版) ---
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

# --- 數學指標計算函數 ---
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

def analyze_drawdown(df):
    df['Rolling_Max'] = df['close'].cummax()
    df['Drawdown'] = (df['close'] / df['Rolling_Max']) - 1
    max_dd = df['Drawdown'].min()
    current_dd = df['Drawdown'].iloc[-1]
    return df, max_dd, current_dd

# --- 新增：蒙地卡羅模擬函數 ---
def run_monte_carlo(trade_log, initial_capital, simulations=1000):
    if trade_log.empty or len(trade_log) < 5:
        return None, "交易次數太少，無法進行模擬"
    
    # 取得每筆交易的獲利百分比 (轉為小數點，例如 5% -> 0.05)
    returns = trade_log['單筆獲利 (%)'].values / 100
    
    simulation_results = []
    final_equities = []
    max_drawdowns = []
    
    for _ in range(simulations):
        # 隨機抽樣 (Bootstrapping)：從歷史交易中隨機抽取 N 筆 (允許重複抽樣)
        # 這模擬了「如果是同樣的策略，但交易順序或頻率改變」的情況
        daily_returns = np.random.choice(returns, size=len(returns), replace=True)
        
        equity_curve = [initial_capital]
        current_equity = initial_capital
        
        for r in daily_returns:
            current_equity = current_equity * (1 + r)
            equity_curve.append(current_equity)
            
        simulation_results.append(equity_curve)
        final_equities.append(current_equity)
        
        # 計算該次模擬的 MDD
        eq_series = pd.Series(equity_curve)
        mdd = calculate_mdd(eq_series)
        max_drawdowns.append(mdd)

    return {
        "paths": simulation_results,
        "final_equities": final_equities,
        "max_drawdowns": max_drawdowns
    }, None

def run_strategy(df_input, short_w, long_w, ma_type, capital):
    df = df_input.copy()
    col_s, col_l = f'MA_{short_w}', f'MA_{long_w}'
    
    df[col_s] = calculate_ma(df['close'], short_w, ma_type)
    df[col_l] = calculate_ma(df['close'], long_w, ma_type)
    
    df['Signal'] = 0
    df.loc[(df[col_s] > df[col_l]) & (df[col_s].shift(1) <= df[col_l].shift(1)), 'Signal'] = 1
    df.loc[(df[col_s] < df[col_l]) & (df[col_s].shift(1) >= df[col_l].shift(1)), 'Signal'] = -1
    
    balance = capital
    position = 0
    equity = []
    trades = 0
    trade_log = [] 
    current_entry_price = 0
    current_entry_time = None
    buy_signals = []
    sell_signals = []
    
    for i, row in df.iterrows():
        price = row['close']
        time = row['timestamp']
        if row['Signal'] == 1 and position == 0:
            position = balance / price
            balance = 0
            trades += 1
            current_entry_price = price
            current_entry_time = time
            buy_signals.append((time, price))
        elif row['Signal'] == -1 and position > 0:
            balance = position * price
            position = 0
            trades += 1
            sell_signals.append((time, price))
            pnl = (price - current_entry_price) / current_entry_price * 100
            trade_log.append({"買入時間": current_entry_time, "買入價格": current_entry_price, "賣出時間": time, "賣出價格": price, "單筆獲利 (%)": pnl})
        equity.append(balance + (position * price))
        
    df['Equity'] = equity
    final_equity = equity[-1]
    roi = ((final_equity - capital) / capital) * 100
    mdd = calculate_mdd(pd.Series(equity))
    return {"final_equity": final_equity, "roi": roi, "trades": trades, "mdd": mdd, "df": df, "buys": buy_signals, "sells": sell_signals, "trade_log": pd.DataFrame(trade_log)}

# --- 主程式執行 ---

if start_date > end_date:
    st.error("❌ 日期設定錯誤")
else:
    st.write(f"正在下載 **{selected_symbol}** 數據...")
    raw_data, source = get_data_by_date_range(selected_symbol, timeframe, start_date, end_date)

    if raw_data is not None and not raw_data.empty:
        st.success(f"✅ 下載完成 (來源: {source}) | 共 {len(raw_data)} 根 K 棒")
        
        # --- Drawdown 分析區塊 (保持收合) ---
        with st.expander("🌊 風險深度與回撤分析 (Drawdown)", expanded=False):
            dd_df, dd_max, dd_curr = analyze_drawdown(raw_data.copy())
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("最大回撤 (Max DD)", f"{dd_max:.2%}", delta_color="inverse")
            kpi2.metric("目前回撤 (Current DD)", f"{dd_curr:.2%}", delta_color="inverse")
            fig_dd = px.area(dd_df, x='timestamp', y='Drawdown', title="水下圖 (Underwater Plot)", color_discrete_sequence=['#EF553B'])
            st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("---")

        bh_equity = initial_capital * (raw_data['close'] / raw_data['close'].iloc[0])
        bh_roi = ((bh_equity.iloc[-1] - initial_capital) / initial_capital) * 100
        bh_mdd = calculate_mdd(bh_equity)

        # 執行策略
        res_a = run_strategy(raw_data, short_a, long_a, ma_type_a, initial_capital)
        res_b = run_strategy(raw_data, short_b, long_b, ma_type_b, initial_capital)
        
        # 績效看板
        st.subheader("🏆 策略績效對決")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🔵 策略 A: {ma_type_a}")
            st.metric("ROI", f"{res_a['roi']:.2f}%", f"{res_a['roi']-bh_roi:.2f}% vs B&H")
            st.metric("MDD", f"{res_a['mdd']:.2f}%", delta_color="inverse")
            st.write(f"交易: {res_a['trades']}")
        with col2:
            st.info(f"🟠 策略 B: {ma_type_b}")
            st.metric("ROI", f"{res_b['roi']:.2f}%", f"{res_b['roi']-bh_roi:.2f}% vs B&H")
            st.metric("MDD", f"{res_b['mdd']:.2f}%", delta_color="inverse")
            st.write(f"交易: {res_b['trades']}")
        with col3:
            st.write("### 🏳️ Buy & Hold")
            st.metric("ROI", f"{bh_roi:.2f}%")
            st.metric("MDD", f"{bh_mdd:.2f}%")

        # 圖表與明細
        st.markdown("---")
        view_option = st.radio("選擇要查看的策略詳情：", ("策略 A", "策略 B"), horizontal=True)
        target_res = res_a if view_option == "策略 A" else res_b
        target_short = short_a if view_option == "策略 A" else short_b
        target_long = long_a if view_option == "策略 A" else long_b
        
        tab1, tab2 = st.tabs(["📈 K 線圖與買賣點", "📋 交易明細表"])

        with tab1:
            fig_k = go.Figure()
            fig_k.add_trace(go.Candlestick(x=target_res['df']['timestamp'], open=target_res['df']['open'], high=target_res['df']['high'], low=target_res['df']['low'], close=target_res['df']['close'], name='價格'))
            fig_k.add_trace(go.Scatter(x=target_res['df']['timestamp'], y=target_res['df'][f'MA_{target_short}'], line=dict(color='orange', width=1), name=f'MA {target_short}'))
            fig_k.add_trace(go.Scatter(x=target_res['df']['timestamp'], y=target_res['df'][f'MA_{target_long}'], line=dict(color='blue', width=1), name=f'MA {target_long}'))
            if target_res['buys']:
                bx, by = zip(*target_res['buys'])
                fig_k.add_trace(go.Scatter(x=bx, y=by, mode='markers', name='買進', marker=dict(symbol='triangle-up', size=15, color='#00CC96')))
            if target_res['sells']:
                sx, sy = zip(*target_res['sells'])
                fig_k.add_trace(go.Scatter(x=sx, y=sy, mode='markers', name='賣出', marker=dict(symbol='triangle-down', size=15, color='#EF553B')))
            fig_k.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_k, use_container_width=True)

        with tab2:
            if not target_res['trade_log'].empty:
                st.dataframe(target_res['trade_log'], use_container_width=True)
            else:
                st.warning("無交易紀錄")

        # --- 新增：蒙地卡羅分析區塊 ---
        st.markdown("---")
        with st.expander("🎲 蒙地卡羅模擬 (Monte Carlo Simulation) - 點擊展開", expanded=True):
            st.write(f"針對 **{view_option}** 的交易邏輯進行 1000 次隨機模擬，評估策略的穩健性。")
            
            mc_result, mc_err = run_monte_carlo(target_res['trade_log'], initial_capital)
            
            if mc_result:
                # 1. 繪製義大利麵圖 (Spaghetti Plot)
                fig_mc = go.Figure()
                
                # 畫出隨機 50 條模擬路徑 (設為半透明)
                sim_paths = mc_result['paths']
                display_paths = sim_paths[:50] 
                
                # 生成 X 軸 (交易次數)
                x_axis = list(range(len(display_paths[0])))
                
                for path in display_paths:
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=path, mode='lines', line=dict(color='gray', width=1), opacity=0.1, showlegend=False))
                
                # 畫出中位數路徑 (Median)
                median_path = np.median(sim_paths, axis=0)
                fig_mc.add_trace(go.Scatter(x=x_axis, y=median_path, mode='lines', name='中位數預期 (Median)', line=dict(color='#00CC96', width=3)))
                
                # 畫出原始策略路徑
                original_equity = target_res['df']['Equity'].iloc[::int(len(target_res['df'])/len(x_axis)) if len(target_res['df']) > len(x_axis) else 1]
                # 注意：這裡只能近似對齊，因為 K 線長度跟交易次數不同。為了圖表簡單，我們只顯示模擬的統計分佈
                
                fig_mc.update_layout(title="模擬未來資產走勢 (1000次模擬)", xaxis_title="交易次數", yaxis_title="資產淨值", template="plotly_dark", height=500)
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # 2. 統計數據
                final_eqs = np.array(mc_result['final_equities'])
                max_dds = np.array(mc_result['max_drawdowns'])
                
                col_m1, col_m2, col_m3 = st.columns(3)
                
                # 中位數獲利
                median_profit = np.median(final_eqs)
                col_m1.metric("模擬中位數資產", f"${median_profit:,.0f}", f"{(median_profit-initial_capital)/initial_capital*100:.1f}% ROI")
                
                # VaR (Value at Risk) 95%
                var_95 = np.percentile(final_eqs, 5) # 最差 5% 的情況
                col_m2.metric("95% 信心水準最差資產 (VaR)", f"${var_95:,.0f}", delta_color="inverse")
                
                # 最差回撤預期
                worst_mdd = np.percentile(max_dds, 5) # 因為 MDD 是負數，取 5th percentile 代表"負得比較多"的那端 (e.g. -40% vs -10%)
                col_m3.metric("模擬最差 MDD (95% 機率)", f"{worst_mdd:.2f}%", delta_color="inverse")
                
                # 破產機率
                ruin_prob = (final_eqs < initial_capital * 0.1).mean() * 100 # 剩不到 10% 視為破產
                if ruin_prob > 0:
                    st.error(f"⚠️ 破產機率 (資產剩 <10%): {ruin_prob:.1f}%")
                else:
                    st.success("✅ 模擬中未出現破產情況")
                
            else:
                st.warning(f"無法執行模擬：{mc_err}")

    else:
        st.error("無法獲取數據")
