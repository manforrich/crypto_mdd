import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 頁面設定 ---
st.set_page_config(page_title="加密貨幣回撤分析儀", layout="wide")

st.title("📊 加密貨幣 Drawdown (回撤) 風險分析")
st.markdown("這個儀表板模擬了從歷史最高點買入後的 **回撤風險** 以及 **回撤深度的機率分佈**。")

# --- 側邊欄：使用者設定 ---
st.sidebar.header("參數設定")
symbol = st.sidebar.text_input("輸入幣種代碼 (Yahoo Finance 格式)", value="BTC-USD")
start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("today"))

# --- 1. 抓取資料函數 ---
@st.cache_data
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        if df.empty:
            return None
        # 處理 MultiIndex (Yahoo Finance 有時會有多層索引)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, level=1, axis=1) if symbol in df.columns.levels[1] else df
            # 如果上面 xs 失敗，嘗試直接取 Close，這裡做簡單化處理
            if 'Close' not in df.columns:
                 df = df.droplevel(1, axis=1) 
        return df
    except Exception as e:
        st.error(f"抓取資料失敗: {e}")
        return None

# --- 2. 核心計算邏輯 ---
def calculate_metrics(df):
    # 計算歷史最高價 (Rolling Max)
    df['Rolling_Max'] = df['Close'].cummax()
    
    # 計算回撤 (Drawdown) -> (現價 - 最高價) / 最高價
    df['Drawdown'] = (df['Close'] / df['Rolling_Max']) - 1
    
    # 最大回撤 (Max Drawdown)
    max_dd = df['Drawdown'].min()
    
    # 目前回撤
    current_dd = df['Drawdown'].iloc[-1]
    
    return df, max_dd, current_dd

# --- 主程式執行 ---
data = load_data(symbol, start_date, end_date)

if data is not None:
    # 進行計算
    df_processed, max_dd, current_dd = calculate_metrics(data)
    
    # --- 顯示關鍵指標 (KPI) ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("目前價格", f"${df_processed['Close'].iloc[-1]:,.2f}")
    col2.metric("歷史最高價 (在此區間)", f"${df_processed['Rolling_Max'].iloc[-1]:,.2f}")
    col3.metric("目前回撤 (Current DD)", f"{current_dd:.2%}", delta_color="inverse")
    col4.metric("最大回撤 (Max DD)", f"{max_dd:.2%}", delta_color="inverse")

    st.markdown("---")

    # --- 圖表區塊 1: 價格與歷史高點 ---
    st.subheader("📈 價格走勢 vs 歷史高點")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_processed.index, y=df_processed['Close'], name='收盤價', line=dict(color='blue')))
    fig_price.add_trace(go.Scatter(x=df_processed.index, y=df_processed['Rolling_Max'], name='歷史最高價', line=dict(color='green', dash='dash')))
    st.plotly_chart(fig_price, use_container_width=True)

    # --- 圖表區塊 2: 水下圖 (Underwater Plot) ---
    st.subheader("🌊 水下圖 (Underwater Plot): 歷史回撤走勢")
    fig_dd = px.area(df_processed, x=df_processed.index, y='Drawdown', 
                     title="歷史回撤幅度 (0% 代表創新高)", color_discrete_sequence=['red'])
    fig_dd.update_yaxes(tickformat=".0%") # 顯示百分比
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- 圖表區塊 3: 回撤機率統計 (您的核心需求) ---
    st.subheader("🎲 回撤機率分佈 (Drawdown Distribution)")
    
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        # 繪製直方圖
        fig_hist = px.histogram(df_processed, x="Drawdown", nbins=50, 
                                title="回撤深度分佈圖 (Histogram)",
                                labels={'Drawdown': '回撤幅度'},
                                histnorm='percent', # 顯示百分比機率
                                template="plotly_white")
        fig_hist.update_xaxes(tickformat=".0%")
        fig_hist.update_yaxes(title="發生機率 (%)")
        # 加一條線顯示目前位置
        fig_hist.add_vline(x=current_dd, line_dash="dash", line_color="red", annotation_text="目前位置")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_stats:
        st.write("#### 統計數據解讀")
        st.write("這個分佈圖告訴你，持有這個資產時，**有多少時間是處於特定的虧損狀態**。")
        
        # 計算一些簡單的統計量
        dd_series = df_processed['Drawdown']
        time_above_20 = (dd_series > -0.2).mean()
        time_below_50 = (dd_series < -0.5).mean()
        
        st.info(f"🛡️ **高於 -20% 的時間**: {time_above_20:.1%}\n\n(代表大部分時間回撤都在 20% 以內)")
        st.warning(f"⚠️ **腰斬 (低於 -50%) 的時間**: {time_below_50:.1%}\n\n(代表你有多少機率會看到資產腰斬)")
        
        st.write("---")
        st.write("**數據說明**：")
        st.write("* **X 軸**：回撤幅度 (0% 是高點，-50% 是腰斬)")
        st.write("* **Y 軸**：該回撤幅度出現的天數佔總天數的比例")

else:
    st.warning("找不到資料，請檢查代碼是否正確 (例如 BTC-USD, ETH-USD, SOL-USD)")
