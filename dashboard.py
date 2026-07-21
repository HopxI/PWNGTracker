from pandas import col
import streamlit as st
import yfinance as yf
import requests
import time
import datetime
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

# --- ML FORECASTING MODULES ---
def get_fundamentals_data(ticker, asset_type):
    """Fetches financial statements, key ratios, and SEC filing metadata."""
    if asset_type != "Stock":
        return None
        
    try:
        asset = yf.Ticker(ticker)
        info = asset.info
        
        # 1. Key Valuation Ratios
        ratios = {
            "PE": info.get("forwardPE") or info.get("trailingPE"),
            "EPS": info.get("trailingEps"),
            "ROE": info.get("returnOnEquity"),
            "PB": info.get("priceToBook"),
            "DebtToEquity": info.get("debtToEquity"),
            "ProfitMargin": info.get("profitMargins")
        }
        
        # 2. Financial Statements (Income Statement, Balance Sheet, Cash Flow)
        financials = {
            "income": asset.financials,
            "balance": asset.balance_sheet,
            "cashflow": asset.cashflow
        }
        
        # 3. SEC CIK & Regulatory Filing Lookup
        cik = info.get("cik")
        sec_url = f"https://www.sec.gov/edgar/browse/?CIK={cik}" if cik else f"https://www.sec.gov/edgar/searchedgar/companysearch?q={ticker}"
        uk_gov_url = f"https://find-and-update.company-information.service.gov.uk/search?q={ticker}"
        
        return {
            "ratios": ratios,
            "statements": financials,
            "sec_url": sec_url,
            "uk_url": uk_gov_url
        }
    except Exception:
        return None
    
def predict_svr(df_close):
    """Predicts next-day price using Support Vector Regression (SVR)."""
    if len(df_close) < 30:
        return None
    
    # Prepare X (day sequence) and y (prices)
    X = np.arange(len(df_close)).reshape(-1, 1)
    y = df_close.values
    
    # Scale features for SVM stability
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Train RBF Kernel SVR
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model.fit(X_scaled, y_scaled)
    
    # Predict next index (day N+1)
    next_day_scaled = scaler_x.transform(np.array([[len(df_close)]]))
    pred_scaled = model.predict(next_day_scaled)
    pred_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    
    return pred_price

def predict_lstm(df_close, lookback=10):
    """Predicts next-day price using a 1D Sequential LSTM Network."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        # Suppress verbose TF logging output in terminal
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        return None

    if len(df_close) < (lookback + 20):
        return None

    data = df_close.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Build sequence pairs (lookback days -> next day)
    X_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        X_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Construct lightweight LSTM Architecture
    model = Sequential([
        LSTM(units=32, return_sequences=False, input_shape=(X_train.shape[1], 1)),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

    # Predict next day sequence
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
    pred_scaled = model.predict(last_sequence, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

    return pred_price

# --- PAGE SETUP ---
st.set_page_config(page_title="Institutional Trading Terminal", layout="wide")
st.title("🏛️ Institutional Research Terminal: Live Market & Signals")

# --- PUT YOUR HARDCODED KEYS HERE ---
if "telegram_token" not in st.session_state:
    st.session_state["telegram_token"] = "8786336007:AAFwpupFITzwWHwB1mLoKmUu1FJyVIGd1WA" 

if "telegram_chat_id" not in st.session_state:
    st.session_state["telegram_chat_id"] = "8026092339" 

# --- CACHING THE AI MODEL ---
@st.cache_resource
def load_ai_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

with st.spinner("Initializing AI Sentiment Engine (FinBERT)..."):
    sentiment_analyzer = load_ai_model()

COINGECKO_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple", 
    "ADA": "cardano", "DOGE": "dogecoin", "DOT": "polkadot", "AVAX": "avalanche-2"
}

# --- REAL-TIME TRACKER MODULE ---
def get_live_ticker_data(tickers):
    """Fetches real-time 1-minute interval data for the live tracker."""
    live_data = {}
    for t in tickers:
        try:
            asset = yf.Ticker(t)
            hist = asset.history(period="2d", interval="1m")
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                start_of_day = hist['Close'].iloc[0] 
                pct_change = ((current - start_of_day) / start_of_day) * 100
                live_data[t] = {"price": current, "change": pct_change}
        except Exception:
            pass
    return live_data

# --- TECHNICAL ANALYSIS & TIMING MODULE ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).iloc[-1]

def get_market_data_expanded(ticker, asset_type):
    symbol = ticker if asset_type == "Stock" else f"{ticker}-USD"
    try:
        asset = yf.Ticker(symbol)
        hist = asset.history(period="1y")
        if hist.empty or len(hist) < 30: return None
        
        current_price = hist['Close'].iloc[-1]
        ema_5 = hist['Close'].ewm(span=5, adjust=False).mean().iloc[-1]
        sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        
        # Support and Resistance for Setups
        support = hist['Low'].tail(20).min()
        resistance = hist['High'].tail(20).max()
        
        sma_200_window = 200 if len(hist) >= 200 else len(hist)
        sma_200 = hist['Close'].rolling(window=sma_200_window).mean().iloc[-1]
        rsi_14 = calculate_rsi(hist['Close'])
        
        return {
            "current_price": current_price,
            "ema_5": ema_5,
            "sma_20": sma_20,
            "sma_200": sma_200,
            "rsi_14": rsi_14,
            "support": support,
            "resistance": resistance
        }
    except Exception:
        return None

# --- FUNDAMENTALS & MACRO SCRAPING ---
def crawl_macro_environment():
    url = "https://finance.yahoo.com/news/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    keywords = ['fed', 'rate', 'sec', 'mica', 'inflation', 'liquidity', 'cpi', 'powell']
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h3.text.strip() for h3 in soup.find_all('h3')]
        macro_news = [h for h in headlines if any(w in h.lower() for w in keywords)]
        return macro_news[:4] if macro_news else ["No major macro catalysts detected today."]
    except Exception:
        return ["Failed to scrape macro environment."]

# --- CONTROL PANEL ---
st.sidebar.header("⚙️ Command Center")
asset_type = st.sidebar.radio("Asset Class:", ["Stock", "Crypto"])
default_tickers = "AAPL, NVDA" if asset_type == "Stock" else "BTC, SOL"
user_input = st.sidebar.text_input("Tickers (comma separated):", default_tickers)

# ADD THIS LINE HERE (Outside the loop, with a unique key)
etherscan_key = st.sidebar.text_input("Etherscan API Key (100% Free):", type="password", key="global_etherscan_key")

# Live Refresh Toggle
auto_refresh = st.sidebar.checkbox("🟢 Enable Live Auto-Refresh (60s)")
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

# --- LIVE MARKET TRACKER (Top Row) ---
st.markdown("### ⚡ Real-Time Market Tracker")
live_symbols = ["SPY", "QQQ", "AAPL"] if asset_type == "Stock" else ["BTC-USD", "ETH-USD", "SOL-USD"]
live_data = get_live_ticker_data(live_symbols)

if live_data:
    cols = st.columns(len(live_symbols))
    for i, sym in enumerate(live_symbols):
        with cols[i]:
            price = live_data[sym]["price"]
            change = live_data[sym]["change"]
            color = "normal" if change == 0 else ("inverse" if change < 0 else "normal")
            st.metric(label=f"Live: {sym}", value=f"${price:,.2f}", delta=f"{change:.2f}% intraday", delta_color=color)
st.divider()

# --- MAIN DASHBOARD LOGIC ---
if run_analysis or "has_run" in st.session_state or auto_refresh:
    st.session_state["has_run"] = True
    tickers = [t.strip().upper() for t in user_input.split(',') if t.strip()]
    
    macro_news = crawl_macro_environment()
    macro_score = sum([1 if sentiment_analyzer(m)[0]['label'] == 'positive' else -1 for m in macro_news if "Failed" not in m and "No major" not in m])
    
    # Session Timing Check
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    current_hour_utc = now_utc.hour
    
    if asset_type == "Stock":
        is_peak_session = 13 <= current_hour_utc < 16 # First 2.5 hours of NY session
        session_text = f"**Current Session (UTC):** {current_hour_utc:02d}:00 | **Peak NY Volume Window:** 13:30 - 16:00 UTC."
        timing_status = "🟢 ACTIVE PEAK SESSION" if is_peak_session else "🟡 LOW VOLUME - AWAIT PEAK HOURS"
    else:
        # Crypto London/NY overlap
        is_peak_session = 12 <= current_hour_utc < 17 
        session_text = f"**Current Session (UTC):** {current_hour_utc:02d}:00 | **Peak London/NY Overlap:** 12:00 - 17:00 UTC."
        timing_status = "🟢 ACTIVE PEAK SESSION" if is_peak_session else "🟡 LOWER VOLUME PHASE"
    
    for ticker in tickers:
        st.markdown(f"### 🔍 Deep Dive: **{ticker}**")
        
        tech = get_market_data_expanded(ticker, asset_type)
        if not tech:
            st.error(f"Could not load historical data for {ticker}.")
            continue
            
        tab_setup, tab_tech, tab_ml, tab_signal, tab_whales, tab_fund = st.tabs([
            "🎯 Entry Points & Timings", 
            "⚡ Technical Data", 
            "📈 ML & Neural Forecasts",
            "🧠 Long-Term View",
            "🐋 Smart Money & Insiders",
            "📊 Fundamentals"
        ])
        
        # TAB 1: SPECIFIC ENTRY POINTS & TIMINGS
        with tab_setup:
            st.markdown("#### 🕰️ Execution Timing Window")
            st.write(session_text)
            st.info(f"**Status:** {timing_status} — Breakout and pullback trades have the highest probability of success during peak volume. Executing outside these hours increases the risk of fakeouts.")
            
            st.markdown("#### 🎯 Entry Triggers (Short-Term 1-5 Day Swing)")
            
            # Entry Logic Calculation
            current_price = tech['current_price']
            support = tech['support']
            resistance = tech['resistance']
            sma_20 = tech['sma_20']
            
            # Distance from key levels
            dist_to_support = ((current_price - support) / support) * 100
            dist_to_res = ((resistance - current_price) / current_price) * 100
            
            c1, c2, c3 = st.columns(3)
            
            # 1. Optimal Pullback Entry
            c1.markdown("**1. The Pullback Entry (Limit Buy)**")
            c1.write(f"Wait for the price to drop back to strong support before buying.")
            c1.metric("Limit Buy Target", f"${support * 1.01:,.2f}") # 1% above support to ensure fill
            
            # 2. Market Momentum Entry
            c2.markdown("**2. The Momentum Entry (Market Order)**")
            if current_price > sma_20 and tech['ema_5'] > sma_20:
                c2.write("Trend is currently bullish. Acceptable to enter at market price.")
                c2.metric("Market Entry", f"${current_price:,.2f}", delta="Trend Supported")
            else:
                c2.write("Trend is bearish. Do not use a market entry here.")
                c2.metric("Market Entry", f"${current_price:,.2f}", delta="Too Risky", delta_color="inverse")
                
            # 3. Breakout Entry
            c3.markdown("**3. The Breakout Entry (Stop-Limit Buy)**")
            c3.write(f"Wait for the price to smash through the ceiling before entering.")
            c3.metric("Breakout Trigger", f"${resistance * 1.01:,.2f}") # 1% above resistance
            
            st.markdown("---")
            st.markdown("#### 🛡️ Risk Management (Exits)")
            e1, e2 = st.columns(2)
            e1.metric("Hard Stop-Loss (SL)", f"${support * 0.98:,.2f}", delta="-2% Below Support", delta_color="inverse")
            e2.metric("Take-Profit (TP)", f"${resistance * 0.98:,.2f}", delta="Front-running Resistance")

        # TAB 2: TECHNICAL DATA
        with tab_tech:
            st.subheader("Volatility Breakdown")
            t1, t2, t3 = st.columns(3)
            t1.metric("Current RSI", f"{tech['rsi_14']:.1f}")
            t2.metric("Short Trend (5D vs 20D)", "🟢 Bullish" if tech['ema_5'] > tech['sma_20'] else "🔴 Bearish")
            t3.metric("Long Trend (200D SMA)", "🟢 Above 200D SMA" if tech['current_price'] > tech['sma_200'] else "🔴 Below 200D SMA")

        # TAB 3: ML & NEURAL FORECASTS
        with tab_ml:
            st.subheader("Machine Learning Price Forecasts")
            st.caption("Using Support Vector Regression (SVR) and LSTM Neural Networks to predict next-day price movements.")

            symbol = ticker if asset_type == "Stock" else f"{ticker}-USD"
            hist_data = yf.Ticker(symbol).history(period="1y")['Close']

            if not hist_data.empty:
                col_svr, col_lstm = st.columns(2)

                with col_svr:
                    with st.spinner("Running SVR Forecast..."):
                        svr_pred = predict_svr(hist_data)
                    if svr_pred:
                        delta_svr = ((svr_pred - tech['current_price']) / tech['current_price']) * 100
                        st.metric(
                            label="SVR Next-Day Prediction",
                            value=f"${svr_pred:,.2f}",
                            delta=f"{delta_svr:+.2f}% vs Current"
                        )
                        st.info("SVR uses historical price patterns to predict the next day's closing price. Not a guarantee, but a statistical estimate.")
                    else:
                        st.write("Insufficient historical data for SVR training.")
                with col_lstm:
                    with st.spinner("Running LSTM Forecast..."):
                        lstm_pred = predict_lstm(hist_data)
                    if lstm_pred:
                        delta_lstm = ((lstm_pred - tech['current_price']) / tech['current_price']) * 100
                        st.metric(
                            label="LSTM Next-Day Prediction",
                            value=f"${lstm_pred:,.2f}",
                            delta=f"{delta_lstm:+.2f}% vs Current"
                        )
                        st.info("LSTM uses deep learning to capture complex temporal patterns in price data. Predictions are probabilistic, not certain.")
                    else:
                        st.warning("Insufficient historical data for LSTM training. Requires at least 30 days of data.")

            else:
                st.error("Failed to retrieve historical price data for ML forecasting.")  
        # TAB 4: LONG-TERM VIEW
        with tab_signal:
            st.subheader("Macro & Structural Investment Thesis")
            is_lt_bullish = tech['current_price'] > tech['sma_200']
            if is_lt_bullish and macro_score >= 0:
                st.success("🟢 **STRUCTURAL BULL TREND** - Trading above the 200-day average with supportive or neutral macro conditions. Suitable for accumulation.")
            else:
                st.warning("🔴 **STRUCTURAL BEAR / NEUTRAL** - Proceed with caution on long-term holds. Waiting for macro conditions to improve or price to reclaim 200-day average.")

        # TAB 5: SMART MONEY & WHALES
        with tab_whales:
            st.subheader("Smart Money & Institutional Tracking")
            
            if asset_type == "Stock":
                st.write("Tracking corporate insiders (CEOs, CFOs, Board Members). When insiders buy with their own money, it is historically a strong conviction signal.")
                try:
                    # Pull SEC Form 4 Insider Trading filings natively via yfinance
                    insiders = yf.Ticker(ticker).insider_transactions
                    
                    if insiders is not None and not insiders.empty:
                        # Clean up the dataframe for the dashboard
                        st.dataframe(insiders.head(8), use_container_width=True)
                        
                        # Basic logic to check if insiders are mostly buying or selling
                        if 'Text' in insiders.columns:
                            purchases = len(insiders[insiders['Text'].str.contains('Purchase', case=False, na=False)])
                            sales = len(insiders[insiders['Text'].str.contains('Sale', case=False, na=False)])
                            
                            st.markdown("#### Insider Sentiment Trigger")
                            if purchases > sales:
                                st.success(f"🟢 **BULLISH CONVICTION:** Insiders are heavily accumulating ({purchases} recent buys vs {sales} recent sells).")
                            elif sales > purchases:
                                st.error(f"🔴 **BEARISH CONVICTION:** Insiders are offloading shares ({sales} recent sells vs {purchases} recent buys).")
                            else:
                                st.info("🟡 **NEUTRAL:** Mixed insider trading activity.")
                    else:
                        st.write("No recent insider transactions filed with the SEC for this asset.")
                except Exception:
                    st.write("Insider data currently unavailable.")
                    
            else:
                # Crypto Tracking (Free Etherscan API)
                st.write("Tracking massive on-chain 'Smart Money' stablecoin movements.")
                st.caption("Monitoring the Ethereum blockchain for USDT transfers over $1,000,000.")
                
                # Check the key we defined up in the sidebar
                if etherscan_key:
                    with st.spinner("Scanning blockchain for massive transfers..."):
                        usdt_contract = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
                        url = f"https://api.etherscan.io/v2/api?chainid=1&module=account&action=tokentx&contractaddress={usdt_contract}&page=1&offset=50&sort=desc&apikey={etherscan_key}"
                        
                        try:
                            response = requests.get(url, timeout=5).json()
                            if response.get('status') == '1':
                                txs = response.get('result', [])
                                whales = [tx for tx in txs if (float(tx['value']) / 10**6) > 1000000]
                                
                                if whales:
                                    st.success(f"Found {len(whales)} massive Tether (USDT) transfers just now.")
                                    KNOWN = {
                                        "0x28c6c06298d514db089934071355e5743bf21d60": "Binance",
                                        "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "Binance",
                                        "0x5041ed759dd4afc3a72b8192c143f72f4724081a": "Coinbase",
                                        "0x00000000219ab540356cbb839cbe05303d7705fa": "Kraken"
                                    }
                                    
                                    for w in whales[:5]:
                                        amount = float(w['value']) / 10**6
                                        f_addr = w['from'].lower()
                                        t_addr = w['to'].lower()
                                        
                                        from_name = KNOWN.get(f_addr, f_addr[:8] + "...")
                                        to_name = KNOWN.get(t_addr, t_addr[:8] + "...")
                                        
                                        # 1. Determine message content and UI alert type
                                        if to_name in KNOWN.values():
                                            alert_type = "success"
                                            msg = f"🟢 **BUYING POWER:** ${amount:,.0f} USDT deposited to {to_name} (Whale preparing to buy)"
                                        else:
                                            alert_type = "error"
                                            msg = f"🔴 **CASHING OUT:** ${amount:,.0f} USDT withdrawn from {from_name} (Whale securing fiat profits)"

                                        # 2. Display on Streamlit dashboard
                                        if alert_type == "success":
                                            st.success(msg)
                                        elif alert_type == "error":
                                            st.error(msg)
                                        else:
                                            st.info(msg)

                                        # 3. Send Telegram alert if not already sent
                                        token = st.session_state.get("telegram_token")
                                        chat_id = st.session_state.get("telegram_chat_id")

                                        if token and chat_it and w['hash'] not in st.session_state["alerted_txs"]:
                                            send_telegram_alert(msg, token, chat_id)
                                            st.session_state["alerted_txs"].add(w['hash'])
                                            
                                else:
                                    st.write("No transfers over $1,000,000 found in recent blocks.")
                            else:
                                api_error = response.get('result', 'Unknown Error')
                                st.error(f"Etherscan API Error: {api_error}")
                        except Exception as e:
                            st.error(f"Failed to connect to Etherscan: {e}")
                else:
                    st.info("🔗 **Integration Required:** Go to etherscan.io/apis to create a 100% free API key and unlock this tab.")
        # TAB 6: FUNDAMENTALS & REGULATORY FILINGS
        with tab_fund:
            if asset_type != "Stock":
                st.info("ℹ️ Fundamental Financial Statements and SEC Filings are only applicable to Stocks.")
            else:
                fund_data = get_fundamentals_data(ticker, asset_type)
                
                if fund_data:
                    # --- SECTION 1: KEY VALUATION RATIOS ---
                    st.subheader("📈 Key Metrics & Valuation Ratios")
                    ratios = fund_data["ratios"]
                    
                    r1, r2, r3, r4 = st.columns(4)
                    
                    pe_val = f"{ratios['PE']:.2f}" if ratios['PE'] else "N/A"
                    r1.metric("P/E Ratio (Price-to-Earnings)", pe_val)
                    
                    eps_val = f"${ratios['EPS']:.2f}" if ratios['EPS'] else "N/A"
                    r2.metric("Earnings Per Share (EPS)", eps_val)
                    
                    roe_val = f"{ratios['ROE'] * 100:.2f}%" if ratios['ROE'] else "N/A"
                    r3.metric("Return on Equity (ROE)", roe_val)
                    
                    margin_val = f"{ratios['ProfitMargin'] * 100:.2f}%" if ratios['ProfitMargin'] else "N/A"
                    r4.metric("Profit Margin", margin_val)
                    
                    st.divider()
                    
                    # --- SECTION 2: FINANCIAL STATEMENTS ---
                    st.subheader("📑 Financial Statements Review")
                    
                    stmt_type = st.radio(
                        "Select Statement:", 
                        ["Income Statement", "Balance Sheet", "Cash Flow"], 
                        horizontal=True,
                        key=f"stmt_select_{ticker}"
                    )
                    
                    statements = fund_data["statements"]
                    
                    if stmt_type == "Income Statement" and not statements["income"].empty:
                        st.caption("Review revenue growth, operating expenses, and net income trajectory.")
                        st.dataframe(statements["income"].style.format("{:,.0f}"), use_container_width=True)
                        
                    elif stmt_type == "Balance Sheet" and not statements["balance"].empty:
                        st.caption("Review total assets, liabilities, and debt levels.")
                        st.dataframe(statements["balance"].style.format("{:,.0f}"), use_container_width=True)
                        
                    elif stmt_type == "Cash Flow" and not statements["cashflow"].empty:
                        st.caption("Review operating cash generation and capital expenditure (CapEx).")
                        st.dataframe(statements["cashflow"].style.format("{:,.0f}"), use_container_width=True)
                    else:
                        st.write("Statement data currently unavailable.")
                        
                    st.divider()
                    
                    # --- SECTION 3: REGULATORY FILINGS & EXECUTIVE COMMENTARY ---
                    st.subheader("🏛️ Corporate Regulatory Filings")
                    st.write("Access official disclosures, 10-K/10-Q management commentary, and strategic outlooks:")
                    
                    col_sec, col_uk = st.columns(2)
                    
                    with col_sec:
                        st.markdown(f"**🇺🇸 US SEC EDGAR Filings**")
                        st.markdown(f"[🔗 Open Official SEC EDGAR Database for {ticker}]({fund_data['sec_url']})")
                        st.caption("Search 10-K (Annual), 10-Q (Quarterly), and 8-K (Material Events) reports.")
                        
                    with col_uk:
                        st.markdown(f"**🇬🇧 UK Companies House**")
                        st.markdown(f"[🔗 Search Companies House UK Registry]({fund_data['uk_url']})")
                        st.caption("Search filing history for UK-listed equities and subsidiaries.")
                else:
                    st.error(f"Could not load fundamental financial statements for {ticker}.")
                    
    if auto_refresh:
        st.caption("Live mode active. Refreshing in 60 seconds...")
        time.sleep(60)
        st.rerun()

else:
    st.info("👈 Set your parameters in the sidebar and click **🚀 Run Analysis** to start the live terminal!")