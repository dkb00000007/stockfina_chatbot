import streamlit as st
import google.generativeai as genai
import json
import yfinance as yf
import pandas as pd
import re
from datetime import datetime, timedelta
import logging # Optional: For better debugging if needed

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Stock Chatbot", page_icon="📈", layout="centered")

# --- Logging Configuration (Optional) ---
# Configure logging to help with debugging if needed
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.info("Starting Streamlit app...")

# --- Configuration & API Key ---
# IMPORTANT: Use Streamlit secrets for your API key!
# 1. Create a file `.streamlit/secrets.toml` in your project directory.
# 2. Add your key like this:
#    GEMINI_API_KEY = "AIzaSy..."

try:
    # Attempt to load the key from Streamlit secrets
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    # logging.info("Gemini API Key configured successfully.")
    # Display success message in sidebar later, after model init
except (FileNotFoundError, KeyError):
    st.error("🚨 Gemini API Key not found! Please add it to Streamlit secrets (`.streamlit/secrets.toml`).")
    st.info("1. Create a folder named `.streamlit` in your project's root directory (if it doesn't exist).\n"
            "2. Inside `.streamlit`, create a file named `secrets.toml`.\n"
            "3. Add the following line to `secrets.toml`, replacing the placeholder with your actual key:\n"
            "   `GEMINI_API_KEY = 'YOUR_ACTUAL_API_KEY'`")
    st.stop() # Stop execution if key is missing
except Exception as e:
    st.error(f"🚨 An error occurred during API key configuration: {e}")
    # logging.error(f"API Key configuration error: {e}")
    st.stop()

# Choose an available Gemini model
# Check https://ai.google.dev/models/gemini for available models
# gemini-1.5-flash is often a good balance of capability and speed
MODEL_NAME = "gemini-1.5-flash"

# --- Gemini Model Initialization ---
try:
    # Add safety settings to potentially reduce refusal/harmful content blocks
    # Adjust thresholds (BLOCK_NONE, BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH)
    # Be aware that BLOCK_NONE might allow undesirable content.
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(model_name=MODEL_NAME, safety_settings=safety_settings)
    st.sidebar.success(f"Gemini Model ({MODEL_NAME}) initialized.", icon="🤖")
    # logging.info(f"Gemini model '{MODEL_NAME}' initialized successfully.")
except Exception as e:
    st.error(f"🚨 Error initializing Gemini model ({MODEL_NAME}): {e}")
    st.error("Possible causes: Invalid API key, incorrect model name, network issues, API quota exceeded.")
    # logging.error(f"Gemini model initialization error: {e}")
    st.stop() # Stop execution if model can't be initialized


# --- Helper Functions ---

def format_chat_history(messages: list[dict], max_history=5) -> str:
    """Formats the last few messages for the LLM prompt, focusing on core content."""
    history_str = ""
    # Take the last 'max_history' messages
    start_index = max(0, len(messages) - max_history)
    for msg in messages[start_index:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Extract just the primary text content for the history prompt
        content = msg.get("content", "").split('\n\n')[0] # Get first part of content if multi-part
        history_str += f"{role}: {content}\n"
    return history_str.strip()

def parse_stock_query(query: str, chat_history: list[dict]) -> dict:
    """Parses the user query using Gemini AI to extract stock details, considering chat history."""
    # logging.info(f"Parsing query: '{query}' with history length: {len(chat_history)}")
    formatted_history = format_chat_history(chat_history, max_history=4) # Limit history length

    system_prompt = f"""
    You are a financial data extraction assistant. Your task is to extract structured information from the user's latest query regarding stocks.
    Use the provided chat history ONLY to understand context if the latest query is ambiguous or incomplete (e.g., "What about GOOGL?" or "Show me YTD").
    Your goal is to determine the stock ticker, the time period, the data interval, and the type of analysis requested.

    Return ONLY a valid JSON object containing the following fields, and nothing else before or after the JSON block:
    {{
        "ticker": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.). Infer from context if possible, default to 'AAPL' if no ticker is mentioned anywhere.",
        "period": "The time period for the data (valid options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'). Infer from context or default to '6mo'.",
        "interval": "The data interval (valid options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'). Infer from context or default to '1d'. Adjust interval logically based on the period (e.g., use '1d' or '1wk' for multi-year periods, not '1m').",
        "analysis_type": "The type of analysis requested (e.g., 'price', 'volume', 'performance', 'trend', 'summary', 'comparison'). Default to 'summary' if unclear."
    }}

    Constraints & Defaults:
    - Focus primarily on the LATEST user query.
    - If multiple tickers are in the latest query, use the first one.
    - Default Ticker: 'AAPL' if no context.
    - Default Period: '6mo'.
    - Default Interval: '1d'. Adjust interval for long periods (>= 1y use '1d', >= 5y consider '1wk' or '1mo'). For short periods (< 5d), '15m' or '1h' might be appropriate if specified.
    - Default Analysis Type: 'summary'.
    - Ensure the ticker symbol is uppercase.
    - Only output the JSON object. Do not add explanations like "Here is the JSON:".

    Chat History (for context):
    {formatted_history}

    Latest User Query: {query}

    JSON Output:
    """
    default_result = {"ticker": "AAPL", "period": "6mo", "interval": "1d", "analysis_type": "summary"}
    json_string = "" # Initialize to handle potential errors

    try:
        response = model.generate_content(system_prompt)

        # Debugging: Print raw response from LLM
        # print("--- LLM Raw Response (parse_stock_query) ---")
        # print(response.text)
        # print("---------------------------------------------")

        # Handle potential refusal/blocking or empty response
        if not response.parts:
             st.error("🚨 The AI response for parsing the query was blocked or empty. This might be due to safety settings or the query content. Please try rephrasing your request.")
             # logging.warning("LLM parsing response was blocked or empty.")
             return default_result

        cleaned_text = response.text.strip()

        # Use regex to find the JSON block robustly, handling potential surrounding text/markdown
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)

        if not json_match:
            st.warning(f"⚠️ LLM did not return a recognizable JSON structure.\nRaw Response:\n```\n{cleaned_text}\n```\nAttempting to use default values.")
            # logging.warning(f"Could not find JSON block in LLM response: {cleaned_text}")
            return default_result

        json_string = json_match.group(0)
        # logging.info(f"Extracted JSON string: {json_string}")
        parsed = json.loads(json_string)

        # Validate and apply defaults, ensuring type safety
        parsed["ticker"] = str(parsed.get("ticker", default_result["ticker"])).upper()
        parsed["period"] = str(parsed.get("period", default_result["period"]))
        parsed["interval"] = str(parsed.get("interval", default_result["interval"]))
        parsed["analysis_type"] = str(parsed.get("analysis_type", default_result["analysis_type"]))

        # --- Refined Period/Interval Validation ---
        period = parsed["period"]
        interval = parsed["interval"]
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

        # Ensure period and interval are valid options recognized by yfinance (basic check)
        if period not in valid_periods:
            st.warning(f"⚠️ Unrecognized period '{period}'. Defaulting to '6mo'.")
            period = '6mo'
        if interval not in valid_intervals:
            st.warning(f"⚠️ Unrecognized interval '{interval}'. Defaulting to '1d'.")
            interval = '1d'

        # Adjust interval based on period if it seems unreasonable
        granular_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        long_periods = ['1y', '2y', '5y', '10y', 'max']
        mid_periods = ['1mo', '3mo', '6mo', 'ytd']

        if period in long_periods and interval in granular_intervals:
            st.warning(f"Interval '{interval}' is too granular for period '{period}'. Changing interval to '1d'.")
            interval = "1d"
        elif period == '5d' and interval in ['1m', '2m']: # 1m/2m often restricted for 5d
             st.warning(f"Interval '{interval}' might not be available for '{period}'. Changing interval to '15m'.")
             interval = "15m" # 5m, 15m, 30m, 1h are usually okay for 5d
        elif period == '1d' and interval not in ['1m', '2m', '5m', '15m', '30m', '60m', '1h']: # 1d needs fine interval
             st.warning(f"Interval '{interval}' is too coarse for period '{period}'. Changing interval to '5m'.")
             interval = "5m" # Default fine interval for 1d

        parsed["period"] = period
        parsed["interval"] = interval

        # logging.info(f"Parsed parameters: {parsed}")
        return parsed

    except json.JSONDecodeError as json_err:
        st.error(f"🚨 Failed to decode the JSON received from the AI. Error: {json_err}")
        st.warning(f"Problematic text received:\n```\n{json_string or cleaned_text}\n```\nUsing default values.")
        # logging.error(f"JSONDecodeError: {json_err}. String: {json_string or cleaned_text}")
        return default_result
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred while parsing the query: {e}")
        # logging.exception("Unexpected error during query parsing")
        st.warning("Using default values due to the parsing error.")
        return default_result

def get_stock_data(ticker: str, period: str, interval: str) -> dict | None:
    """Fetches stock historical data and company info using yfinance."""
    # logging.info(f"Fetching data for Ticker: {ticker}, Period: {period}, Interval: {interval}")
    status_placeholder = st.empty() # Placeholder for status messages
    status_placeholder.info(f"⏳ Fetching data for {ticker} (Period: {period}, Interval: {interval})...")

    try:
        stock = yf.Ticker(ticker)
        info = {} # Initialize info dict

        # 1. Try fetching company info first - helps validate ticker
        try:
            info = stock.info
            # Basic validation: Check if essential info exists. Sometimes 'info' returns even for bad tickers.
            if not info or not info.get('symbol') or (info.get('quoteType') == "MUTUALFUND" and not info.get('shortName')): # Mutual funds might lack symbol sometimes
                 # More checks needed? e.g., empty marketCap for valid stocks? For now, rely on history fetch.
                 pass # Let history fetch be the final validation
            company_name = info.get('shortName') or info.get('longName') or ticker # Default to ticker if name not found
            # logging.info(f"Fetched company info for {ticker}. Name: {company_name}")

        except Exception as info_err:
            # yfinance can sometimes fail on .info even for valid tickers intermittently or if blocked
            st.warning(f"⚠️ Could not fetch detailed company info for {ticker} ({info_err}). Will rely solely on historical data if available.")
            # logging.warning(f"Failed to fetch .info for {ticker}: {info_err}")
            company_name = ticker # Use ticker as name if info fails


        # 2. Fetch historical data
        data = stock.history(period=period, interval=interval)

        # 3. Validate fetched data
        if data.empty:
            st.error(f"❌ No historical data found for ticker '{ticker}' with period '{period}' and interval '{interval}'.")
            st.warning("Possible reasons: Invalid ticker symbol, stock delisted, data unavailable for the requested period/interval combination on Yahoo Finance, temporary network issue.")
            # logging.warning(f"No history data found for {ticker}, {period}, {interval}")
            status_placeholder.empty() # Clear status message
            return None

        # Clean data: Drop rows with NaN Close price if any (should be rare for Close)
        initial_rows = len(data)
        data.dropna(subset=['Close'], inplace=True)
        if len(data) < initial_rows:
             st.warning(f"Note: Removed {initial_rows - len(data)} row(s) with missing 'Close' data.")
             # logging.warning(f"Removed {initial_rows - len(data)} NaN rows for {ticker}")

        if data.empty: # Check again after dropping NaN
             st.error(f"❌ All fetched data for '{ticker}' had missing 'Close' prices for the period/interval.")
             # logging.warning(f"All data had NaN Close for {ticker}, {period}, {interval}")
             status_placeholder.empty()
             return None

        # logging.info(f"Successfully fetched {len(data)} data points for {ticker}.")
        status_placeholder.success(f"✅ Data received for {company_name} ({ticker}).")

        return {
            "data": data, # The pandas DataFrame
            "company_name": company_name,
            "info": info, # The dictionary from stock.info
            "ticker": ticker, # Store for convenience
            "period": period,
            "interval": interval,
        }
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred while fetching data for {ticker}: {e}")
        # logging.exception(f"Error fetching stock data for {ticker}")
        status_placeholder.empty() # Clear status message
        return None

def generate_analysis(stock_info: dict, analysis_type: str) -> str:
    """Generates analysis text using Gemini AI based on fetched data."""
    # logging.info(f"Generating analysis for {stock_info['ticker']}, type: {analysis_type}")
    try:
        df = stock_info["data"]
        if df.empty:
            # logging.warning("Attempted to generate analysis with empty DataFrame.")
            return "No data available to generate analysis."

        ticker = stock_info["ticker"]
        company_name = stock_info["company_name"]
        period = stock_info["period"]
        interval = stock_info["interval"]

        # --- Safely extract key metrics from DataFrame ---
        data_close = df["Close"]
        current_price = data_close.iloc[-1] if not data_close.empty else None
        start_price = data_close.iloc[0] if not data_close.empty else None
        change = None
        percent_change = None

        if current_price is not None and start_price is not None:
            change = current_price - start_price
            if start_price != 0:
                percent_change = (change / start_price) * 100
            elif change == 0:
                 percent_change = 0.0
            else:
                 percent_change = float('inf') # Avoid division by zero, indicate large change

        high = df["High"].max() if not df["High"].empty else None
        low = df["Low"].min() if not df["Low"].empty else None
        avg_volume = df["Volume"].mean() if not df["Volume"].empty and 'Volume' in df else None
        latest_volume = df["Volume"].iloc[-1] if not df["Volume"].empty and 'Volume' in df else None

        # --- Format metrics for the prompt ---
        current_price_str = f"${current_price:.2f}" if current_price is not None else "N/A"
        change_str = f"${change:+.2f}" if change is not None else "N/A" # Add sign
        percent_change_str = f"{percent_change:+.2f}%" if percent_change is not None else "N/A" # Add sign
        high_str = f"${high:.2f}" if high is not None else "N/A"
        low_str = f"${low:.2f}" if low is not None else "N/A"
        avg_volume_str = f"{avg_volume:,.0f}" if avg_volume is not None else "N/A"
        latest_volume_str = f"{latest_volume:,.0f}" if latest_volume is not None else "N/A"


        # --- Prepare prompt for LLM - Tailor focus based on analysis_type ---
        analysis_focus = analysis_type
        if analysis_type == 'summary':
            analysis_focus = "overall performance, including price trends, volatility (high/low range), and average volume"
        elif analysis_type == 'price':
             analysis_focus = "price movements, the starting and ending price, and the period's high and low points"
        elif analysis_type == 'volume':
             analysis_focus = f"trading volume patterns, comparing average volume ({avg_volume_str}) to the latest volume ({latest_volume_str})"
        elif analysis_type == 'performance':
             analysis_focus = f"overall percentage change ({percent_change_str}) and the absolute price change ({change_str}) over the period"
        elif analysis_type == 'trend':
             analysis_focus = "the general direction of the price movement (upward, downward, sideways) and any notable changes in momentum"


        metrics_prompt = f"""
        Analyze the stock performance of {company_name} ({ticker}) based *only* on the provided data for the period '{period}' with a data interval of '{interval}'.

        Focus your analysis specifically on: **{analysis_focus}**.

        Key Data Points Provided:
        - Period Start Price: {f"${start_price:.2f}" if start_price is not None else "N/A"}
        - Period End Price (Current): {current_price_str}
        - Price Change (Period): {change_str} ({percent_change_str})
        - Period High Price: {high_str}
        - Period Low Price: {low_str}
        - Average Daily Volume (Period): {avg_volume_str}
        - Latest Volume: {latest_volume_str}

        Instructions:
        - Provide a concise analysis (2-4 sentences maximum).
        - Base the analysis *strictly* on the provided data points for the given period.
        - Do NOT give financial advice, make predictions, or incorporate external information/news.
        - Do NOT evaluate the company's fundamentals unless data was provided (it wasn't here).
        - Output plain text only. No lists, markdown formatting, or greetings.
        """

        response = model.generate_content(metrics_prompt)

        # Debugging: Print raw response from LLM
        # print("--- LLM Raw Response (generate_analysis) ---")
        # print(response.text)
        # print("----------------------------------------------")


        # Handle potential refusal/blocking
        if not response.parts:
             st.warning("⚠️ AI analysis generation was blocked or empty. This might be due to safety settings or the data provided. Returning a default message.")
             # logging.warning(f"LLM analysis response was blocked or empty for {ticker}")
             return f"AI analysis could not be generated for {ticker} based on the provided data. Please check the data."

        analysis_text = response.text.strip()
        # logging.info(f"Generated analysis for {ticker}: {analysis_text}")
        return analysis_text

    except Exception as e:
        st.error(f"🚨 An unexpected error occurred during analysis generation: {e}")
        # logging.exception(f"Error during analysis generation for {stock_info.get('ticker', 'N/A')}")
        return "Unable to generate analysis at this time due to an internal error."


# --- Streamlit App UI ---

st.title("📈 Stock Analysis Chatbot")
st.caption("Ask about stock prices, performance, and trends in natural language (e.g., 'NVDA YTD summary', 'show me AAPL price last 5 days 15m interval', 'what about MSFT?')")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Optional: Add a welcoming message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! How can I help you with stock information today?",
        "stock_data": None # No data associated with welcome message
        })

# --- Display previous messages ---
# This loop runs on every interaction to rebuild the chat display from history
for msg_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Display the core text content stored in the message history
        st.markdown(msg["content"])

        # --- Check if we need to REDISPLAY visual data associated with this message ---
        # Only assistant messages can have associated stock_data
        if msg["role"] == "assistant" and msg.get("stock_data"):
            stock_info = msg["stock_data"] # Retrieve the stored data dictionary
            df = stock_info["data"] # Get the DataFrame
            company_name = stock_info["company_name"]
            ticker = stock_info["ticker"]
            period = stock_info["period"]
            interval = stock_info["interval"]
            info = stock_info["info"]

            # --- Re-display Chart ---
            st.line_chart(df["Close"], use_container_width=True, color="#007bff")
            st.caption(f"Closing prices for {ticker} ({period}, {interval} interval)")

            # --- Re-display Key Metrics ---
            expander = st.expander("View Key Metrics", expanded=False) # Keep metrics collapsed by default in history
            with expander:
                col1, col2 = st.columns(2)

                # --- Metric Calculation (safe extraction) ---
                data_close = df["Close"]
                current = data_close.iloc[-1] if not data_close.empty else None
                prev_close = data_close.iloc[-2] if len(data_close) > 1 else None
                start_price = data_close.iloc[0] if not data_close.empty else None

                day_change = None
                day_change_pct = None
                day_delta_str = "N/A"
                if current is not None and prev_close is not None:
                    day_change = current - prev_close
                    if prev_close != 0: day_change_pct = (day_change / prev_close) * 100
                    elif day_change == 0: day_change_pct = 0.0
                    else: day_change_pct = float('inf')
                    day_delta_str = f"{day_change:+.2f} ({day_change_pct:+.2f}%)"

                period_change = None
                period_change_pct = None
                period_delta_str = "N/A"
                if current is not None and start_price is not None:
                     period_change = current - start_price
                     if start_price != 0: period_change_pct = (period_change / start_price) * 100
                     elif period_change == 0: period_change_pct = 0.0
                     else: period_change_pct = float('inf')
                     period_delta_str = f"{period_change_pct:+.2f}%"


                # --- Display Metrics using st.metric ---
                with col1:
                    st.metric(
                        label="Last Close Price",
                        value=f"${current:.2f}" if current is not None else "N/A",
                        delta=day_delta_str if interval not in ['1d','5d','1wk','1mo','3mo'] and day_change is not None else None, # Show daily delta only for intra-day intervals
                        delta_color="normal"
                    )
                    period_high = df['High'].max() if not df['High'].empty else None
                    st.metric(label=f"Period High ({period})", value=f"${period_high:.2f}" if period_high is not None else "N/A")

                    market_cap = info.get('marketCap')
                    cap_display = "N/A"
                    if market_cap and isinstance(market_cap, (int, float)):
                        if market_cap >= 1e12: cap_display = f"${market_cap / 1e12:.2f} T"
                        elif market_cap >= 1e9: cap_display = f"${market_cap / 1e9:.2f} B"
                        elif market_cap >= 1e6: cap_display = f"${market_cap / 1e6:.2f} M"
                        else: cap_display = f"${market_cap:,.0f}"
                    st.metric("Market Cap", cap_display)


                with col2:
                    st.metric(
                        label=f"Period Change ({period})",
                        value=f"${period_change:+.2f}" if period_change is not None else "N/A",
                        delta=period_delta_str if period_change_pct is not None else None,
                        delta_color="normal"
                    )
                    period_low = df['Low'].min() if not df['Low'].empty else None
                    st.metric(label=f"Period Low ({period})", value=f"${period_low:.2f}" if period_low is not None else "N/A")

                    pe_ratio = info.get('trailingPE') or info.get('forwardPE') # Prefer trailing
                    pe_display = "N/A"
                    if pe_ratio and isinstance(pe_ratio, (int, float)):
                         pe_display = f"{pe_ratio:.2f}"
                    st.metric("P/E Ratio", pe_display)

            st.caption(f"Data for {company_name} ({ticker}) via yfinance. Displaying stored results.")


# --- User Input Handling ---
if prompt := st.chat_input("Ask about a stock (e.g., 'Show NVDA YTD')"):
    # 1. Add user message to state and display it immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process the request and display assistant response
    with st.chat_message("assistant"):
        # Use a placeholder for spinner and intermediate messages
        processing_placeholder = st.empty()
        processing_placeholder.markdown("🧠 Thinking...")

        # 3. Parse the user query using LLM, considering history
        parsed_params = parse_stock_query(prompt, st.session_state.messages[:-1]) # Pass history *before* this user prompt
        ticker = parsed_params["ticker"]
        period = parsed_params["period"]
        interval = parsed_params["interval"]
        analysis_type = parsed_params["analysis_type"]

        # Clear "Thinking..." and show parameters being used
        processing_placeholder.info(f"🔍 Analyzing request for: **{ticker}** | Period: **{period}** | Interval: **{interval}** | Analysis: **{analysis_type.capitalize()}**")

        # 4. Get stock data using yfinance
        stock_info = get_stock_data(ticker, period, interval) # This function now handles its own status messages

        assistant_response_text = ""
        assistant_stock_data_to_store = None # Data dict to store in session state

        # 5. Generate analysis and display results if data fetch was successful
        if stock_info:
            processing_placeholder.markdown("✍️ Generating analysis...") # Update status
            # Generate analysis text using LLM
            analysis = generate_analysis(stock_info, analysis_type)

            # --- Construct and display the Assistant's full response ---
            processing_placeholder.empty() # Clear status message

            company_name = stock_info["company_name"]
            df = stock_info["data"] # Get DataFrame from fetched data
            info = stock_info["info"]

            # Display analysis text first
            st.markdown(f"**{company_name} ({ticker}) - {period.upper()} Analysis ({analysis_type.capitalize()})**")
            st.markdown(analysis)
            # Prepare the text part to be stored in history
            assistant_response_text = f"**{company_name} ({ticker}) - {period.upper()} Analysis ({analysis_type.capitalize()})**\n\n{analysis}"

            # Display Chart
            st.line_chart(df["Close"], use_container_width=True, color="#007bff")
            st.caption(f"Closing prices ({interval} interval)")

            # Display Key Metrics (in an expander for the latest message too)
            expander = st.expander("View Key Metrics", expanded=True) # Expand by default for latest message
            with expander:
                col1, col2 = st.columns(2)

                # --- Metric Calculation (copied logic for consistency, safe extraction) ---
                data_close = df["Close"]
                current = data_close.iloc[-1] if not data_close.empty else None
                prev_close = data_close.iloc[-2] if len(data_close) > 1 else None
                start_price = data_close.iloc[0] if not data_close.empty else None

                day_change = None
                day_change_pct = None
                day_delta_str = "N/A"
                if current is not None and prev_close is not None:
                    day_change = current - prev_close
                    if prev_close != 0: day_change_pct = (day_change / prev_close) * 100
                    elif day_change == 0: day_change_pct = 0.0
                    else: day_change_pct = float('inf')
                    day_delta_str = f"{day_change:+.2f} ({day_change_pct:+.2f}%)"

                period_change = None
                period_change_pct = None
                period_delta_str = "N/A"
                if current is not None and start_price is not None:
                     period_change = current - start_price
                     if start_price != 0: period_change_pct = (period_change / start_price) * 100
                     elif period_change == 0: period_change_pct = 0.0
                     else: period_change_pct = float('inf')
                     period_delta_str = f"{period_change_pct:+.2f}%"

                # --- Display Metrics using st.metric ---
                with col1:
                    st.metric(
                        label="Last Close Price",
                        value=f"${current:.2f}" if current is not None else "N/A",
                        delta=day_delta_str if interval not in ['1d','5d','1wk','1mo','3mo'] and day_change is not None else None,
                        delta_color="normal"
                    )
                    period_high = df['High'].max() if not df['High'].empty else None
                    st.metric(label=f"Period High ({period})", value=f"${period_high:.2f}" if period_high is not None else "N/A")

                    market_cap = info.get('marketCap')
                    cap_display = "N/A"
                    if market_cap and isinstance(market_cap, (int, float)):
                        if market_cap >= 1e12: cap_display = f"${market_cap / 1e12:.2f} T"
                        elif market_cap >= 1e9: cap_display = f"${market_cap / 1e9:.2f} B"
                        elif market_cap >= 1e6: cap_display = f"${market_cap / 1e6:.2f} M"
                        else: cap_display = f"${market_cap:,.0f}"
                    st.metric("Market Cap", cap_display)

                with col2:
                    st.metric(
                        label=f"Period Change ({period})",
                        value=f"${period_change:+.2f}" if period_change is not None else "N/A",
                        delta=period_delta_str if period_change_pct is not None else None,
                        delta_color="normal"
                    )
                    period_low = df['Low'].min() if not df['Low'].empty else None
                    st.metric(label=f"Period Low ({period})", value=f"${period_low:.2f}" if period_low is not None else "N/A")

                    pe_ratio = info.get('trailingPE') or info.get('forwardPE') # Prefer trailing
                    pe_display = "N/A"
                    if pe_ratio and isinstance(pe_ratio, (int, float)):
                         pe_display = f"{pe_ratio:.2f}"
                    st.metric("P/E Ratio", pe_display)

            # Prepare the data dictionary to be stored in session state
            # IMPORTANT: Ensure data types are suitable for session state if needed
            # For DataFrames, Streamlit usually handles them well.
            assistant_stock_data_to_store = stock_info # Store the whole dict fetched

        else:
            # Handle case where get_stock_data returned None (error occurred/no data)
            processing_placeholder.empty() # Clear status message
            # Error message was already displayed by get_stock_data or parse_stock_query
            assistant_response_text = f"Sorry, I encountered an issue retrieving or processing the data for **{ticker}**. Please check the error messages above."
            # assistant_stock_data_to_store remains None

    # 6. Add the final assistant text response and associated data to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response_text,
        "stock_data": assistant_stock_data_to_store # Store dict or None
    })


# --- Sidebar Information ---
with st.sidebar:
    st.title("💡 Usage Tips")
    st.markdown("""
    Ask questions about specific stocks in natural language. The AI will try to understand your request and fetch the relevant data. It remembers the context of the current session.

    **Examples:**
    - *Show me Apple stock for the last 6 months*
    - *What's the year-to-date performance of GOOGL?*
    - *And TSLA?* (will use previous period/interval context)
    - *Tesla price chart for 5 days with 15 minute interval*
    - *Give me a summary of Microsoft (MSFT) stock over the last year*
    - *Amazon stock data max period volume analysis*

    **Supported Periods:**
    `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

    **Supported Intervals:**
    `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`
    *(Note: Finer intervals like '1m' are only available for shorter recent periods, typically <= 7 days. Longer periods require coarser intervals like '1d' or '1wk')*

    ---
    Powered by Google Gemini and yfinance.
    """)
    st.caption("Disclaimer: This is for informational purposes only and not financial advice. Data provided by Yahoo Finance may have delays or inaccuracies.")