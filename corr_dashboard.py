# Add these imports at the top
from functools import lru_cache

# Page configuration
st.set_page_config(
    page_title="Correlation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the data loading function
@st.cache_data(ttl=3600, show_spinner="Downloading market data...")  # Cache for 1 hour
def load_data(tickers, start_date, end_date):
    try:
        df = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            auto_adjust=True,
            progress=False
        )["Close"]
        return df
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# Cache the returns calculation
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def calculate_returns(df, basis):
    if df is None or df.empty:
        return None
        
    if basis == "Daily Returns":
        ret_df = df.pct_change().dropna()
    elif basis == "Monthly Returns":
        ret_df = df.resample('M').last().pct_change().dropna()
    else:  # Annual Returns
        ret_df = df.resample('Y').last().pct_change().dropna()
    
    return ret_df

# Cache the metrics calculation
@st.cache_data(ttl=1800)
def calculate_metrics(ret_df, risk_free_rate=0.025, basis="Daily Returns"):
    if ret_df is None or ret_df.empty:
        return None, None, None, None
    
    # Calculate annualization factors based on frequency
    if basis == "Daily Returns":
        periods_per_year = 252
    elif basis == "Monthly Returns":
        periods_per_year = 12
    else:  # Annual Returns
        periods_per_year = 1
    
    # Calculate annualized returns and volatility
    if len(ret_df) > 1:
        annualized_returns = ret_df.mean() * periods_per_year
        annualized_vol = ret_df.std() * np.sqrt(periods_per_year)
        sharpe_ratios = (annualized_returns - risk_free_rate) / annualized_vol
        
        # Maximum Drawdown
        cum_returns = (1 + ret_df).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return annualized_returns, annual_vol, sharpe_ratios, max_drawdown
    
    return None, None, None, None

# Cache the correlation calculation (most expensive operation)
@st.cache_data(ttl=1800)
def calculate_correlations(ret_df, window):
    if ret_df is None or ret_df.empty:
        return None
        
    # Calculate rolling correlation
    if len(ret_df.columns) > 1:
        roll_corr = ret_df.rolling(window=window).corr().dropna()
        return roll_corr
    return None

# Cache the recession data processing
@st.cache_data(ttl=86400)  # Cache for 24 hours (recession data doesn't change often)
def get_recession_periods():
    return recessions

# Later in your code, update the recessions reference:
recessions = get_recession_periods()