import streamlit as st
import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io
import warnings
warnings.filterwarnings('ignore')

# Recession data
recessions = [
    # Official U.S. Recessions (NBER-defined)
    #{"name": "Early 1980s Recession", "start": "1980-01-01", "end": "1980-07-01"},
    #{"name": "1981-1982 Recession", "start": "1981-07-01", "end": "1982-11-01"},
    #{"name": "Savings and Loan Crisis", "start": "1986-01-01", "end": "1995-12-01"}, # Major crisis leading to the next recession
    #{"name": "Early 1990s Recession", "start": "1990-07-01", "end": "1991-03-01"},
    {"name": "Dot-Com Bubble", "start": "2001-03-01", "end": "2001-11-01"},
    {"name": "Global Financial Crisis", "start": "2007-12-01", "end": "2009-06-01"},
    {"name": "COVID-19 Recession", "start": "2020-02-01", "end": "2020-04-01"},
    
    # Major Financial Crises (Global/Regional Impact)
    {"name": "Latin American Debt Crisis", "start": "1982-01-01", "end": "1989-12-01"},
    {"name": "Black Monday", "start": "1987-10-01", "end": "1987-12-01"},
    {"name": "Asian Financial Crisis", "start": "1997-07-01", "end": "1998-12-01"},
    #{"name": "Russian Financial Crisis/LTCM Collapse", "start": "1998-08-01", "end": "1998-12-01"},
    {"name": "European Sovereign Debt Crisis", "start": "2010-01-01", "end": "2014-12-01"},
    {"name": "Liberation Day tariffs", "start": "2025-04-02", "end": "2025-04-10"}
]

# Page configuration
st.set_page_config(
    page_title="Correlation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Correlation Dashboard")

# Model Configuration Section
st.markdown("## Configuration")

# Input widgets
col1, col2, col3 = st.columns(3)

with col1:
    tickers_input = st.text_input("**Tickers ❶**", "GLD SLV", 
                                 help="Enter stock tickers separated by spaces (e.g., 'JNJ CMG SPY')")
    tickers = [ticker.upper().strip() for ticker in tickers_input.split()]
    
with col2:
    start_date = st.date_input("**Start Date ❶**", value=dt.date(1980, 1, 1),
                              help="Select start date for analysis")
    
with col3:
    end_date = st.date_input("**End Date ❶**", value=dt.date.today(),
                            help="Select end date for analysis")

col4, col5 = st.columns(2)

with col4:
    correlation_basis = st.selectbox("**Correlation Basis ❶**", 
                                    ["Daily Returns", "Monthly Returns", "Annual Returns"],
                                    index=0,
                                    help="Select the return frequency for correlation calculation")
    
with col5:
    # Context-aware rolling window slider
    if correlation_basis == "Daily Returns":
        rolling_window = st.slider("**Rolling Correlation Window (Days) ❶**", 
                                  min_value=5, max_value=252, 
                                  value=30, 
                                  help="Window size for rolling correlation in days")
        window_label = f"{rolling_window} Days"
    elif correlation_basis == "Monthly Returns":
        rolling_window = st.slider("**Rolling Correlation Window (Months) ❶**", 
                                  min_value=1, max_value=36, 
                                  value=12, 
                                  help="Window size for rolling correlation in months")
        window_label = f"{rolling_window} Months"
    else:  # Annual Returns
        rolling_window = st.slider("**Rolling Correlation Window (Years) ❶**", 
                                  min_value=1, max_value=10, 
                                  value=3, 
                                  help="Window size for rolling correlation in years")
        window_label = f"{rolling_window} Years"

# Advanced options
with st.expander("Advanced Options"):
    col6, col7 = st.columns(2)
    with col6:
        risk_free_rate = st.number_input("Risk-Free Rate (%)", 
                                        min_value=0.0, 
                                        max_value=10.0, 
                                        value=2.5, 
                                        step=0.1) / 100
    with col7:
        benchmark_ticker = st.text_input("Benchmark Ticker", "SPY")
        
    # Add option to show/hide recession periods
    show_recessions = st.checkbox("Show Recession Periods", value=True)

# Add some spacing
st.markdown("---")

# Process data
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
        
        return annualized_returns, annualized_vol, sharpe_ratios, max_drawdown
    
    return None, None, None, None

def calculate_correlations(ret_df, window):
    if ret_df is None or ret_df.empty:
        return None
        
    # Calculate rolling correlation
    if len(ret_df.columns) > 1:
        roll_corr = ret_df.rolling(window=window).corr().dropna()
        return roll_corr
    return None

def add_recession_shades(fig, data_start_date, data_end_date):
    """Add shaded areas for recession periods that fall within the data range"""
    for recession in recessions:
        rec_start = pd.to_datetime(recession["start"])
        rec_end = pd.to_datetime(recession["end"])
        
        # Only add if the recession period overlaps with our data range
        if rec_end >= data_start_date and rec_start <= data_end_date:
            # Adjust recession dates to fit within our data range
            shade_start = max(rec_start, data_start_date)
            shade_end = min(rec_end, data_end_date)
            
            fig.add_vrect(
                x0=shade_start, 
                x1=shade_end,
                fillcolor="gray", 
                opacity=0.2,
                line_width=0,
                annotation_text=recession["name"],
                annotation_position="top left"
            )
    return fig

# Load and process data
if st.button("Calculate Correlations", type="primary"):
    with st.spinner("Loading data and calculating correlations..."):
        df = load_data(tickers, start_date, end_date)
        
        if df is not None:
            # Check if we have data for all tickers
            missing_tickers = [ticker for ticker in tickers if ticker not in df.columns]
            if missing_tickers:
                st.warning(f"No data found for: {', '.join(missing_tickers)}")
            
            if df.empty:
                st.error("No data available for the selected tickers and date range.")
            else:
                # Calculate returns based on selected basis
                ret_df = calculate_returns(df, correlation_basis)
                
                if ret_df is not None and not ret_df.empty:
                    # Calculate metrics
                    annual_returns, annual_vol, sharpe_ratios, max_drawdown = calculate_metrics(
                        ret_df, risk_free_rate, correlation_basis
                    )
                    
                    # Calculate rolling correlations
                    roll_corr = calculate_correlations(ret_df, rolling_window)
                    
                    # Get the actual date range of our data
                    data_start_date = df.index.min()
                    data_end_date = df.index.max()
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Correlation Results")
                    
                    # Show data availability note
                    min_date = df.index.min()
                    max_date = df.index.max()
                    st.info(f"**Data Availability**: {min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}")
                    st.info(f"**Analysis Parameters**: {correlation_basis} with {window_label} rolling window")
                    
                    # Display correlation matrix
                    overall_corr = ret_df.corr()
                    
                    # Display rolling correlations if we have multiple tickers
                    if len(tickers) > 1 and roll_corr is not None:
                        st.markdown("### Rolling Correlation Time Series")
                        
                        # Create a list of all unique pairs
                        pairs = []
                        for i, ticker1 in enumerate(ret_df.columns):
                            for j, ticker2 in enumerate(ret_df.columns):
                                if i < j:  # Avoid duplicates and self-correlations
                                    pairs.append((ticker1, ticker2))
                        
                        # Let user select which pairs to display
                        selected_pairs = st.multiselect(
                            "Select pairs to display:",
                            options=[f"{p[0]}-{p[1]}" for p in pairs],
                            default=[f"{p[0]}-{p[1]}" for p in pairs[:min(3, len(pairs))]]
                        )
                        
                        if selected_pairs:
                            fig = go.Figure()
                            
                            # Find the first date where we have correlation data
                            first_corr_date = None
                            
                            for pair in selected_pairs:
                                ticker1, ticker2 = pair.split('-')
                                pair_corr = roll_corr.unstack()[ticker1][ticker2]
                                
                                # Remove NaN values from the correlation data
                                pair_corr_clean = pair_corr.dropna()
                                
                                if not pair_corr_clean.empty:
                                    # Track the first date with correlation data
                                    if first_corr_date is None or pair_corr_clean.index[0] < first_corr_date:
                                        first_corr_date = pair_corr_clean.index[0]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=pair_corr_clean.index,
                                        y=pair_corr_clean.values,
                                        mode='lines',
                                        name=pair,
                                        hovertemplate='Date: %{x}<br>Correlation: %{y:.3f}<extra></extra>'
                                    ))
                            
                            if first_corr_date is not None:
                                # Add recession shades if enabled
                                if show_recessions:
                                    fig = add_recession_shades(fig, first_corr_date, data_end_date)
                                
                                # Set x-axis range to match our correlation data
                                fig.update_xaxes(range=[first_corr_date, data_end_date])
                                
                                fig.update_layout(
                                    title=f"Rolling Correlation ({window_label}) with Recession Periods",
                                    xaxis_title="Date",
                                    yaxis_title="Correlation",
                                    hovermode='x unified',
                                    yaxis=dict(range=[-1, 1])
                                )
                                
                                # Add a horizontal line at zero
                                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No correlation data available for the selected pairs after removing NaN values.")
                    
                    # Create detailed results table
                    st.markdown("### Detailed Performance Statistics")
                    
                    results_data = []
                    for ticker in ret_df.columns:
                        row = {
                            "Ticker": ticker,
                            "Annualized Return": f"{annual_returns[ticker]:.2%}" if annual_returns is not None else "N/A",
                            "Annualized Volatility": f"{annual_vol[ticker]:.2%}" if annual_vol is not None else "N/A",
                            "Sharpe Ratio": f"{sharpe_ratios[ticker]:.2f}" if sharpe_ratios is not None else "N/A",
                            "Max Drawdown": f"{max_drawdown[ticker]:.2%}" if max_drawdown is not None else "N/A",
                        }
                        
                        # Add correlation values
                        for other_ticker in ret_df.columns:
                            row[other_ticker] = f"{overall_corr.loc[ticker, other_ticker]:.2f}"
                        
                        results_data.append(row)
                    
                    # Display as table
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Add download options
                    st.markdown("### Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV download
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name="correlation_results.csv",
                            mime="text/csv"
                        )
        else:
            st.error("Failed to load data. Please check your ticker symbols and try again.")

# Add information at the bottom
st.markdown("---")
st.markdown("""
**Notes:**
- Correlations can change significantly during different market regimes
- Consider testing for statistical significance of correlation coefficients
- Rolling correlations help identify time-varying relationships between assets
- For portfolio construction, also consider covariance and beta relationships
- Different rolling window sizes can reveal different patterns - experiment with various timeframes
- Gray shaded areas represent recession periods as defined by the NBER that fall within your data range
""")

# LinkedIN
st.markdown("### Connect with Me!")

st.markdown("""
<a href="https://www.linkedin.com/in/saqif-juhaimee-17322a119/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20">
    Saqif Juhaimee
</a>
""", unsafe_allow_html=True)