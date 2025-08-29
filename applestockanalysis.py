import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


df = yf.download("AAPL", start="2024-01-01", end="2025-07-29", group_by='ticker')
df.columns = df.columns.droplevel(0)  # Drop 'Price'
df.columns.name = None  # Clean the column name
print(df.head())
st.write("Rows in DataFrame:", len(df))
st.dataframe(df.head())

df.to_csv()
st.title("🔍Apple Stock Analysis")
st.markdown(
    """
**📌This tool provides interactive analysis of historical stock data**  
using key indicators: **Open, High, Low, Close, Volume, and Date**.

**Explore features like:**
- Daily and long-term price trends  
- Volume analysis to spot trading activity spikes  
- Volatility insights using high-low price ranges  
- Moving averages for trend detection
    """
)
df.reset_index(inplace=True)
st.subheader("🧾Preview of the data for Stocks ")
st.write(df.head())
st.subheader("📋Description of the Overall Data")
st.write(df.describe())

st.subheader("Daily Market Entry and Exit Prices")
st.line_chart(df.set_index('Date')[['Close','Open']])

df['Date'] = pd.to_datetime(df['Date'])
st.subheader("⏳Price Range throughout the Days")
df['Price Range']=df['High']-df['Low']
st.write(df[['Date','Price Range']].sort_values(['Date','Price Range'], ascending=False))
st.line_chart(df.set_index("Date")['Price Range'])

st.subheader("Volume Analysis")
st.dataframe(df[['Date','Volume']].sort_values(['Date','Volume'], ascending=False))
st.bar_chart(df.set_index('Date')['Volume'])

#average_volume=df['Volume'].mean()
#st.write(f"The average trading volume = {average_volume}")
st.subheader("🔍Volume vs Price Range")
st.markdown(
    """
**Volume vs. Price Range Insights:**

- 📈 **High volume + large price range** = strong moves, possibly due to news  
- 🔄 **High volume + small price range** = consolidation or indecision  
- ⚠️ **Low volume + large price range** = suspicious or unstable moves  
    """
)
num_volume=df['Volume'].head(20)
num_pr=df['Price Range'].head(20)
fig, ax= plt.subplots()
ax.scatter(num_volume,num_pr, alpha=0.8, color='blue')
ax.set_xlabel('Volume')
ax.set_ylabel('Price Range')
ax.set_title('Volume vs Price Range')
st.pyplot(fig)




