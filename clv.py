import streamlit as st
#!pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format",lambda x: "%.4f" % x)

def outlier_tresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range=quartile3-quartile1
    up_limit = quartile3+1.5*interquantile_range
    low_limit = quartile1-1.5*interquantile_range
    return low_limit,up_limit

def replace_with_tresholds(dataframe,variable):
    low_limit,up_limit = outlier_tresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable]>up_limit),variable] = up_limit

# Title
st.title('Customer Lifetime Value Analysis')

# Sample CLV data (you should replace this with your own data)
data = pd.read_csv('clv.csv', sep=';')

data['invoice_date'] = pd.to_datetime(data['invoice_date'])
data.isnull().sum()
data.dropna(inplace=True)

#Suppressing outliers
replace_with_tresholds(data,"total_price")

data["totalprice"]=data["total_price"]

#Using the maximum date as toSday's date and setting date for analysis
today_date= data['invoice_date'].max()

#Preparation of Lifetime Data Structure 
#P.S Everything we do we will assume customers bought weekly
cltv_df = data.groupby('customer_id').agg({
    "invoice_date": [lambda invoice_date:(invoice_date.max()-invoice_date.min()).days,lambda date:(today_date-date.min()).days],
    'invoice': lambda num: num.nunique(),
    "totalprice": lambda totalprice: totalprice.sum()})
#Renaming the columns to match the RMF Analysis

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency","T","frequency","monetary"]

#We need people who bought atleast twice
cltv_df=cltv_df[(cltv_df["frequency"]>1)]
cltv_df=cltv_df[(cltv_df["monetary"]>1)]

#calculate the average earnings per transaction
cltv_df["monetary"] = cltv_df["monetary"]/cltv_df["frequency"]

#expressing the recency value in weekly terms
cltv_df["recency"]=cltv_df["recency"]/7

#calculating how many weeks the customer has been our customer
cltv_df["T"]=cltv_df["T"]/7

#Modelling part(Buy till you die), BG/NBD ( Beta Geometric / Negative Binomial Distribution) model
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],cltv_df["recency"],cltv_df["T"])

#Predictions for multiple periods
cltv_df['estiamted_purchase_1_week'] = bgf.predict(1,cltv_df["frequency"],cltv_df["recency"],cltv_df["T"])
cltv_df['estiamted_purchase_1_month'] = bgf.predict(4,cltv_df["frequency"],cltv_df["recency"],cltv_df["T"])
cltv_df['estiamted_purchase_3_month'] = bgf.predict(12,cltv_df["frequency"],cltv_df["recency"],cltv_df["T"])

#Using the Gamma-Gamma Model to predict the average profit

ggf=GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"],cltv_df["monetary"])
ggf.conditional_expected_average_profit(cltv_df["frequency"],cltv_df["monetary"]).head()
ggf.conditional_expected_average_profit(cltv_df["frequency"],cltv_df["monetary"]).sort_values(ascending=False).head()
cltv_df["expected_average_profit"]=ggf.conditional_expected_average_profit(cltv_df["frequency"],cltv_df["monetary"])
cltv_df.sort_values("expected_average_profit",ascending=False)
#Calculation of CLTV with BG/NBD and Gamma Gamma Model


cltv=ggf.customer_lifetime_value(bgf,
                                 cltv_df["frequency"],
                                 cltv_df["recency"],
                                 cltv_df["T"],
                                 cltv_df["monetary"],
                                 time=3 ,   
                                 freq="W", 
                                 discount_rate=0.01)
cltv.head()
cltv=cltv.reset_index()
cltv_final=cltv_df.merge(cltv,on="customer_id",how="left")
cltv_final.sort_values(by="clv",ascending=False).head(10)

# Sidebar filters
st.sidebar.header('Data Filters')
min_recency = st.sidebar.slider('Minimum Recency', min_value=0, max_value=90, value=0)
min_frequency = st.sidebar.slider('Minimum Frequency', min_value=0, max_value=10, value=0)
min_monetary = st.sidebar.slider('Minimum Monetary Value', min_value=0, max_value=250, value=0)

# Filter data based on user selections
filtered_data = cltv_final[(cltv_final['recency'] >= min_recency) & (cltv_final['frequency'] >= min_frequency) & (cltv_final['monetary'] >= min_monetary)]

# Display the filtered data
st.subheader('Filtered Customer Data')
st.write(filtered_data)

# CLV Distribution Plot
st.subheader('Customer Lifetime Value Distribution')
plt.hist(cltv_final['clv'], bins=20)
st.pyplot()

# Average CLV
average_clv = filtered_data['clv'].mean()
st.subheader('Average Customer Lifetime Value')
st.write(f'The average CLV for the selected customers is R{average_clv:.2f}')

# CLV by Customer
st.subheader('Customer Lifetime Value by Customer')
customer_select = st.selectbox('Select a Customer', filtered_data['customer_id'].tolist())
if customer_select:
    selected_customer = filtered_data[filtered_data['customer_id'] == customer_select]
    st.write(selected_customer)

# CLV vs Recency Scatter Plot
st.subheader('CLV vs Recency')
plt.scatter(filtered_data['recency'], filtered_data['clv'])
plt.xlabel('Recency')
plt.ylabel('CLV')
st.pyplot()

# CLV vs Frequency Scatter Plot
st.subheader('CLV vs Frequency')
plt.scatter(filtered_data['frequency'], filtered_data['clv'])
plt.xlabel('Frequency')
plt.ylabel('CLV')
st.pyplot()

# CLV vs Monetary Value Scatter Plot
st.subheader('CLV vs Monetary Value')
plt.scatter(filtered_data['monetary'], filtered_data['clv'])
plt.xlabel('Monetary Value')
plt.ylabel('CLV')
st.pyplot()
