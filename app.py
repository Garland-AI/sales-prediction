import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape

# Title for the Streamlit app
st.title('Sales Prediction Dashboard')

# Upload Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    st.write("### Data Overview", df.head())

    # Filter data
    df['Delivery Date'] = pd.to_datetime(df['Delivery Date'], errors='coerce')
    df = df.dropna(subset=['Delivery Date'])
    df['YearMonth'] = df['Delivery Date'].dt.to_period('M')
    df['Unit Price'] = df['Total Sales'] / df['Quantity']
    df['Item Description'] = df['Item Description'].str.strip()
    df['Customer ID'] = df['Customer ID'].str.strip()
    
    # Extract unique item codes and descriptions
    item_dict = df.drop_duplicates(subset=['Itemcode']).set_index('Itemcode')['Item Description'].to_dict()
    
    # Extract unique item codes and unit prices
    unit_price_dict = df.drop_duplicates(subset=['Itemcode']).set_index('Itemcode')['Unit Price'].to_dict()
    
    # Group by 'YearMonth', 'Itemcode', and 'Customer ID'
    grouped = df.groupby(['YearMonth', 'Itemcode', 'Customer ID']).agg({'Total Sales': 'sum'}).reset_index()
    grouped = grouped.sort_values(by=['YearMonth', 'Customer ID', 'Itemcode'])
    
    # Filter out the current month
    current_month = pd.Period.now('M')
    grouped = grouped[grouped['YearMonth'] != current_month]
    grouped = grouped[grouped['Customer ID'] != '']
    
    last_period = grouped['YearMonth'].max()
    filtered_study = grouped[grouped['YearMonth'] == last_period]
    study_items = np.sort(filtered_study['Itemcode'].unique())
    study_customers = np.sort(filtered_study['Customer ID'].unique())
    grouped = grouped[grouped['Itemcode'].isin(study_items) & grouped['Customer ID'].isin(study_customers)]
    
    st.write("### Grouped Data Overview", grouped.head())
    
    study_periods = grouped['YearMonth'].unique()
    all_combinations = pd.MultiIndex.from_product([study_periods, study_items, study_customers], names=['YearMonth', 'Itemcode', 'Customer ID']).to_frame(index=False)
    complete_data = pd.merge(all_combinations, grouped, on=['YearMonth', 'Itemcode', 'Customer ID'], how='left').fillna({'Total Sales': 0})
    complete_data = complete_data.sort_values(by=['YearMonth', 'Customer ID', 'Itemcode'])
    
    # Plot Total Sales per Month
    monthly_sales = grouped.groupby('YearMonth')['Total Sales'].sum().reset_index()
    
    st.write("### Total Sales Per Month")
    fig, ax = plt.subplots()
    ax.plot(monthly_sales['YearMonth'].astype(str), monthly_sales['Total Sales'], marker='o')
    ax.set_title('Total Sales Per Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Sales')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True)
    st.pyplot(fig)

    # Prepare the DataFrame for the specific item IDs and customers
    study_data = complete_data[complete_data['Itemcode'].isin(study_items) & complete_data['Customer ID'].isin(study_customers)]
    pivoted_df = study_data.pivot_table(index=['YearMonth', 'Customer ID'], columns='Itemcode', values='Total Sales').reset_index()
    pivoted_df = pivoted_df.sort_values(by=['YearMonth', 'Customer ID']).drop(columns=["YearMonth", "Customer ID"])
    
    def prepare_data(data, n):
        result = []
        for index in range(0, len(data), n):
            result.append(data[index: index + n])
        return np.array(result)
    
    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps)])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)
    
    # Create sequences
    prepared_data = prepare_data(pivoted_df.values, len(study_customers))
    time_steps = len(prepared_data) // 10
    X, y = create_sequences(prepared_data, time_steps)
    
    st.write("### Data Shape")
    st.write("X shape:", X.shape)
    st.write("y shape:", y.shape)
    
    # Reshape the input data
    X_reshaped = X.reshape(X.shape[0], X.shape[1], -1)
    input_shape = (X_reshaped.shape[1], X_reshaped.shape[2])
    total_size = y.shape[1] * y.shape[2]
    
    
    
