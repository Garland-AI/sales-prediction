import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.models import load_model
import os
import gdown
# Title for the Streamlit app
st.title('Sales Prediction Dashboard')
# Upload Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Path to the directory
    params_dir = './params'

    # Create the directory if it doesn't exist
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    else:
        print(f"{params_dir} already exists.")
        
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
    
    # Google Drive file ID from the shared link
    file_id = '1jsDg72kFWuMiu1W2vmktAs5ot-Nh0RtC'
    url = f'https://drive.google.com/uc?id={file_id}'

    # Download the file and save it locally
    output = './params/study_data.npz'
    # Check if the file already exists before downloading
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    else:
        print(f"{output} already exists, skipping download.")

    # Load the data from the downloaded file
    data_load = np.load(output, allow_pickle=True)
    study_items = data_load['study_items']
    study_customers = data_load['study_customers']
    study_periods = grouped['YearMonth'].unique()

    grouped = grouped[grouped['Itemcode'].isin(study_items) & grouped['Customer ID'].isin(study_customers)]
    
    st.write("### Grouped Data Overview", grouped.head())
    
   
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
    
    
    
    
    # Define the model architecture
    X_reshaped = X.reshape(X.shape[0], X.shape[1], -1)  # Reshape the input data
    input_shape = (X_reshaped.shape[1], X_reshaped.shape[2])  # Define the input shape
    total_size = y.shape[1] * y.shape[2]  # Total size for reshaping

    # Define custom objects dictionary

    # Google Drive file ID from the shared link
    file_id = '1WbZMTszxHKo59R5kIMvA0UtX89EFWyia'
    url = f'https://drive.google.com/uc?id={file_id}'

    # Download the file and save it locally
    output = './params/sales_prediction_model.h5'
    # Check if the file already exists before downloading
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    else:
        print(f"{output} already exists, skipping download.")

    # Load the model with custom objects
    model = load_model(output, custom_objects={'mse': 'mean_squared_error'})
 

    # Make predictions
    y_pred = model.predict(X_reshaped)


    # User input for item_id and customer_id
    item_id = st.text_input("Enter Item ID", value=110020)
    customer_id = st.text_input("Enter Customer ID", value='COSTCO-CSC280')

    if st.button("Inspect Prediction"):
        # Convert inputs to the appropriate format
        item_id = int(item_id)
        item_index = np.where(study_items == item_id)[0][0]
        customer_index = np.where(study_customers == customer_id)[0][0]


        predicted_data = y_pred[:, customer_index, item_index]
        true_data = y[:, customer_index, item_index]
        plot_index = study_periods.astype(str)

        st.write("### Real vs. Predicted Sales")
        fig, ax = plt.subplots()
        ax.plot(plot_index[time_steps:], true_data, label='Actual Sales')
        ax.plot(plot_index[time_steps:], predicted_data, label='Predicted Sales', linestyle='--')
        ax.set_title(f'Sales for Item: {item_dict[item_id]}, Customer: {customer_id}')
        ax.set_xlabel('YearMonth')
        ax.set_ylabel('Total Sales')
        ax.legend()
        ax.tick_params(axis='x', rotation=90)
        last_pred_value = predicted_data[-1]
        ax.text(plot_index[-1], last_pred_value, f'{last_pred_value:.2f}', color='red')
        st.pyplot(fig)
    
    if st.button("Show Prediction Data"):
        max_period = study_periods[-1]

        # Get the name of the current month
        current_month_name = max_period.strftime('%B')

        # Calculate the next period and get the name of the next month
        next_period = max_period + 1
        next_month_name = next_period.strftime('%B')


        last_X = prepared_data[-time_steps:].reshape(1, time_steps, -1)

        last_prediction = model.predict(last_X).reshape(y.shape[1] , y.shape[2])

        last_real = y[-1].reshape(y.shape[1] , y.shape[2])

        last_train = y_pred[-1].reshape(y.shape[1] , y.shape[2])

        last_prediction_df = pd.DataFrame(last_prediction, index=study_customers, columns=study_items).reset_index().melt(id_vars='index', var_name='Item ID', value_name=f'Predicted Total Sales {next_month_name}').rename(columns={'index': 'Customer ID'})

        last_train_df = pd.DataFrame(last_train, index=study_customers, columns=study_items).reset_index().melt(id_vars='index', var_name='Item ID', value_name=f'Predicted Total Sales {current_month_name}').rename(columns={'index': 'Customer ID'})

        # Assuming last_real and last_prediction have the same shape
        last_real_df = pd.DataFrame(last_real, index=study_customers, columns=study_items).reset_index().melt(id_vars='index', var_name='Item ID', value_name=f'Real Sales {current_month_name}').rename(columns={'index': 'Customer ID'})


        merged_df = pd.merge(last_prediction_df, last_real_df , on=['Customer ID', 'Item ID'], how='left')


        merged_df =  pd.merge(merged_df, last_train_df , on=['Customer ID', 'Item ID'], how='left')




        merged_df = merged_df[(merged_df[f'Predicted Total Sales {next_month_name}'] > 0 ) & (merged_df[f'Predicted Total Sales {current_month_name}'] > 0)]


        # Map the unit prices from the dictionary to a new column
        merged_df['Unit Price'] = merged_df['Item ID'].map(unit_price_dict)

        # Calculate the quantity
        merged_df[f'Predicted Quantity {next_month_name}'] = merged_df[f'Predicted Total Sales {next_month_name}'] / merged_df['Unit Price']

        merged_df[f'Real Quantity {current_month_name}'] = merged_df[f'Real Sales {current_month_name}'] / merged_df['Unit Price']

        merged_df[f'Predicted Quantity {current_month_name}'] = merged_df[f'Predicted Total Sales {current_month_name}'] / merged_df['Unit Price']

        merged_df = merged_df.drop(columns=['Unit Price'])[['Customer ID', 'Item ID',f'Real Quantity {current_month_name}', f'Real Sales {current_month_name}' , f'Predicted Quantity {current_month_name}' , f'Predicted Total Sales {current_month_name}',f'Predicted Quantity {next_month_name}', f'Predicted Total Sales {next_month_name}'] ]
        st.write("### Prediction Data")       
        st.dataframe(merged_df)
      

    
    
    
    
    
