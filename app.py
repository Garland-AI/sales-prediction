import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import gdown

# Inject custom CSS to hide the element
hide_profile_css = """
<style>
    [xpath="//*[@id='root']/div[1]/div/div/div/div/a/img"] {
        display: none;
    }
</style>
"""
st.markdown(hide_profile_css, unsafe_allow_html=True)

# Title for the Streamlit app
st.title('Sales Prediction Dashboard')

under_maintenance = False

if under_maintenance:
    st.write("## Under Maintenance")
    st.write("This app is currently under maintenance. Please check back later.")
    uploaded_file = None
else:
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")  

if uploaded_file is not None:
    params_dir = './params'
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)

    df = pd.read_excel(uploaded_file, engine='openpyxl')
    st.write("### Data Overview", df.head())

    df['Delivery Date'] = pd.to_datetime(df['Delivery Date'], errors='coerce')
    df = df.dropna(subset=['Delivery Date'])
    df['YearMonth'] = df['Delivery Date'].dt.to_period('M')
    df['Unit Price'] = df['Total Sales'] / df['Quantity']
    df['Item Description'] = df['Item Description'].str.strip()
    df['Customer ID'] = df['Customer ID'].str.strip()

    item_dict = df.drop_duplicates(subset=['Itemcode']).set_index('Itemcode')['Item Description'].to_dict()
    unit_price_dict = df.drop_duplicates(subset=['Itemcode']).set_index('Itemcode')['Unit Price'].to_dict()

    grouped = df.groupby(['YearMonth', 'Itemcode', 'Customer ID']).agg({'Total Sales': 'sum'}).reset_index()
    grouped = grouped.sort_values(by=['YearMonth', 'Customer ID', 'Itemcode'])

    current_month = pd.Period.now('M')
    grouped = grouped[grouped['YearMonth'] != current_month]
    grouped = grouped[grouped['Customer ID'] != '']

    last_period = grouped['YearMonth'].max()
    filtered_study = grouped[grouped['YearMonth'] == last_period]

    file_id = '1jsDg72kFWuMiu1W2vmktAs5ot-Nh0RtC'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = './params/study_data.npz'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    data_load = np.load(output, allow_pickle=True)
    study_items = data_load['study_items']
    study_customers = data_load['study_customers']
    study_periods = grouped['YearMonth'].unique()

    grouped = grouped[grouped['Itemcode'].isin(study_items) & grouped['Customer ID'].isin(study_customers)]
    st.write("### Grouped Data Overview", grouped.head())

    all_combinations = pd.MultiIndex.from_product([study_periods, study_items, study_customers], names=['YearMonth', 'Itemcode', 'Customer ID']).to_frame(index=False)
    complete_data = pd.merge(all_combinations, grouped, on=['YearMonth', 'Itemcode', 'Customer ID'], how='left').fillna({'Total Sales': 0})
    complete_data = complete_data.sort_values(by=['YearMonth', 'Customer ID', 'Itemcode'])

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

    scaler = MinMaxScaler()
    pivoted_scaled = scaler.fit_transform(pivoted_df.values)

    prepared_data = prepare_data(pivoted_scaled, len(study_customers))
    time_steps = len(prepared_data) // 10
    X, y = create_sequences(prepared_data, time_steps)

    st.write("### Data Shape")
    st.write("X shape:", X.shape)
    st.write("y shape:", y.shape)

    X_reshaped = X.reshape(X.shape[0], X.shape[1], -1)

    file_id = '1WbZMTszxHKo59R5kIMvA0UtX89EFWyia'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = './params/sales_prediction_model.h5'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    model = load_model(output, custom_objects={'mse': 'mean_squared_error'})
    y_pred = model.predict(X_reshaped)

    y_pred = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[2])).reshape(y_pred.shape)
    y = scaler.inverse_transform(y.reshape(-1, y.shape[2])).reshape(y.shape)

    item_id = st.text_input("Enter Item ID", value=110020)
    customer_id = st.text_input("Enter Customer ID", value='COSTCO-CSC280')

    if st.button("Inspect Prediction"):
        if item_id and customer_id:
            try:
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
            except IndexError:
                st.error("Item ID o Customer ID no encontrados en los datos de entrenamiento.")

    if st.button("Show Prediction Data"):
        last_X = prepared_data[-time_steps:].reshape(1, time_steps, -1)
        last_prediction = model.predict(last_X).reshape(y.shape[1], y.shape[2])
        last_prediction = scaler.inverse_transform(last_prediction.reshape(-1, 1)).reshape(y.shape[1], y.shape[2])

        last_real = y[-1].reshape(y.shape[1], y.shape[2])
        last_train = y_pred[-1].reshape(y.shape[1], y.shape[2])

        max_period = study_periods[-1]
        current_month_name = max_period.strftime('%B')
        next_period = max_period + 1
        next_month_name = next_period.strftime('%B')

        df_pred = pd.DataFrame(last_prediction, index=study_customers, columns=study_items).reset_index().melt(id_vars='index', var_name='Item ID', value_name=f'Predicted Total Sales {next_month_name}').rename(columns={'index': 'Customer ID'})
        df_real = pd.DataFrame(last_real, index=study_customers, columns=study_items).reset_index().melt(id_vars='index', var_name='Item ID', value_name=f'Real Sales {current_month_name}').rename(columns={'index': 'Customer ID'})
        df_train = pd.DataFrame(last_train, index=study_customers, columns=study_items).reset_index().melt(id_vars='index', var_name='Item ID', value_name=f'Predicted Total Sales {current_month_name}').rename(columns={'index': 'Customer ID'})

        merged_df = df_pred.merge(df_real, on=['Customer ID', 'Item ID']).merge(df_train, on=['Customer ID', 'Item ID'])

        merged_df['Unit Price'] = merged_df['Item ID'].map(unit_price_dict)
        merged_df[f'Predicted Quantity {next_month_name}'] = merged_df[f'Predicted Total Sales {next_month_name}'] / merged_df['Unit Price']
        merged_df[f'Real Quantity {current_month_name}'] = merged_df[f'Real Sales {current_month_name}'] / merged_df['Unit Price']
        merged_df[f'Predicted Quantity {current_month_name}'] = merged_df[f'Predicted Total Sales {current_month_name}'] / merged_df['Unit Price']

        merged_df = merged_df.drop(columns=['Unit Price'])[
            ['Customer ID', 'Item ID',
             f'Real Quantity {current_month_name}', f'Real Sales {current_month_name}',
             f'Predicted Quantity {current_month_name}', f'Predicted Total Sales {current_month_name}',
             f'Predicted Quantity {next_month_name}', f'Predicted Total Sales {next_month_name}']
        ]

        st.write("### Prediction Data")
        st.dataframe(merged_df)
