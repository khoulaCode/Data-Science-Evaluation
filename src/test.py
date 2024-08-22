import streamlit as st
import pandas as pd


def load_data(data_path):
    df = pd.read_csv(data_path, sep=';', parse_dates={'Datetime': ['Date', 'Time']}, infer_datetime_format=True, na_values=['?'])
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    return df

# Load the datasets
train_data_url = r"..\TS1\train_household_power_consumption.txt"
test_data_url = r"..\TS1\test_household_power_consumption.txt"

train_data= load_data(train_data_url) 
test_data= load_data(test_data_url) 


st.write("### Training Data Sample")
st.write(train_data.head())

st.write("### Testing Data Sample")
st.write(test_data.head())
