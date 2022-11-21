import streamlit as st
import pandas as pd
import numpy as np

from google.oauth2 import service_account
from google.cloud import storage


st.title("PSA-ONTIME")

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)

    blob = bucket.blob(file_path)
    with blob.open("r") as f:
        df = pd.read_csv(f)

    return df


bucket_name = "psa_ontime_streamlit"
file_path = "0928TableMapping_Reliability-SCHEDULE_RELIABILITY_PP.csv"


# read in reliability schedule data
schedule_data = read_file(bucket_name, file_path)


if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(schedule_data)



# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# # Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(10000)
# # Notify the reader that the data was successfully loaded.
# data_load_state.text('Done! (using st.cache)')

# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)

# st.subheader('Number of pickups by hours')

# hist_values = np.histogram(
#     data[DATE_COLUMN].dt.hour, bins=24, range=(0,24)
# )[0]

# st.bar_chart(hist_values)

# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# st.map(filtered_data)
