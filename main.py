import datetime

import streamlit as st
import pandas as pd
import numpy as np

from google.oauth2 import service_account
from google.cloud import storage

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from utils import split_data, process_schedule_data, restrict_by_coverage, get_carr_serv_mask
from baseline import BaselineModel


# TITLE
# TODO: alternative?
st.title("PSA-ONTIME")

# INTRODUCTION
st.subheader("Introduction")

st.write("We train a baseline (aggregate) model on shipping schedule data by \
    carrier and service and evaluate it by choosing a time horizon \
    (June by default). In order to gauge model performance, we compute an MAPE \
    (or mean average percentage error), where percentage error is given as below:"
)

st.latex(r'''
            \text{percent error} = \frac{\text{pred} - \text{actual}}{\text{actual}}
''')

st.write("For completeness, we include predictions and an additional metric (i.e. MAE or mean \
    absolute error). To see how models perform on delayed transit times, we compute both\
    metrics on prediction results with negative percent error.")


# DATA
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

rel_df_nona = process_schedule_data(schedule_data)
rel_df_nona = restrict_by_coverage(rel_df_nona)



with st.sidebar:
    label = st.selectbox(
            'Label: ',
            ("Avg_TTDays", "OnTime_Reliability"))

    # time horizon for train split
    split_month = st.slider('Time Horizon (month)', 3, 6, 6)

    overall_pred = st.button("Predict (overall)")


# split date
datetime_split = datetime.datetime(2022, split_month, 2)
train_df, val_res = split_data(rel_df_nona, datetime_split, label=label)

val_X = val_res[["Carrier", "Service", "POD", "POL"]]
val_y = val_res[label]


with st.sidebar:
    carrier_options = tuple(
        val_X["Carrier"].unique()
    )

    carrier_option = st.selectbox(
        'Carrier: ',
        carrier_options)

    service_options = tuple(val_X[
        val_X["Carrier"]==carrier_option
    ]["Service"].unique()
    )

    service_option = st.selectbox(
        'Service: ',
        service_options)

    partial_pred = st.button("Predict (Carrier, Service)")


train_df_filtered = train_df.copy()
val_X_filtered = val_X.copy()
val_y_filtered = val_y.copy()


if partial_pred:
    # train
    train_mask = get_carr_serv_mask(train_df_filtered, carrier_option, service_option)
    train_df_filtered = train_df_filtered[train_mask]
    # val
    val_mask = get_carr_serv_mask(val_X_filtered, carrier_option, service_option)
    val_X_filtered = val_X_filtered[val_mask]
    val_y_filtered = val_y_filtered[val_mask]


if val_X_filtered.shape[0] == 0 or train_df_filtered.shape[0] == 0:
    st.error('Insufficient data, pease choose another split', icon="🚨")


if partial_pred or overall_pred:
    # instantiate baseline model
    base_model = BaselineModel(train_df_filtered, label=label)
    preds = []
    with st.spinner("Computing predictions..."):
        for ind, row in val_X_filtered.iterrows():
            pred = base_model.predict(*row)
            preds.append(pred)

    nonzero_mask = val_y_filtered != 0
    nonzero_mask = nonzero_mask.reset_index()[label]


    if sum(nonzero_mask) != 0:

        preds = pd.Series(preds)[nonzero_mask]
        val_y_filtered = val_y_filtered.reset_index()[label]
        val_y_filtered = val_y_filtered[nonzero_mask]

        val_X_filtered = val_X_filtered.reset_index().drop("index", axis=1)
        val_X_filtered = val_X_filtered[nonzero_mask]

        preds_array = np.array(preds)
        val_gt = val_y_filtered.values

        baseline_mae = mean_absolute_error(val_gt, preds_array)
        baseline_mape = mean_absolute_percentage_error(val_gt, preds_array)

        # calculate underestimates mape
        diff = preds_array - val_gt
        mask = diff < 0

        if sum(mask) != 0:
            preds_array_under = preds_array[mask]
            val_y_values_under = val_gt[mask]
            mae_under = mean_absolute_error(preds_array_under, val_y_values_under)
            mape_under = mean_absolute_percentage_error(val_y_values_under, preds_array_under)
            mae_under = round(mae_under, 2)
            mape_under = round(mape_under, 2)
        else:
            mae_under = "NA"
            mape_under = "NA"

        st.subheader("Predictions")
        df_preds = val_X_filtered.copy()
        df_preds.loc[:, "actual"] = val_y_filtered
        df_preds.loc[:, "pred"] = preds_array

        df_preds.loc[:, "perc_error"] = (preds - val_y_filtered) / val_y_filtered
        st.write(df_preds)


        st.subheader("Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", round(baseline_mae,2))
        col2.metric("MAPE", round(baseline_mape,2))
        col3.metric("MAE (delays)", mae_under)
        col4.metric("MAPE (delays)", mape_under)

    else:
        st.error('All expected labels are zero', icon="🚨")

# END

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
