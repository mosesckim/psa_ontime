import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from google.oauth2 import service_account
from google.cloud import storage

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from utils import split_data, process_schedule_data, restrict_by_coverage, get_carr_serv_mask
from baseline import BaselineModel


st.set_page_config(
    page_title="PSA-ONTIME: Schedule",
    page_icon="📅",
)


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
            ("Avg_TTDays", "Avg_WaitTime_POD_Days")) #"OnTime_Reliability"))

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
    preds_std = []
    with st.spinner("Computing predictions..."):
        for ind, row in val_X_filtered.iterrows():
            pred, pred_std = base_model.predict(*row)

            preds.append(pred)
            preds_std.append(pred_std)


    preds_array = np.array(preds)
    preds_std_array = np.array(preds_std)

    nonzero_mask = val_y_filtered != 0
    nonzero_mask = nonzero_mask.reset_index()[label]


    if sum(nonzero_mask) != 0:

        preds = pd.Series(preds)[nonzero_mask]
        preds_std = pd.Series(preds_std)[nonzero_mask]

        val_y_filtered = val_y_filtered.reset_index()[label]
        val_y_filtered = val_y_filtered[nonzero_mask]

        val_X_filtered = val_X_filtered.reset_index().drop("index", axis=1)
        val_X_filtered = val_X_filtered[nonzero_mask]

        preds_array = np.array(preds)
        preds_std_array = np.array(preds_std)

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
            mae_under = round(mae_under, 3)
            mape_under = round(mape_under, 3)
        else:
            mae_under = "NA"
            mape_under = "NA"

        st.subheader("Predictions")
        df_preds = val_X_filtered.copy()
        df_preds.loc[:, "actual"] = val_y_filtered
        df_preds.loc[:, "pred"] = preds_array
        df_preds.loc[:, "error"] = preds_array - val_y_filtered
        df_preds.loc[:, "perc_error"] = (preds - val_y_filtered) / val_y_filtered
        st.write(df_preds)



        # chart_data = pd.DataFrame(
        #     np.random.randn(20, 3),
        #     columns=['a', 'b', 'c'])

        # c = alt.Chart(chart_data).mark_circle().encode(
        #     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

        # st.altair_chart(c, use_container_width=True)
        if overall_pred:
            st.subheader("Error Analysis")

            st.write("""The scatter plot below shows predictions with an absolute percentage error
                        greater than 50 percent and absolute error greater than 4 days.
                    """)


            # filter by error and error percentage
            perc_error_thresh = 0.50
            error_thresh = 4.0

            abs_errors = np.abs(df_preds["error"].values)
            abs_perc_errors = np.abs(df_preds["perc_error"].values)

            df_preds.loc[:, "abs_perc_error"] = 100 * abs_perc_errors

            df_preds.loc[:, "abs_error"] = abs_errors

            pred_outliers = df_preds[
                (abs_perc_errors > perc_error_thresh) &
                (abs_errors > error_thresh)
            ][
                [
                    "Carrier",
                    "Service",
                    "POL",
                    "POD",
                    "actual",
                    "pred",
                    "error",
                    "perc_error",
                    "abs_perc_error",
                    "abs_error"

                ]
            ]

            pred_scatter = alt.Chart(pred_outliers).mark_circle(size=60).encode(
                x='actual',
                y='pred',
                color='Carrier',
                size='abs_error',
                tooltip=['POL', 'POD', 'actual', 'pred', 'perc_error']
            ).interactive()

            st.altair_chart(pred_scatter, use_container_width=True)

        # percentage correct within window
        # window = 5
        # pred_interval_acc = np.mean(np.abs(df_preds["error"]) <= window)
        # st.write(f"Prediction Interval ({window} day(s)): {pred_interval_acc}")

        # # negative errors
        # errors = df_preds["error"]
        # errors_neg = errors[errors < 0]

        # pred_interval_acc = np.mean(np.abs(errors_neg) <= window)
        # st.write(f"Prediction Interval ({window} day(s), negative error): {pred_interval_acc}")

        # prediction interval accuracy
        no_std = 2
        abs_errors = np.abs(df_preds["error"].values)
        # print("len(preds_std_array)", len(preds_std_array))
        # print("len(abs_errors)", len(abs_errors))


        pred_interval_acc = np.mean(abs_errors < no_std * preds_std_array)

        # st.write(f"Accuracy within (within {no_std} standard deviation): {pred_interval_acc}")
        # st.metric("Accuracy (95\% CI)", round(pred_interval_acc, 2))


        st.subheader("Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE", round(baseline_mae,3))
        col2.metric("MAPE", round(baseline_mape,3))
        col3.metric("MAE (delays)", mae_under)
        col4.metric("MAPE (delays)", mape_under)
        col5.metric("Accuracy (95\% CI)", round(pred_interval_acc, 2))

    else:
        st.error('All expected labels are zero', icon="🚨")
