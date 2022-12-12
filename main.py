import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from google.oauth2 import service_account
from google.cloud import storage

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

from utils import split_data, process_schedule_data, restrict_by_coverage, get_carr_serv_mask, \
    get_reg_train_test, compute_train_val_mae
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
# @st.experimental_memo(ttl=600)
# def read_file(bucket_name, file_path):
#     bucket = client.bucket(bucket_name)

#     blob = bucket.blob(file_path)
#     with blob.open("r") as f:
#         df = pd.read_csv(f)
#     return df

@st.experimental_memo(ttl=600)
def read_file(bucket_name, file_path, is_csv=True):
    bucket = client.bucket(bucket_name)

    blob = bucket.blob(file_path)

    if is_csv:
        with blob.open("r") as f:
            df = pd.read_csv(f)
        return df

    data_bytes = blob.download_as_bytes()

    return pd.read_excel(data_bytes)


@st.experimental_memo(ttl=600)
def read_file_(bucket_name, file_path, is_csv=True, sheet=None):
    bucket = client.bucket(bucket_name)

    blob = bucket.blob(file_path)

    if is_csv:
        with blob.open("r") as f:
            df = pd.read_csv(f)
        return df

    data_bytes = blob.download_as_bytes()

    return pd.read_excel(data_bytes, sheet)


bucket_name = "psa_ontime_streamlit"
# file_path = "0928TableMapping_Reliability-SCHEDULE_RELIABILITY_PP.csv"
file_path = "2022-11-29TableMapping_Reliability-SCHEDULE_RELIABILITY_PP.csv"

# read in reliability schedule data
schedule_data = read_file(bucket_name, file_path)

rel_df_nona = process_schedule_data(schedule_data)
rel_df_nona = restrict_by_coverage(rel_df_nona)

# exclude rows with port code USORF from rel_df since it's missing
rel_df_no_orf = rel_df_nona[~rel_df_nona.POD.isin(["USORF"])]


# PORT PERFORMANCE
file_path = "Port_Performance.xlsx"
# read in reliability schedule data
port_data = read_file(bucket_name, file_path, is_csv=False)

port_call_df = port_data


# process port call data
# ALIGN PORT DATA WITH SCHEDULE
# create new column seaport_code
# for port_call_df and rel_df
# eliminating ambiguous port codes
seaport_code_map= {"CNSHG": "CNSHA", "CNTNJ": "CNTXG", "CNQIN": "CNTAO"}

# add seaport_code column to port data
port_call_df.loc[:, "seaport_code"] = port_call_df["UNLOCODE"].apply(
    lambda x: seaport_code_map[x] if x in seaport_code_map else x
)

# do the same for rel_df
rel_df_no_orf.loc[:, "seaport_code"] = rel_df_no_orf["POD"]

# compute average hours per call
agg_cols = ["seaport_code", "Month", "Year"]
target_cols = ["Total_Calls", "Port_Hours", "Anchorage_Hours"]

# sum up calls, port/anchorage hours
# and aggregate by port, month, and year
port_hours_avg = port_call_df[target_cols + agg_cols].groupby(
    agg_cols
).sum().reset_index()

# average port hours by port, month
port_hours_avg.loc[:, "Avg_Port_Hours(by_call)"] = port_hours_avg[
    "Port_Hours"
] / port_hours_avg["Total_Calls"]

# average anchorage hours by port, month
port_hours_avg.loc[:, "Avg_Anchorage_Hours(by_call)"] = port_hours_avg[
    "Anchorage_Hours"
] / port_hours_avg["Total_Calls"]

port_hours_avg_2022 = port_hours_avg[port_hours_avg["Year"]==2022]

# merge avg hours
rel_df_no_orf_pt_hrs = rel_df_no_orf.merge(
    port_hours_avg_2022,
    left_on=["Calendary_Year", "Month(int)", "seaport_code"],
    right_on=["Year", "Month", "seaport_code"]
)



# schedule + retail

# reliability POL mapping -> retail_sales country/region
rel_port_map = {
    'AEAUH': 'Agg Middle East & Africa',
    'AEJEA': 'Agg Middle East & Africa',
    'BEANR': 'Belgium',
    'BRRIG': 'Brazil',
    'CNNGB': 'China',
    'CNSHA': 'China',
    'CNSHK': 'China',
    'CNTAO': 'China',
    'CNYTN': 'China',
    'COCTG': 'Colombia',
    'DEHAM': 'Denmark',
    'ESBCN': 'Spain',
    'ESVLC': 'Spain',
    'GBLGP': 'U.K.',
    'GRPIR': 'Greece',
    'HKHKG': 'Hong Kong',
    'JPUKB': 'Japan',
    'KRPUS': 'South Korea',
    'LKCMB': 'Agg Asia Pacific',
    'MAPTM': 'Agg Middle East & Africa',
    'MXZLO': 'Mexico',
    'MYPKG': 'Agg Asia Pacific',
    'MYTPP': 'Agg Asia Pacific',
    'NLRTM': 'Netherlands',
    'NZAKL': 'Agg Asia Pacific',
    'PAMIT': 'Agg Latin America',
    'SAJED': 'Agg Middle East & Africa',
    'SAJUB': 'Agg Middle East & Africa',
    'SGSIN': 'Singapore',
    'THLCH': 'Thailand',
    'TWKHH': 'Taiwan',
    'USBAL': 'U.S.',
    'USCHS': 'U.S.',
    'USHOU': 'U.S.',
    'USILM': 'U.S.',
    'USLAX': 'U.S.',
    'USLGB': 'U.S.',
    'USMOB': 'U.S.',
    'USMSY': 'U.S.',
    'USNYC': 'U.S.',
    'USORF': 'U.S.',
    'USSAV': 'U.S.',
    'USTIW': 'U.S.'
}

rel_df_nona.loc[:, "region"] = rel_df_nona["POL"].apply(
    lambda x: rel_port_map[x]
)




# retail sales
sales_filename = "Retail Sales 202210.xlsx"
sales_sheet_name = "Sales"
sales_df = read_file_(
    bucket_name,
    sales_filename,
    sheet=sales_sheet_name,
    is_csv=False
)


# process retail sales data
new_cols = [col.strip() for col in sales_df.columns]
sales_df.columns = new_cols

sales_df.loc[:, "month"] = sales_df["MonthYear"].apply(
    lambda x: int(x.split("/")[0])
)

sales_df.loc[:, "year"] = sales_df["MonthYear"].apply(
    lambda x: int(x.split("/")[1])
)

sales_df.loc[:, "date"] = sales_df["MonthYear"].apply(
    lambda x: datetime.datetime.strptime(
        x, "%m/%Y"
    )
)

# create offset date column
sales_df.loc[:, "date(offset)"] = sales_df['date'] + pd.DateOffset(months=1)

# create a retail sales map given date and country/region
# date, country/region -> retail sales index
regions = [
    'Agg North America', 'U.S.', 'Canada', 'Mexico',
    'Agg Western Europe', 'Austria', 'Belgium', 'Cyprus', 'Denmark',
    'Euro Area', 'Finland', 'France', 'Germany', 'Greece', 'Iceland',
    'Ireland', 'Italy', 'Luxembourg', 'Netherlands', 'Norway', 'Portugal',
    'Spain', 'Sweden', 'Switzerland', 'U.K.', 'Agg Asia Pacific',
    'Australia', 'China', 'Hong Kong', 'Indonesia', 'Japan', 'Kazakhstan',
    'Macau', 'Singapore', 'South Korea', 'Taiwan', 'Thailand', 'Vietnam',
    'Agg Eastern Europe', 'Bulgaria', 'Croatia', 'Czech Republic',
    'Estonia', 'Hungary', 'Latvia', 'Lithuania', 'Poland', 'Romania',
    'Russia', 'Serbia', 'Slovenia', 'Turkey', 'Agg Latin America',
    'Argentina', 'Brazil', 'Chile', 'Colombia', 'Agg Middle East & Africa',
    'Israel', 'South Africa'
]


date_region_sales = {}
for region in regions:
    region_dict = dict(
        zip(
            sales_df["date(offset)"],
            sales_df[region]
        )
    )

    date_region_sales[region] = region_dict


# calculate max date to avoid index error
max_date = sales_df["date(offset)"].max()

# finally, create new columns
# iterate over rows
rel_df_nona.loc[:, "retail_sales"] = rel_df_nona.apply(
    lambda x: date_region_sales[x["region"]][x["Date"]] if x["Date"] <= max_date else None, axis=1
)

rel_df_sales = rel_df_nona.copy()


with st.sidebar:
    label = st.selectbox(
            'Label: ',
            ("Avg_TTDays", "Avg_WaitTime_POD_Days")) #"OnTime_Reliability"))

    # time horizon for train split
    # split_month = st.slider('Time Horizon (month)', 3, 8, 8)
    split_month = st.slider('Time Horizon (month)', 3, 10, 10)
    # split_month = st.slider('Time Horizon (month)', 3, 6, 6)

    include_reg = st.checkbox("Linear Regression")

    overall_pred = st.button("Predict (overall)")


# split date

# baseline
datetime_split = datetime.datetime(2022, split_month, 1)
train_df, val_res = split_data(rel_df_nona, datetime_split, label=label)

# since we only have port call data up to august we restrict val_res
if include_reg:
    val_res = val_res[val_res["Date"] < datetime.datetime(2022, 9, 1)]

    split_month = min(8, split_month)
    datetime_split = datetime.datetime(2022, split_month, 1)

    # linear regression split (port hours)
    train_X_rg, train_y_rg, val_X_rg, val_y_rg = get_reg_train_test(
        rel_df_no_orf_pt_hrs,
        datetime_split,
        label=label
    )

    # linear regression split (retail)
    train_X_rg_ret, train_y_rg_ret, val_X_rg_ret, val_y_rg_ret = get_reg_train_test(
        rel_df_sales,
        datetime_split,
        label=label,
        use_retail=True
    )


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

eval_lin_reg = False

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

            limit_thresh = 81 if label=="Avg_TTDays" else 21

            line = pd.DataFrame(
                {
                    'actual': range(0, limit_thresh),
                    'pred': range(0, limit_thresh)
                }
            )

            line_plot = alt.Chart(line).mark_line(color='red').encode(
                x='actual',
                y='pred'
            )

            st.altair_chart(pred_scatter + line_plot, use_container_width=True)


            if include_reg:
                # evaluate linear regression
                linreg = LinearRegression()

                val_mae_rg, val_mape_rg, val_mae_over_rg, val_mape_over_rg = compute_train_val_mae(
                    linreg,
                    train_X_rg,
                    val_X_rg,
                    train_y_rg,
                    val_y_rg,
                    calc_mape=True,
                    label=label
                )

                linreg = LinearRegression()  # I am no too sure if we need to instantiate twice
                val_mae_rg_ret, val_mape_rg_ret, val_mae_over_rg_ret, val_mape_over_rg_ret = compute_train_val_mae(
                    linreg,
                    train_X_rg_ret,
                    val_X_rg_ret,
                    train_y_rg_ret,
                    val_y_rg_ret,
                    calc_mape=True,
                    label=label
                )

                eval_lin_reg = True

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

        st.markdown("#### Baseline")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE", round(baseline_mae,3))
        col2.metric("MAPE", round(baseline_mape,3))
        col3.metric("MAE (delays)", mae_under)
        col4.metric("MAPE (delays)", mape_under)
        col5.metric("Accuracy (95\% CI)", round(pred_interval_acc, 2))

        if eval_lin_reg:
            st.markdown("#### Linear Regression")
            st.markdown("##### Port Hours")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("MAE", round(val_mae_rg, 3))
            col2.metric("MAPE", round(val_mape_rg, 3))
            col3.metric("MAE (delays)", round(val_mae_over_rg, 3))
            col4.metric("MAPE (delays)", round(val_mape_over_rg, 3))
            col5.metric("Accuracy (95\% CI)", "NA")

            st.markdown("##### Retail Sales")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("MAE", round(val_mae_rg_ret, 3))
            col2.metric("MAPE", round(val_mape_rg_ret, 3))
            col3.metric("MAE (delays)", round(val_mae_over_rg_ret, 3))
            col4.metric("MAPE (delays)", round(val_mape_over_rg_ret, 3))
            col5.metric("Accuracy (95\% CI)", "NA")


    else:
        st.error('All expected labels are zero', icon="🚨")
