import streamlit as st
import altair as alt
import pandas as pd

from sklearn.feature_selection import f_regression
from google.oauth2 import service_account
from google.cloud import storage

from utils import process_schedule_data, restrict_by_coverage


st.set_page_config(page_title="Port Performance", page_icon="⚓")

st.markdown("# Port Dwell")
st.sidebar.header("Port Performance")
st.write(
    """We visualize the relationship between average wait time (schedule data)
    and port dwell time (port performance data) and compute an F-statistic p-value
    as a preliminary check for linear regression on wait time."""
)

# DATA
# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
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


bucket_name = "psa_ontime_streamlit"

# SCHEDULE
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

show_port_call = st.checkbox("Show port call data")
if show_port_call:
    st.write(port_call_df)

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


with st.sidebar:
    POD_options = tuple(
        rel_df_no_orf_pt_hrs["POD"].unique()
    )

    POD_option = st.selectbox(
        'POD: ',
        POD_options)

    # TODO: include in a diff. page
    # carrier_options = tuple(rel_df_no_orf_pt_hrs[
    #     rel_df_no_orf_pt_hrs["POD"]==POD_option
    # ]["Carrier"].unique()
    # )

    # carrier_option = st.selectbox(
    #     'Carrier: ',
    #     carrier_options
    # )

    plot = st.button("Plot")


if plot:

    pod_mask = rel_df_no_orf_pt_hrs["POD"]==POD_option

    # TODO: include in a different page
    # carrier_mask = rel_df_no_orf_pt_hrs["Carrier"]==carrier_option
    # source = rel_df_no_orf_pt_hrs[
    #     pod_mask &
    #     carrier_mask
    # ]

    source = rel_df_no_orf_pt_hrs[
        pod_mask
    ]

    # TODO: implement drop down menu for target labels
    # compute p-value
    label = "Avg_WaitTime_POD_Days"
    predictor_label = "Avg_Port_Hours(by_call)"  #"Avg_Anchorage_Hours"
    target = source[label]

    predictor_format_label = ""

    if predictor_label == "Avg_Anchorage_Hours(by_call)":
        predictor_format_label = "Anchorage"
    else:
        predictor_format_label = "Service"  # Dhaval and Jiahao found service hours include anchorage time

    predictors = source[[predictor_label]]

    p_value = round(f_regression(predictors, target)[1][0], 8)

    # TODO: include in a different page
    # base = alt.Chart(source, title=f"Wait/{predictor_format_label} Time at port {POD_option} for carrier {carrier_option}\n(p-value={p_value})").encode(
    #     alt.X('month(Date):T', axis=alt.Axis(title=None))
    # )

    base = alt.Chart(source, title=f"Wait/{predictor_format_label} Time at port {POD_option} \n(p-value={p_value})").encode(
        alt.X('month(Date):T', axis=alt.Axis(title=None))
    )

    transittime = base.mark_line(stroke='#5276A7', interpolate='monotone').encode(
        alt.Y(f'average({label})',
            axis=alt.Axis(title='Avg. Wait Time (days)', titleColor='#5276A7'))
    )

    anchoragetime = base.mark_line(stroke='green', interpolate='monotone').encode(
        alt.Y(f'average({predictor_label})',
            axis=alt.Axis(title=f'Avg. {predictor_format_label} Time (hours)', titleColor='green'))
    )

    c = alt.layer(transittime, anchoragetime).resolve_scale(
        y = 'independent'
    )

    st.altair_chart(c, use_container_width=True)
