import streamlit as st
import altair as alt
import pandas as pd

from sklearn.feature_selection import f_regression
from google.oauth2 import service_account
from google.cloud import storage

from utils import process_schedule_data, restrict_by_coverage


st.set_page_config(page_title="Freight", page_icon="🚢")

st.markdown("# Carrier Rate")
st.sidebar.header("Freight")
st.write(
    """Similar to port performance, we visualize the relationship between average transit time
    and carrier rate for a given carrier and port of destination, computing an F-statistic p-value
    as a preliminary check for linear regression on average transit time."""
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
def read_file(bucket_name, file_path, is_csv=True, sheet=None):
    bucket = client.bucket(bucket_name)

    blob = bucket.blob(file_path)

    if is_csv:
        with blob.open("r") as f:
            df = pd.read_csv(f)
        return df

    data_bytes = blob.download_as_bytes()

    return pd.read_excel(data_bytes, sheet)


bucket_name = "psa_ontime_streamlit"

# SCHEDULE
file_path = "0928TableMapping_Reliability-SCHEDULE_RELIABILITY_PP.csv"
# read in reliability schedule data
schedule_data = read_file(bucket_name, file_path)

rel_df_nona = process_schedule_data(schedule_data)
rel_df_nona = restrict_by_coverage(rel_df_nona)

# exclude rows with port code USORF from rel_df since it's missing
rel_df_no_orf = rel_df_nona[~rel_df_nona.POD.isin(["USORF"])]


# CARRIER RATE
carrier_rate_filename = "Xeneta Benchmarks and Carrier Spread 2022-08-29 08_33 FEWB.xlsx"
sheet_name = "Raw Data (Carrier Spread)"

carrier_data = read_file(
    bucket_name,
    carrier_rate_filename,
    is_csv=False,
    sheet=sheet_name
)

freight_df = carrier_data

# restrict to 2022
freight_df.loc[:, "year"] = freight_df["Date"].apply(
    lambda x: int(x[:4])
)

freight_df.loc[:, "month"] = freight_df["Date"].apply(
    lambda x: int(x[5:7])
)

freight_df_2022 = freight_df[freight_df["year"]==2022]

# now group by month and carrier
freight_df_2022_month_carr = freight_df_2022.groupby(
    ["Carrier Name", "month"]
).mean().reset_index()[["Carrier Name", "month", "Carrier Average"]]

# write map dictionary
freight_carrier_map = {
    'American President Line': 'APL',
    'China Ocean Shipping Group': 'COSCO SHIPPING',
    'Evergreen': 'EVERGREEN',
    'Hapag Lloyd': 'HAPAG-LLOYD',
    'Maersk Line': 'MAERSK',
    'Mediterranean Shipping Company': 'MSC',
    'Orient Overseas Container Line': 'OOCL',
    'Yang Ming Lines': 'YANG MING',
    'Hamburg Süd': 'HAMBURG SÜD',
    'HYUNDAI Merchant Marine': 'HMM',
    'Ocean Network Express (ONE)': 'ONE'
}


# create new column
freight_df_2022_month_carr.loc[:, "Carrier"] = freight_df_2022_month_carr[
    "Carrier Name"
].apply(lambda x: freight_carrier_map[x] if x in freight_carrier_map else x)

freight_df_2022_month_carr.drop("Carrier Name", inplace=True, axis=1)

# ALIGN
# merge rollover
rel_df_no_orf_freight = rel_df_no_orf.merge(
    freight_df_2022_month_carr,
    left_on=["Carrier", "Month(int)"],
    right_on=["Carrier", "month"]
)


with st.sidebar:
    POD_options = tuple(
        rel_df_no_orf_freight["POD"].unique()
    )

    POD_option = st.selectbox(
        'POD: ',
        POD_options)

    carrier_options = tuple(rel_df_no_orf_freight[
        rel_df_no_orf_freight["POD"]==POD_option
    ]["Carrier"].unique()
    )

    carrier_option = st.selectbox(
        'Carrier: ',
        carrier_options
    )

    plot = st.button("Plot")


if plot:

    pod_mask = rel_df_no_orf_freight["POD"]==POD_option
    carrier_mask = rel_df_no_orf_freight["Carrier"]==carrier_option

    source = rel_df_no_orf_freight[
        pod_mask &
        carrier_mask
    ]

    # compute p-value
    target = source["Avg_TTDays"]
    predictors = source[["Carrier Average"]]

    p_value = round(f_regression(predictors, target)[1][0], 2)

    base = alt.Chart(source, title=f"Transit Time vs. Carrier Rate at port {POD_option} for carrier {carrier_option}\n(p-value={p_value})").encode(
        alt.X('month(Date):T', axis=alt.Axis(title=None))
    )

    transittime = base.mark_line(stroke='#5276A7', interpolate='monotone').encode(
        alt.Y('average(Avg_TTDays)',
            axis=alt.Axis(title='Avg. Transit Time (days)', titleColor='#5276A7'))
    )

    anchoragetime = base.mark_line(stroke='green', interpolate='monotone').encode(
        alt.Y('average(Carrier Average)',
            axis=alt.Axis(title='Avg. Carrier Rate', titleColor='green'))
    )

    c = alt.layer(transittime, anchoragetime).resolve_scale(
        y = 'independent'
    )

    st.altair_chart(c, use_container_width=True)
