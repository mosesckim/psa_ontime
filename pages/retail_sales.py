import datetime

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from google.oauth2 import service_account
from google.cloud import storage
from collections import defaultdict

from utils import process_schedule_data, restrict_by_coverage, get_stats


st.set_page_config(page_title="Retail Sales", page_icon="📈")

st.markdown("# Demand")
st.sidebar.header("Retail Sales")
st.write(
    """We visualize the relationship between average transit time and retail sales
    for a give route (i.e. origin, destination). Since the demand is a precursor to
    changes in supply chain, we offset the date by a month and observe if there is any
    correlation between demand and transit time. Later on, we will look at CPI which
    is a measure of demand related to retail sales but adjusted for inflation."""
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

# schedule analysis


file_path = "2022-11-29TableMapping_Reliability-SCHEDULE_RELIABILITY_PP.csv"
# read in reliability schedule data
schedule_data = read_file(bucket_name, file_path)

rel_df_nona = process_schedule_data(schedule_data)
# rel_df_nona = restrict_by_coverage(rel_df_nona)


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
sales_df = read_file(
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

# troubleshooting
# st.write(rel_df_nona.head())


with st.sidebar:

    # drop down for pol
    pol_options = list(rel_df_nona["POL"].unique())
    pol_option = st.selectbox(
        'POL: ',
        pol_options)

    rel_df_nona_pol = rel_df_nona[
        rel_df_nona["POL"]==pol_option
    ]
    # drop down for pod
    pod_options = list(rel_df_nona_pol["POD"].unique())
    pod_option = st.selectbox(
        'POD: ',
        pod_options
    )

    plot = st.button("Plot")


if plot:

    source = rel_df_nona[
        (rel_df_nona["POD"]==pod_option) &
        (rel_df_nona["POL"]==pol_option)
    ]

    # drop nas
    source = source[~source["retail_sales"].isna()]

    if source.shape[0] == 0:
        st.error('Insufficient data, pease choose another split', icon="🚨")

    else:

        base = alt.Chart(
            source,
            title=[
                "Monthly Retail vs. Avg Transit Time",
                f"(ORIGIN: {pol_option}, DESTINATION: {pod_option})"
            ]
        ).encode(
            alt.X('Date:T', axis=alt.Axis(format="%Y %B"))
        )

        sales = base.mark_line(stroke='#5276A7', interpolate='monotone').encode(
            alt.Y(f"average(retail_sales)",
                axis=alt.Axis(title=f'Avg. Retail Sales', titleColor='#5276A7'))
        )

        predictor = base.mark_line(stroke='green', interpolate='monotone').encode(
            alt.Y("average(Avg_TTDays)",
            # alt.Y(predictor_column,
                axis=alt.Axis(title=f'Avg. Transit Time', titleColor='green'))
        )


        comb_charts = sales + predictor
        transit_retail_graph = alt.layer(sales, predictor).resolve_scale(
            # y="shared"
            y = 'independent'
        )

        st.altair_chart(transit_retail_graph, use_container_width=True)
