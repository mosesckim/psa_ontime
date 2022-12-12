import datetime

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from google.oauth2 import service_account
from google.cloud import storage
from collections import defaultdict

from utils import process_schedule_data, restrict_by_coverage, get_stats


st.set_page_config(page_title="Macroeconomic", page_icon="📈")

st.markdown("# Macroeconomic Factors")
st.sidebar.header("Macroeconomic factors")
st.write(
    """We visualize the relationship between freight rate and macroeconomic indicators
    for the Far East-US West Coast trading routes"""
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
rel_df_nona = restrict_by_coverage(rel_df_nona)


# read in freight data
# freight (carrier rates)
# carrier filenames
carrier_filenames = [
    "Xeneta Freight Rates_FEEB_N. Europe to Far East_DFE.xlsx",
    "Xeneta Freight Rates_FEWB_Far East to N. Europe_DFE.xlsx",
    "Xeneta Freight Rates_TAEB_USEC to N. Europe_DFE.xlsx",
    "Xeneta Freight Rates_TAWB_N. Europe to USEC_DFE.xlsx",
    "Xeneta Freight Rates_TPEB_Far East to USWC_DFE.xlsx",
    "Xeneta Freight Rates_TPWB_USWC to Far East_DFE.xlsx"
]

# corresp. sheet names
carrier_sheetnames = [
    "N. Europe to Far East",
    "Far East to N. Europe",
    "USEC to N.Europe",
    "N. Europe to USEC",
    "Far East to USWC",
    "USWC to Far East"
]

# carrier df dictionary
freight_filename_dict = dict(
    zip(
        carrier_sheetnames,
        carrier_filenames
    )
)

# now load files
freight_dfs = {}

def add_date_col(df):

    df.loc[
        :, "date"
    ] = df["Day"].apply(
        lambda x: datetime.datetime.strptime(
            x, "%Y-%m-%d"
        )
    )


for route in freight_filename_dict:

    filename = freight_filename_dict[route]

    freight_df = read_file(
        bucket_name,
        filename,
        sheet=route,
        is_csv=False
    )

    # add necessary date columns
    add_date_col(freight_df)

    freight_dfs[route] = freight_df



with st.sidebar:
    route_option = st.selectbox(
        'Route: ',
        carrier_sheetnames)


freight_trade_dict = {
    'N. Europe to Far East': 'Asia-Europe',
    'Far East to N. Europe': 'Asia-Europe',
    'USEC to N.Europe': 'Europe-North America East Coast',
    'N. Europe to USEC': 'Europe-North America East Coast',
    'Far East to USWC': 'Asia-North America West Coast',
    'USWC to Far East': 'Asia-North America West Coast'
}

# restrict trade to chosen route
rel_df_nona_trade = rel_df_nona.copy()

trade_option = freight_trade_dict[route_option]
rel_df_nona_trade = rel_df_nona_trade[
    rel_df_nona_trade["Trade"]==trade_option
]

# restrict pod to chosen route
europe_pods = [
    'BEANR',
    'DEHAM',
    'NLRTM',
    'ITGOA',
    'GBLGP',
    'FRLEH',
]

asian_pods = [
    'CNNGB',
    'CNSHA',
    'CNTAO',
    'KRPUS',
    'MYPKG',
    'HKHKG',
    'LKCMB',
    'JPTYO',
    'JPYOK',
    'CNDLC',
    'VNHPH',
    'CNSHK',
    'CNTXG',
    'THBKK',
    'VNSGN',
    'CNXMN',
    'CNYTN',
    'TWKHH',
    'JPUKB',
]

usec_pods = [
    'USCHS',
    'USNYC',
    'USORF',
    'USSAV',
]

uswc_pods = [
    'USLAX',
    'USLGB',
    'USSEA',
    'USTIW',
]


freight_pod_dict = {
    'N. Europe to Far East': asian_pods,
    'Far East to N. Europe': europe_pods,
    'USEC to N.Europe': europe_pods,
    'N. Europe to USEC': usec_pods,
    'Far East to USWC': uswc_pods,
    'USWC to Far East': asian_pods
}

route_pods = freight_pod_dict[route_option]
rel_df_nona_trade = rel_df_nona_trade[
    rel_df_nona_trade["POD"].isin(route_pods)
]


# with st.sidebar:
#     trade_option = st.selectbox(
#         'Trade: ',
#         trade_options)


months = list(rel_df_nona_trade["Month(int)"].unique())

rel_df_nona_trade = rel_df_nona_trade[
    rel_df_nona_trade["Trade"]==trade_option
]

agg_cols = [
    "POL",
    "POD",
    "Carrier",
    "Service"
]

reL_df_nona_delay_outlier = rel_df_nona_trade.groupby(
    agg_cols
).apply(get_stats).reset_index()



# prepare data frame for grouped bar chart
res_dict = defaultdict(list)

months = [col for col in reL_df_nona_delay_outlier.columns if str(col).isnumeric()]

for ind, row in reL_df_nona_delay_outlier.iterrows():

    for month in months:
        delay, outlier = row[month]

        res_dict["Month"].append(month)
        res_dict["Type"].append("delay")
        res_dict["Count"].append(int(delay))

        res_dict["Month"].append(month)
        res_dict["Type"].append("outlier")
        res_dict["Count"].append(int(outlier))

res_df = pd.DataFrame(res_dict)



# DELAY ANALYSIS BY SPECIFIC ROUTE

# helper method
def convert_to_frame(delay_dict):

    route_delay_df = pd.DataFrame(
        delay_dict
    ).T

    route_delay_df.columns = [
        "Avg_TTDays",
        "delay"
    ]

    # get outliers
    tt_col = route_delay_df["Avg_TTDays"].values

    sigma = tt_col.std()
    mu = tt_col.mean()
    outlier_mask = np.abs(tt_col - mu) > 2*sigma

    above_mean_mask = tt_col > mu


    route_delay_df.loc[:, "outlier"] = outlier_mask

    route_delay_df.loc[:, "above_mean"] = above_mean_mask


    # generate outlier column



    route_delay_df.reset_index(inplace=True)
    route_delay_df.columns = [
        "Month",
        "Avg_TTDays",
        "delay",
        "outlier",
        "above_mean"
    ]

    return route_delay_df


Trade = trade_option #'Asia-North America East Coast'  #'Asia-MEA'
col_label = "Avg_TTDays"
rel_df_nona_delays = rel_df_nona.copy()

delay_col = rel_df_nona_delays["OnTime_Reliability"]==0
rel_df_nona_delays.loc[:, "delay"] = delay_col


rel_df_nona_delays = rel_df_nona_delays[
    rel_df_nona_delays["Trade"]==Trade
]

tt_delay_analysis_by_route = rel_df_nona_delays.groupby(
    [
        "POL",
        "POD",
        "Carrier",
        "Service",
        "Trade"
    ]
).apply(
    lambda x: dict(zip(x["Month(int)"], zip(x[col_label], x["delay"])) )
)

no_indices = len(tt_delay_analysis_by_route)
idx = min(200, no_indices - 1)

route = tt_delay_analysis_by_route.index[idx]

delay_df = convert_to_frame(tt_delay_analysis_by_route[route])
source = delay_df

# alt.Chart(source).mark_circle(size=60).encode(
#     x='Month',
#     y='Avg_TTDays',
#     color='delay',
#     tooltip=['Avg_TTDays', 'delay', 'outlier']
# ).interactive()


# choose freight data given route

freight_df = freight_dfs[route_option]


# # Northern Europe to Far East
# freight_filename = "Xeneta Freight Rates_TPEB_Far East to USWC_DFE.xlsx"  #"Xeneta Benchmarks and Carrier Spread 2022-08-29 08_33 FEWB.xlsx"
# freight_sheet_name = "Far East to USWC"
# # freight indext (carrier spread)
# # FAR EAST to WEST COAST
# freight_df = read_file(
#     bucket_name,
#     freight_filename,
#     sheet=freight_sheet_name,
#     is_csv=False
# )


# AIR FREIGHT
# SHANGHAI TO LAX
air_freight_filename = "AirFrieght total Rate USD per 1000kg Shanghai to Los angeles.xlsx"
air_freight_sheet_name = "Sheet1"
air_freight_df = read_file(
    bucket_name,
    air_freight_filename,
    sheet=air_freight_sheet_name,
    is_csv=False
)


# Baltic Dry Index (BDI)
bdi_filename = "BDI 202210.xlsx"
bdi_sheet_name = "BDI"
bdi_df = read_file(
    bucket_name,
    bdi_filename,
    sheet=bdi_sheet_name,
    is_csv=False
)


# Consumer Price Index
cpi_filename = "CPI Core 202210.xlsx"
cpi_sheet_name = "Core CPI"
cpi_df = read_file(
    bucket_name,
    cpi_filename,
    sheet=cpi_sheet_name,
    is_csv=False
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


# industrial production
ind_prod_filename = "Industrial Production 202210.xlsx"
ind_prod_sheet_name = "IndustrialProduction"
ind_prod_df = read_file(
    bucket_name,
    ind_prod_filename,
    sheet=ind_prod_sheet_name,
    is_csv=False
)


# pre-process data
# freight_df.loc[:, "date"] = freight_df["Day"].apply(
#     lambda x: datetime.datetime.strptime(
#         x, "%Y-%m-%d"
#     )
# )

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


bdi_df.loc[:, "date"] = bdi_df["Date"]


cpi_df.columns = [
    col.strip() for col in cpi_df.columns
]
cpi_df.loc[:, "date"] = cpi_df["MonthYear"].apply(
    lambda x: datetime.datetime.strptime(
        x, "%m/%Y"
    )
)


ind_prod_df.columns = [
    col.strip() for col in ind_prod_df.columns
]
ind_prod_df.loc[:, "date"] = ind_prod_df["MonthYear"].apply(
    lambda x: datetime.datetime.strptime(
        x, "%m/%Y"
    )
)


air_freight_df.columns = ["date", "DAF TA Index"]



# PLOTTING

# merge
predictors = [
    sales_df,
    bdi_df,
    cpi_df,
    ind_prod_df,
    air_freight_df
]

predictors_str = [
    "sales",
    "bdi",
    "cpi",
    "ind_prod",
    "air_freight"
]

merge_res = {}

# restrict dates
datetime_min_thresh = datetime.datetime(2022, 1, 1)
freight_df = freight_df[freight_df["date"]>=datetime_min_thresh]

for pred_str in predictors_str:
    merge_res[pred_str] = freight_df.merge(
        eval(f"{pred_str}_df"),
        on="date"
    )


# predictor_cols = dict(
#     zip(
#         predictors_str,
#         [
#             "Agg North America",
#             "BDI",
#             "Agg_North America",
#             "Canada", #"100(U.S.)",
#             "DAF TA Index"
#         ]
#     )
# )

freight_sales_dict = {
    'N. Europe to Far East': 'Agg Asia Pacific',
    'Far East to N. Europe': 'Euro Area',
    'USEC to N.Europe': 'Euro Area',
    'N. Europe to USEC':'Agg North America',
    'Far East to USWC': 'Agg North America',
    'USWC to Far East': 'Agg Asia Pacific'
}

freight_cpi_dict = {
    'N. Europe to Far East': 'Agg_Asia Pacific',
    'Far East to N. Europe': 'Euro Area',
    'USEC to N.Europe': 'Euro Area',
    'N. Europe to USEC':'Agg_North America',
    'Far East to USWC': 'Agg_North America',
    'USWC to Far East': 'Agg_Asia Pacific'
}



predictor_cols = dict(
    zip(
        predictors_str,
        [
            freight_sales_dict[route_option],
            "BDI",
            freight_cpi_dict[route_option],
            "Canada", #"100(U.S.)",
            "DAF TA Index"
        ]
    )
)


predictor_titles = dict(
    zip(
        predictors_str,
        [
            "Retail Sales",
            "Baltic Dry Index",
            "Consumer Price Index",
            "Industrial Production",
            "Air Freight (SHG -> LAX)"
        ]
    )
)


with st.sidebar:
    predictor_option = st.selectbox(
        'Predictor: ',
        [
            "sales",
            # "bdi",
            "cpi",
            # "air_freight"
         ])

    plot = st.button("Plot")


if plot:

    no_samples = min(5000, res_df.shape[0])
    bar_chart = alt.Chart(res_df.sample(no_samples), title=f"Outlier/Delay analysis for {trade_option}").mark_bar().encode(
        # x=alt.X('Type', scale=alt.Scale(8), title=None),
        x=alt.X('Type', title=None),
        y=alt.Y('mean(Count)', title='Percentage'),
        color='Type',
        column='Month',
        tooltip=['Type', 'mean(Count)']
    ).configure_view(
        stroke='transparent'
    )
    st.altair_chart(bar_chart)


    # delay scatter
    delay_scatter = alt.Chart(source, title=f"Sample Transit Time in \n{Trade}").mark_circle().encode(
        x='Month',
        y='Avg_TTDays',
        color='delay',
        tooltip=['Avg_TTDays', 'delay', 'outlier']
    ).interactive()

    st.altair_chart(delay_scatter, use_container_width=True)


    predictor = predictor_option
    freight_column = "Market Average"
    predictor_title = predictor_titles[predictor]
    predictor_column = predictor_cols[predictor]
    source = merge_res[predictor]



    base = alt.Chart(
        source,
        title=f"Freight(Market Average) vs. {predictor_title}"
    ).encode(
        alt.X('date:T', axis=alt.Axis(format="%Y %B"))
    )

    freight = base.mark_line(stroke='#5276A7', interpolate='monotone').encode(
        alt.Y(f"average({freight_column})",
            axis=alt.Axis(title=f'Avg. Freight Rate', titleColor='#5276A7'))
    )

    predictor = base.mark_line(stroke='green', interpolate='monotone').encode(
        alt.Y(f"average({predictor_column})",
        # alt.Y(predictor_column,
            axis=alt.Axis(title=f'Avg. {predictor_title}', titleColor='green'))
    )

    c = alt.layer(freight, predictor).resolve_scale(
        # y="shared"
        y = 'independent'
    )

    st.altair_chart(c, use_container_width=True)



