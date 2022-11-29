import datetime

import streamlit as st
import altair as alt
import pandas as pd

from google.oauth2 import service_account
from google.cloud import storage


st.set_page_config(page_title="Macroeconomic", page_icon="🚢")

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

# freight (carrier rates)

# Northern Europe to Far East
freight_filename = "Xeneta Freight Rates_TPEB_Far East to USWC_DFE.xlsx"  #"Xeneta Benchmarks and Carrier Spread 2022-08-29 08_33 FEWB.xlsx"
freight_sheet_name = "Far East to USWC"
# freight indext (carrier spread)
# FAR EAST to WEST COAST
freight_df = read_file(
    bucket_name,
    freight_filename,
    sheet=freight_sheet_name,
    is_csv=False
)

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


freight_df.loc[:, "date"] = freight_df["Day"].apply(
    lambda x: datetime.datetime.strptime(
        x, "%Y-%m-%d"
    )
)

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

for pred_str in predictors_str:
    merge_res[pred_str] = freight_df.merge(
        eval(f"{pred_str}_df"),
        on="date"
    )


predictor_cols = dict(
    zip(
        predictors_str,
        [
            "Agg North America",
            "BDI",
            "Agg_North America",
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
            "bdi",
            "cpi",
            "air_freight"
         ])

    plot = st.button("Plot")


if plot:
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
