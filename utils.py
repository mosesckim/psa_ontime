import datetime

import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


# we need to restrict to common inputs
def compute_common_cols(train_X, val_X):
    "Return train and val restricted to common cols"
    # get dummies for categorical variables
    train_X_rg = pd.get_dummies(train_X)
    val_X_rg = pd.get_dummies(val_X)

    # restrict to common columns
    common_cols = list(
        set(train_X_rg.columns).intersection(
            set(val_X_rg.columns)
        )
    )

    return train_X_rg[common_cols], val_X_rg[common_cols]


# def compute_train_val_mae(
#     model,
#     train_X,
#     val_X,
#     train_y,
#     val_y,
#     is_xgboost=False,
#     calc_mape=False,
# ):
#     train_X_rg, val_X_rg = compute_common_cols(train_X, val_X)

#     if is_xgboost:
#         data_dmatrix = xgb.DMatrix(data=train_X_rg, label=train_y)
#         model = xgb.XGBRegressor(
#             objective='reg:linear',
#             colsample_bytree=0.3,
#             learning_rate=0.1,
#             max_depth=5,
#             alpha=10,
#             n_estimators=10
#         )

#     # fit model
#     model.fit(train_X_rg, train_y)

#     # need to make sure reliability predictions are capped at 100 and 0

#     train_preds = list(map(lambda x: 100 if x >= 100 else x, model.predict(train_X_rg)))
#     val_preds = list(map(lambda x: 100 if x >= 100 else x, model.predict(val_X_rg)))

#     train_preds = list(map(lambda x: 0 if x<=0 else x, train_preds))
#     val_preds = list(map(lambda x: 0 if x<=0 else x, val_preds))

#     # evaluate
#     # train MAE
#     train_mae = mean_absolute_error(train_y, train_preds)
#     # val MAE
#     val_mae = mean_absolute_error(val_y, val_preds)

#     # computing MAE for overestimates
#     diff = val_preds - val_y
#     mask = diff > 0
#     val_mae_over = diff[mask].mean()
#     # print("val_preds", pd.Series(val_preds))
#     # print("mask", mask.values)
#     # mask_reset = mask.reset_index().Avg_TTDays
#     # print("val_preds", pd.Series(val_preds)[mask_reset])
#     # print("val_y", val_y)
#     # print("val_y masked", pd.Series(list(val_y))[mask_reset])

#     # mape
#     if calc_mape:
#         val_mape = mean_absolute_percentage_error(val_y, val_preds)
#         print(f"val MAPE: {val_mape}")

#         mask_ser = mask.reset_index().Avg_TTDays
#         val_preds_over = pd.Series(val_preds)[mask_ser]
#         val_y_over = pd.Series(list(val_y))[mask_ser]
#         val_mape_over = mean_absolute_percentage_error(val_y_over, val_preds_over)
#         print(f"val MAPE (overestimates): {val_mape_over}")


#     print(f"train MAE: {train_mae}")
#     print(f"val MAE: {val_mae}")
#     print(f"val MAE (overestimates): {val_mae_over}")


# SPLIT
def split_data(rel_df_nona, datetime_split, label="Avg_TTDays"):

    # train
    train = rel_df_nona[rel_df_nona["Date"] < datetime_split]
    # val
    val = rel_df_nona[rel_df_nona["Date"] >= datetime_split]

    # let's get multi-index pairs from train
    train_indices = list(
        train[
            ["Carrier", "Service", "POD", "POL", label]
        ].groupby(["Carrier", "Service", "POD", "POL"]).mean().index
    )

    # now find the intersection between train an val
    indices_inter = []
    for ind, row in val.iterrows():
        ind_pair = (row["Carrier"], row["Service"], row["POD"], row["POL"])
        if ind_pair in train_indices:
            indices_inter.append(ind)

    # now restrict to the indices in the intersection
    val_res = val.loc[indices_inter, :]

    # group by unique tuples for train data (baseline model)
    # train_on_time_rel_by_carr_ser = train[[
    #     "Carrier", "Service", "POD", "POL", label
    # ]].groupby(["Carrier", "Service", "POD", "POL"]).median().reset_index()
    #
    # train_on_time_rel_by_carr_ser.columns = [
    #     "Carrier", "Service", "POD", "POL", label
    # ]

    # use weighted average
    train_on_time_rel_by_carr_ser = train[[
        "Carrier", "Service", "POD", "POL", label
    ]].groupby(["Carrier", "Service", "POD", "POL"]).apply(
        lambda x: (weighted_average_ser(x[label].values), x[label].values.std())
    ).reset_index()

    train_on_time_rel_by_carr_ser.loc[:, f"{label}"] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[0])
    train_on_time_rel_by_carr_ser.loc[:, f"{label}(std)"] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[1])

    train_on_time_rel_by_carr_ser.drop(0, axis=1, inplace=True)

    train_df = train_on_time_rel_by_carr_ser.copy()

    return train_df, val_res


def weighted_average_ser(ser):

    wts = pd.Series([1 / val if val != 0 else 0 for val in ser])

    if wts.sum() == 0: return 0

    return (ser * wts).sum() / wts.sum()


# process data
def process_schedule_data(schedule_data):
    # process schedule data
    # exclude rows with null reliability values
    rel_df_nona = schedule_data[~schedule_data["OnTime_Reliability"].isna()]

    # add date column
    # convert 3-letter month abbrev to integer equivalent
    rel_df_nona["Month(int)"] = rel_df_nona[
        "Month"
    ].apply(
        lambda x:
        datetime.datetime.strptime(x, '%b').month
    )
    # add date
    rel_df_nona["Date"] = rel_df_nona.apply(
        lambda x: datetime.datetime(
            x["Calendary_Year"], x["Month(int)"], 1
        ), axis=1
    )

    # change target field data type to float
    rel_df_nona.loc[:, "OnTime_Reliability"] = rel_df_nona[
        "OnTime_Reliability"
    ].apply(lambda x: float(x[:-1]))

    # create new variable
    # Avg_TurnoverDays = Avg_TTDays + Avg_WaitTime_POD_Days
    rel_df_nona.loc[:, "Avg_TurnoverDays"] = rel_df_nona[
        "Avg_TTDays"
    ] + rel_df_nona["Avg_WaitTime_POD_Days"]

    return rel_df_nona


# restrict to carrier service routes with all months covered
def restrict_by_coverage(rel_df_nona, min_no_months=9):

    rel_df_nona_cvg = rel_df_nona.groupby(
        ["POL", "POD", "Carrier", "Service"]
    ).apply(lambda x: len(x["Month"].unique())
    )

    rel_df_nona_full_cvg = rel_df_nona_cvg[rel_df_nona_cvg==min_no_months]  # TODO: REMOVE hardcoded value

    rel_df_nona_full_cvg_indices = rel_df_nona_full_cvg.index

    base_features = zip(
        rel_df_nona["POL"],
        rel_df_nona["POD"],
        rel_df_nona["Carrier"],
        rel_df_nona["Service"]
    )

    new_indices = []
    for idx, base_feature in enumerate(base_features):
        if base_feature in rel_df_nona_full_cvg_indices:
            new_indices.append(idx)


    return rel_df_nona.iloc[new_indices, :]


def get_carr_serv_mask(df, carrier, service):

    return (df["Carrier"]==carrier) & \
        (df["Service"]==service)


def add_delay_column(df):

    # df_delay = df[["Month(int)", "OnTime_Reliability"]].copy()

    # df_delay.loc[:, "delay"] = (df_delay["OnTime_Reliability"]==0).apply(
    #     lambda x: "delay" if x else "non-delay"
    # )

    # df_delay.loc[:, "Month(int)"] += 1

    # return df.merge(
    #     df_delay,
    #     on=["Month(int)"]
    # )

    # df.loc[:, "delay"] = (df["OnTime_Reliability"]==0).apply(
    #     lambda x: "delay" if x else "non-delay"
    # )

    df.loc[:, "delay"] = (df["OnTime_Reliability"]==0).apply(
        lambda x: "delay" if x else "non-delay"
    )

    df = gen_lag(
        df,
        target_cols=["delay"]
    )

    return df


def gen_lag(
    df,
    lag=1,
    lag_col="Month(int)",
    target_cols=["OnTime_Reliability"],
    common_cols=["Carrier", "Service", "POL", "POD", "Trade", "Month(int)"]
):

    """Generate month lag on a target column by some number of months.
    """

    # make a copy of dataframe
    # to apply lag
    df_lag = df.copy()

    # apply lag
    df_lag.loc[:, lag_col] += 1

    # rename the target columns

    for target_col in target_cols:
        new_col_name = f"{target_col}_lag_{lag}"
        df_lag.loc[:, new_col_name] = df_lag[target_col]
        df_lag.drop(target_col, inplace=True, axis=1)

    # get common cols to merge on
    # common_cols = ["POL", "POD", "Carrier", "Service"]

    # now merge lagged feature onto original df

    df_with_lag_feature = df.merge(df_lag, on=common_cols)

    df_with_lag_feature.rename(

        columns={
            "Date_x": "Date",
            "Avg_TTDays_x": "Avg_TTDays",
            "Avg_WaitTime_POD_Days_x": "Avg_WaitTime_POD_Days",
            "Avg_Port_Hours(by_call)_x": "Avg_Port_Hours(by_call)",
            "Avg_Anchorage_Hours(by_call)_x": "Avg_Anchorage_Hours(by_call)"
        },
        inplace=True
    )

    return df_with_lag_feature


def get_reg_train_test(df, datetime_split, label='Avg_TTDays', use_retail=False):

    df = add_delay_column(df)

    date_column = "Date"
    # train
    train = df[df[date_column] < datetime_split]

    train_wt_mean = get_train_wt_avg(train, datetime_split, label=label)

    train_wt_mean.columns = [
        'Carrier',
        'Service',
        'POD',
        'POL',
        f'{label}_train',
        f'{label}(std)_train'
    ]

    train_min = get_train_wt_avg(train, datetime_split, label=label, agg_fun=np.min)
    train_min.columns = [
        'Carrier',
        'Service',
        'POD',
        'POL',
        f'{label}_min_train',
        f'{label}(std)_min_train'
    ]



    train_max = get_train_wt_avg(train, datetime_split, label=label, agg_fun=np.max)
    train_max.columns = [
        'Carrier',
        'Service',
        'POD',
        'POL',
        f'{label}_max_train',
        f'{label}(std)_max_train'
    ]


    train = train_wt_mean.merge(train, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    train = train_min.merge(train, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    train = train_max.merge(train, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])




    # val
    val = df[df[date_column] >= datetime_split]

    #
    val = train_wt_mean.merge(val, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    val = train_min.merge(val, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    val = train_max.merge(val, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    # predictors

    if not use_retail:
        predictors = [
            "POL",
            "POD",
            "Carrier",
            "Service",
            "Trade",
            "Avg_Port_Hours(by_call)",
            "Avg_Anchorage_Hours(by_call)",
            f"{label}_train",
            f"{label}(std)_train",
            f"{label}_min_train",
            f"{label}_max_train",
            "delay"
        ]
    else:

        train.rename(columns={"retail_sales_x": "retail_sales"}, inplace=True)
        val.rename(columns={"retail_sales_x": "retail_sales"}, inplace=True)

        # drop retail sales na values
        train = train[~train["retail_sales"].isna()]
        val = val[~val["retail_sales"].isna()]

        predictors = [
            "POL",
            "POD",
            "Carrier",
            "Service",
            "Trade",
            "retail_sales",
            f"{label}_train",
            f"{label}(std)_train",
            f"{label}_min_train",
            f"{label}_max_train",
        ]


    train_X, train_y = train[predictors], train[label]
    val_X, val_y = val[predictors], val[label]

    return train_X, train_y, val_X, val_y


def get_train_wt_avg(rel_df_nona, datetime_split, label="Avg_TTDays", agg_fun=weighted_average_ser):


    train = rel_df_nona[rel_df_nona["Date"] < datetime_split]

    # weighted average
    train_on_time_rel_by_carr_ser = train[[
        "Carrier", "Service", "POD", "POL", label,
    ]].groupby(["Carrier", "Service", "POD", "POL"]).apply(lambda x: (agg_fun(x[label].values), x[label].values.std())).reset_index()

    train_on_time_rel_by_carr_ser.loc[:, f"{label}"] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[0])
    train_on_time_rel_by_carr_ser.loc[:, f"{label}(std)"] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[1])

    train_on_time_rel_by_carr_ser.drop(0, axis=1, inplace=True)

    train_df = train_on_time_rel_by_carr_ser.copy()

    return train_df


# we need to restrict to common inputs
def compute_common_cols(train_X, val_X):
    "Return train and val restricted to common cols"
    # get dummies for categorical variables
    train_X_rg = pd.get_dummies(train_X)
    val_X_rg = pd.get_dummies(val_X)

    # restrict to common columns
    common_cols = list(
        set(train_X_rg.columns).intersection(
            set(val_X_rg.columns)
        )
    )

    return train_X_rg[common_cols], val_X_rg[common_cols]


def compute_train_val_mae(
    model,
    train_X,
    val_X,
    train_y,
    val_y,
    is_xgboost=False,
    calc_mape=False,
    label="Avg_TTDays"
):

    train_X_rg, val_X_rg = compute_common_cols(train_X, val_X)

    if is_xgboost:
        data_dmatrix = xgb.DMatrix(data=train_X_rg, label=train_y)
        model = xgb.XGBRegressor(
            objective='reg:linear',
            colsample_bytree=0.3,
            learning_rate=0.1,
            max_depth=5,
            alpha=10,
            n_estimators=10
        )

    # fit model
    model.fit(train_X_rg, train_y)

    # need to make sure reliability predictions are capped at 100 and 0
    train_preds = list(map(lambda x: 100 if x >= 100 else x, model.predict(train_X_rg)))
    val_preds = list(map(lambda x: 100 if x >= 100 else x, model.predict(val_X_rg)))

    train_preds = list(map(lambda x: 0 if x<=0 else x, train_preds))
    val_preds = list(map(lambda x: 0 if x<=0 else x, val_preds))


    preds_array = np.array(val_preds)

    nonzero_mask = val_y != 0
    nonzero_mask = nonzero_mask.reset_index()[label]


    # filtering zero wait time
    if sum(nonzero_mask) != 0:

        preds = pd.Series(val_preds)[nonzero_mask]

        val_y = val_y.reset_index()[label]
        val_y = val_y[nonzero_mask]

        val_X = val_X.reset_index().drop("index", axis=1)
        val_X = val_X[nonzero_mask]

        preds_array = np.array(preds)

        val_gt = val_y.values

        val_mae = mean_absolute_error(val_gt, preds_array)
        val_mape = mean_absolute_percentage_error(val_gt, preds_array)

    # evaluate
    # train MAE
    train_mae = mean_absolute_error(train_y, train_preds)
    # # val MAE
    # val_mae = mean_absolute_error(val_y, val_preds)

    # computing MAE for overestimates
    # diff = val_preds - val_y

    diff = preds_array - val_y
    mask = diff > 0
    val_mae_over = diff[mask].mean()
    # print("val_preds", pd.Series(val_preds))
    # print("mask", mask.values)
    # mask_reset = mask.reset_index().Avg_TTDays
    # print("val_preds", pd.Series(val_preds)[mask_reset])
    # print("val_y", val_y)
    # print("val_y masked", pd.Series(list(val_y))[mask_reset])

    # mape
    if calc_mape:
        # val_mape = mean_absolute_percentage_error(val_y, val_preds)
        mask_ser = mask.reset_index()[label]
        val_preds_over = pd.Series(preds_array)[mask_ser]
        # val_preds_over = pd.Series(val_preds)[mask_ser]
        val_y_over = pd.Series(list(val_y))[mask_ser]
        # val_y_over = pd.Series(list(val_y))[mask_ser]
        val_mape_over = mean_absolute_percentage_error(val_y_over, val_preds_over)

        return val_mae, val_mape, val_mae_over, val_mape_over

    return val_mae, val_mape



def get_stats(df, label="Avg_TTDays"):
    """Given a route (origin, destination, carrier, service),
       return delay and outlier count
    """

    res = {}
    # count the no. of delays

    if df["OnTime_Reliability"].isna().sum() != 0: print("Found na!")


    if df.shape[0] != 9:

        mean_by_month = df.groupby("Month(int)").mean()

        label_col = df.groupby("Month(int)").mean()[label]
        ontime_col = df.groupby("Month(int)").min()["OnTime_Reliability"]
        months = mean_by_month.reset_index()['Month(int)']


        delay_mask = ontime_col==0

        sigma = label_col.std()
        mu = label_col.mean()

        outlier_mask = np.abs(label_col - mu) > 2*sigma

    else:
        delay_mask = df["OnTime_Reliability"]==0
        # count the no. of outliers (col="Avg_TTDays")
        sigma = df[label].std()
        mu = df[label].mean()
        outlier_mask = np.abs(df[label] - mu) > 2*sigma
        months = df["Month(int)"]


    res = dict(
            zip(
                months,
                zip(delay_mask, outlier_mask)
            )
        )

    return pd.Series(res)