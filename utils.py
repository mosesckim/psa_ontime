import datetime

import xgboost as xgb
import pandas as pd
import streamlit as st

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


def compute_train_val_mae(
    model,
    train_X,
    val_X,
    train_y,
    val_y,
    is_xgboost=False,
    calc_mape=False,
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

    # evaluate
    # train MAE
    train_mae = mean_absolute_error(train_y, train_preds)
    # val MAE
    val_mae = mean_absolute_error(val_y, val_preds)

    # computing MAE for overestimates
    diff = val_preds - val_y
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
        val_mape = mean_absolute_percentage_error(val_y, val_preds)
        print(f"val MAPE: {val_mape}")

        mask_ser = mask.reset_index().Avg_TTDays
        val_preds_over = pd.Series(val_preds)[mask_ser]
        val_y_over = pd.Series(list(val_y))[mask_ser]
        val_mape_over = mean_absolute_percentage_error(val_y_over, val_preds_over)
        print(f"val MAPE (overestimates): {val_mape_over}")


    print(f"train MAE: {train_mae}")
    print(f"val MAE: {val_mae}")
    print(f"val MAE (overestimates): {val_mae_over}")


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
def restrict_by_coverage(rel_df_nona):

    rel_df_nona_cvg = rel_df_nona.groupby(
        ["POL", "POD", "Carrier", "Service"]
    ).apply(lambda x: len(x["Month"].unique())
    )

    rel_df_nona_full_cvg = rel_df_nona_cvg[rel_df_nona_cvg==7]  # TODO: REMOVE hardcoded value

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
