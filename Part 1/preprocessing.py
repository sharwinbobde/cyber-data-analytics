import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder


def relabel(df):
    df_relabeled = df.loc[~(df["simple_journal"] == "Refused")]
    df_relabeled.loc[
        df_relabeled["simple_journal"] == "Chargeback", "simple_journal"
    ] = 1
    df_relabeled.loc[df_relabeled["simple_journal"] == "Settled", "simple_journal"] = 0
    df_relabeled.simple_journal = df_relabeled.simple_journal.astype("int32")
    return df_relabeled


def encode(df, col, values=None):
    """
    encodes the column passed
    returns the df and the encoder object
    """
    enc = LabelEncoder()
    if values is None:
        enc.fit(df[col])
    else:
        enc.fit(values)
    df[col] = enc.transform(df[col])
    return df, enc


def replace_na_with(df, col, replace_value):
    """
    "cardverificationcodesupplied", False,
    "issuercountrycode", "ZZ",
    "shoppercountrycode", "ZZ",


    """
    df.loc[df[col].isna(), col] = replace_value
    return df


def conv_to_eur(row):
    currency_dict = {
        "BGN": 1.9558,
        "NZD": 1.6805,
        "ILS": 4.0448,
        "RUB": 72.2099,
        "CAD": 1.5075,
        "USD": 1.1218,
        "PHP": 58.125,
        "CHF": 1.1437,
        "ZAR": 16.0224,
        "AUD": 1.5911,
        "JPY": 124.93,
        "TRY": 6.6913,
        "HKD": 8.8007,
        "MYR": 4.6314,
        "THB": 35.802,
        "HRK": 7.413,
        "NOK": 9.6678,
        "IDR": 15953.68,
        "DKK": 7.4646,
        "CZK": 25.659,
        "HUF": 322.97,
        "GBP": 0.86248,
        "MXN": 21.2829,
        "KRW": 1308.01,
        "ISK": 136.2,
        "SGD": 1.5263,
        "BRL": 4.405,
        "PLN": 4.2868,
        "INR": 78.0615,
        "RON": 4.7596,
        "CNY": 7.5541,
        "SEK": 10.635,
    }
    return row["amount"] / (currency_dict[row["currencycode"]] * 100)
