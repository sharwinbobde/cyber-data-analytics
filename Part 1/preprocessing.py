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

    
def preprocess(df):
    df = relabel(df)

    df = replace_na_with(df, "cardverificationcodesupplied", False)
    df = replace_na_with(df, "issuercountrycode", "ZZ")
    df = replace_na_with(df, "shoppercountrycode", "ZZ")

    df = encode(df, "card_id")
    df = encode(df, "ip_id")
    df = encode(df, "txvariantcode")
    df = encode(df, "shopperinteraction")
    df = encode(df, "cardverificationcodesupplied")

    df["creationdate"] = pd.to_datetime(df["creationdate"])
    df["date"] = df["creationdate"].dt.date

    unique_issuer_cc = df["issuercountrycode"].unique().tolist()
    unique_shopper_cc = df["shoppercountrycode"].unique().tolist()
    unique_codes = list(set(unique_issuer_cc + unique_shopper_cc))
    df = encode(df, "issuercountrycode", values=unique_codes)
    df = encode(df, "shoppercountrycode", values=unique_codes)

    df["amount_eur"] = df.apply(lambda x: conv_to_eur(x), axis=1)
    df = encode(df, "currencycode")

    df["accountcode"] = df["accountcode"].apply(lambda x: re.sub("Account", "", x))
    df["accountcode_cc"] = 0
    df.loc[(df["accountcode"] == "UK"), "accountcode_cc"] = "GB"
    df.loc[(df["accountcode"] == "Mexico"), "accountcode_cc"] = "MX"
    df.loc[(df["accountcode"] == "Sweden"), "accountcode_cc"] = "SE"
    df.loc[(df["accountcode"] == "APAC"), "accountcode_cc"] = "APAC"
    df = encode(df, "accountcode_cc")

    df.loc[df["mail_id"].str.contains("na", case=False), "mail_id"] = "email99999"
    df = encode(df, "mail_id")

    df["bookingdate"] = pd.to_datetime(df["bookingdate"])

    df.loc[df["cvcresponsecode"] > 2, "cvcresponsecode"] = 3
    df["countries_equal"] = df["shoppercountrycode"] == df["issuercountrycode"]
    df.loc[df["countries_equal"] == False, "countries_equal"] = 0
    df.loc[df["countries_equal"] == True, "countries_equal"] = 1

    df["day_of_week"] = df["creationdate"].dt.dayofweek
    df["hour"] = df["creationdate"].dt.hour


    return df


if __name__ == "__main__":
    data = "./data/data_for_student_case.csv"
    df = pd.read_csv(data)
    preprocess(df)