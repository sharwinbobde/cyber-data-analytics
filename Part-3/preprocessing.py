import pandas as pd
from sklearn.preprocessing import LabelEncoder


def drop_background_flows(df):
    filtered_df = df[~(df["Label"] == 2)]
    return filtered_df

def encode_labels(df):
    
    # normal
    df.loc[
        (df["Label"].str.startswith("flow=From-Normal"))|
        (df["Label"].str.startswith("flow=To-Normal"))|
        (df["Label"].str.startswith("flow=Normal")), "Label"] = "0"

    # background
    df.loc[
        (df["Label"].str.startswith("flow=Background"))|
        (df["Label"].str.startswith("flow=To-Background"))|
        (df["Label"].str.startswith("flow=From-Background")), "Label"] = "2"
    
    # botnet
    df.loc[df["Label"].str.startswith("flow=From-Botnet"), "Label"] = "1"
    
    enc = LabelEncoder()
    enc.fit(df["Label"])
    df.loc[:, "Label"] = enc.transform(df["Label"])
    
    return df