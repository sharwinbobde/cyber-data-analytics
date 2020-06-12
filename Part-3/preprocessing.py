from sklearn.preprocessing import LabelEncoder


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


def drop_background_flows(df):
    filtered_df = df[~(df["Label"] == 2)]
    return filtered_df


def encode_labels(df):

    # normal
    df.loc[
        (df["Label"].str.startswith("flow=From-Normal"))
        | (df["Label"].str.startswith("flow=To-Normal"))
        | (df["Label"].str.startswith("flow=Normal")),
        "Label",
    ] = "0"

    # background
    df.loc[
        (df["Label"].str.startswith("flow=Background"))
        | (df["Label"].str.startswith("flow=To-Background"))
        | (df["Label"].str.startswith("flow=From-Background")),
        "Label",
    ] = "2"

    # botnet
    df.loc[df["Label"].str.startswith("flow=From-Botnet"), "Label"] = "1"

    enc = LabelEncoder()
    enc.fit(df["Label"])
    df.loc[:, "Label"] = enc.transform(df["Label"])

    return df

def encode_features(df):
    """
    Numerically encode all the features
    """

    df.loc[df["Sport"].isna(), "Sport"] = "UNK"
    sport_enc = LabelEncoder()
    df.loc[:, "Sport"] = sport_enc.fit_transform(df["Sport"])

    df.loc[df["Dport"].isna(), "Dport"] = "UNK"
    dport_enc = LabelEncoder()
    df.loc[:, "Dport"] = dport_enc.fit_transform(df["Dport"])

    df.loc[df["State"].isna(), "State"] = "UNK"
    state_enc = LabelEncoder()
    df.loc[:, "State"] = state_enc.fit_transform(df["State"])

    df.loc[df["sTos"].isna(), "sTos"] = -999
    stos_enc = LabelEncoder()
    df.loc[:, "sTos"] = stos_enc.fit_transform(df["sTos"])

    df.loc[df["dTos"].isna(), "dTos"] = -999
    dtos_enc = LabelEncoder()
    df.loc[:, "dTos"] = dtos_enc.fit_transform(df["dTos"])

    proto_enc = LabelEncoder()
    df.loc[:, "Proto"] = proto_enc.fit_transform(df["Proto"])

    dir_enc = LabelEncoder()
    df.loc[:, "Dir"] = dir_enc.fit_transform(df["Dir"])

    srcaddr_enc = LabelEncoder()
    df.loc[:, "SrcAddr"] = srcaddr_enc.fit_transform(df["SrcAddr"])

    dstaddr_enc = LabelEncoder()
    df.loc[:, "DstAddr"] = dstaddr_enc.fit_transform(df["DstAddr"])

    return df
