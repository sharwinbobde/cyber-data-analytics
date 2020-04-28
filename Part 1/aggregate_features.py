import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

# ## Read the Data

# In[3]:


data = "./data/data_for_student_case.csv"
df1 = pd.read_csv(data)

# ## Data Preprocessing

# In[7]:


# Prepare class label, card id, ip id and date for plotting
# 1.Class label
df1 = df1.loc[~(df1["simple_journal"] == "Refused")]
df1.loc[df1["simple_journal"] == "Chargeback", "simple_journal"] = 1  # fraud
df1.loc[df1["simple_journal"] == "Settled", "simple_journal"] = 0

# 2.Card ID
card_enc = LabelEncoder()
card_enc.fit(df1["card_id"])
df1["card_id"] = card_enc.transform(df1.card_id)

# 3.IP ID
ip_enc = LabelEncoder()
ip_enc.fit(df1["ip_id"])
df1["ip_id"] = ip_enc.transform(df1.ip_id)

# 4. Date
df1["creationdate"] = pd.to_datetime(df1["creationdate"])
df1["date"] = df1["creationdate"].dt.date


# ## Data Preprocessing

# In[11]:


# cleaning country codes
df1.loc[
    df1["cardverificationcodesupplied"].isna(), "cardverificationcodesupplied"
] = False
df1.loc[df1["issuercountrycode"].isna(), "issuercountrycode"] = "ZZ"
df1.loc[df1["shoppercountrycode"].isna(), "shoppercountrycode"] = "ZZ"

unique_issuer_cc = df1["issuercountrycode"].unique()
unique_shopper_cc = df1["shoppercountrycode"].unique()
both = np.append(unique_issuer_cc, unique_shopper_cc)
df_countrycodes = pd.DataFrame(both)
unique_codes = df_countrycodes[0].unique()
enc = LabelEncoder()
enc.fit(unique_codes)
df1["issuercountrycode"] = enc.transform(df1.issuercountrycode)
df1["shoppercountrycode"] = enc.transform(df1.shoppercountrycode)


# In[12]:


def conv(row):
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


# In[13]:


df1["amount_eur"] = df1.apply(lambda x: conv(x), axis=1)


# In[14]:


enc1 = LabelEncoder()
enc1.fit(df1["txvariantcode"])
df1["txvariantcode"] = enc1.transform(df1.txvariantcode)


# In[15]:


enc2 = LabelEncoder()
enc2.fit(df1["currencycode"])
df1["currencycode"] = enc2.transform(df1.currencycode)


# In[16]:


enc3 = LabelEncoder()
enc3.fit(df1["shopperinteraction"])
df1["shopperinteraction"] = enc3.transform(df1.shopperinteraction)


# In[17]:


df1["accountcode"] = df1["accountcode"].apply(lambda x: re.sub("Account", "", x))
df1["accountcode_cc"] = 0
df1.loc[(df1["accountcode"] == "UK"), "accountcode_cc"] = "GB"
df1.loc[(df1["accountcode"] == "Mexico"), "accountcode_cc"] = "MX"
df1.loc[(df1["accountcode"] == "Sweden"), "accountcode_cc"] = "SE"
df1.loc[(df1["accountcode"] == "APAC"), "accountcode_cc"] = "APAC"


# In[18]:


enc4 = LabelEncoder()
enc4.fit(df1["accountcode"])
df1["accountcode"] = enc4.transform(df1.accountcode)


# In[19]:


enc5 = LabelEncoder()
enc5.fit(df1["cardverificationcodesupplied"])
df1["cardverificationcodesupplied"] = enc5.transform(df1.cardverificationcodesupplied)


# In[20]:


df1.loc[df1["mail_id"].str.contains("na", case=False), "mail_id"] = "email99999"

enc6 = LabelEncoder()
enc6.fit(df1["mail_id"])
df1["mail_id"] = enc6.transform(df1.mail_id)


# In[21]:


df1["bookingdate"] = pd.to_datetime(df1["bookingdate"])


# In[22]:


df1.loc[df1["cvcresponsecode"] > 2, "cvcresponsecode"] = 3


# ### Feature Engineering

# In[23]:


df1["countries_equal"] = df1["shoppercountrycode"] == df1["issuercountrycode"]
df1.loc[df1["countries_equal"] == False, "countries_equal"] = 0
df1.loc[df1["countries_equal"] == True, "countries_equal"] = 1


# In[24]:


df1["day_of_week"] = df1["creationdate"].dt.dayofweek
df1["hour"] = df1["creationdate"].dt.hour


# In[25]:


df1 = df1.sort_values(by=["card_id", "creationdate"], ignore_index=True)


# In[95]:


def agg_features(row):
    now = row["creationdate"]
    same_card_id = df1["card_id"] == row["card_id"]
    same_country = df1["shoppercountrycode"] == row["shoppercountrycode"]
    same_currency = df1["currencycode"] = row["currencycode"]
    prev_txns = (df1["creationdate"].lt(now)) & (same_card_id)
    total_prev_txns = prev_txns.sum()

    if total_prev_txns > 1:
        #         print(row)
        month_ago = now - pd.offsets.DateOffset(months=1)
        week_ago = now - pd.offsets.DateOffset(weeks=1)
        day_ago = now - pd.offsets.DateOffset(days=1)

        prev_month_txns = (df1["creationdate"].between(month_ago, now)) & (same_card_id)
        prev_week_txns = (df1["creationdate"].between(week_ago, now)) & (same_card_id)
        prev_day_txns = (df1["creationdate"].between(day_ago, now)) & (same_card_id)

        prev_month_num_txns = df1[prev_month_txns].shape[0]
        prev_month_num_same_country = df1[(prev_month_txns) & (same_country)].shape[0]
        prev_month_num_same_currency = df1[(prev_month_txns) & (same_currency)].shape[0]

        row["prev_month_avg_amount"] = df1.loc[prev_month_txns, "amount_eur"].mean()
        row["prev_week_avg_amount"] = df1.loc[prev_week_txns, "amount_eur"].mean()
        row["prev_day_amount"] = df1.loc[prev_day_txns, "amount_eur"].sum()
        row["daily_avg_over_month"] = (
            df1.loc[prev_txns, ["date", "amount_eur"]]
            .groupby("date")
            .sum()
            .mean()
            .to_numpy()[0]
        )
        row["prev_day_perc_same_country"] = (
            df1[(prev_day_txns) & (same_country)].shape[0] / df1[prev_day_txns].shape[0]
        )
        row["prev_month_perc_same_country"] = (
            prev_month_num_same_country / prev_month_num_txns
        )
        row["prev_month_perc_same_currency"] = (
            prev_month_num_same_currency / prev_month_num_txns
        )

        if prev_month_num_same_country == 0.0:
            row["prev_month_avg_amount_same_country"] = 0.0
        else:
            row["prev_month_avg_amount_same_country"] = df1.loc[
                (prev_month_txns) & (same_country), "amount_eur"
            ].mean()

        if prev_month_num_same_currency == 0.0:
            row["prev_month_avg_amount_same_currency"] = 0.0
        else:
            row["prev_month_avg_amount_same_currency"] = df1.loc[
                (prev_month_txns) & (same_currency), "amount_eur"
            ].mean()
    else:
        row["prev_month_avg_amount"] = row["amount_eur"]
        row["prev_week_avg_amount"] = row["amount_eur"]
        row["prev_day_amount"] = row["amount_eur"]
        row["daily_avg_over_month"] = row["amount_eur"]
        row["prev_day_perc_same_country"] = 1.0
        row["prev_month_perc_same_country"] = 1.0
        row["prev_month_avg_amount_same_country"] = row["amount_eur"]
        row["prev_month_perc_same_currency"] = 1.0
        row["prev_month_avg_amount_same_currency"] = row["amount_eur"]
    return row


# In[103]:


df1_agg = df1.apply(agg_features, axis=1)

df1_agg.to_csv("./data/agg_feat.csv", index=False)
