import pandas as pd


def aggregate_features(df):
    """
    We use transaction aggregation strategy as a way to
    capture consumer spending behavior in the recent past.
    
    We put each credit card transaction into the
    historical context of past shopping behavior.

    In essence these attributes provide information on
    card holders' buying behavior in the immediate past.

    - `prev_month_avg_amount`: Average amount spent in the previous month
    - `prev_week_avg_amount`: Average amount spent in the previous week
    - `prev_day_amount`: Average amount spent in the previous day
    - `daily_avg_over_month`: Average amount spent per day in the last month of transactions
    - `prev_day_same_country`: Number of transactions in the last 1 day in the same country as in this transaction
    - `prev_month_same_country`: Number of transactions in the last 1 month in the same country as in this transaction
    - `prev_month_avg_amount_same_country`: Average amount of transactions in the last month in the same country as in this transactions
    - `prev_month_same_currency`: Number of transactions in the last 1 month with the same currency as in this transaction
    - `prev_month_avg_amount_same_currency`: Average amount of transactions in the last month with the same currency as in this transactions
    - `prev_total_transactions`: Total previous transactions with this credit card
    """

    # get aggregat features for a single sample
    def agg_fn(row):
        # current date
        now = row["creationdate"]
        transaction_date = pd.to_datetime(df["creationdate"])

        # rows that have same card_id as current row
        same_card_id = df["card_id"] == row["card_id"]

        # rows that have same shopper country as current row
        same_country = df["shoppercountrycode"] == row["shoppercountrycode"]

        # rows that have same currency as current row
        same_currency = df["currencycode"] = row["currencycode"]

        # all previous transactions with same card id
        prev_txns = (transaction_date.lt(now)) & (same_card_id)

        # total transaction amount of previous transactions
        total_prev_txns = prev_txns.sum()

        # if more than one transaction exists
        if total_prev_txns > 1:
            # dates for month, week and day aggregation windows
            month_ago = now - pd.offsets.DateOffset(months=1)
            week_ago = now - pd.offsets.DateOffset(weeks=1)
            day_ago = now - pd.offsets.DateOffset(days=1)

            # transactions in the previous month with same card id
            prev_month_txns = (transaction_date.between(month_ago, now)) & (
                same_card_id
            )

            # transactions in the previous week with same card id
            prev_week_txns = (transaction_date.between(week_ago, now)) & (same_card_id)

            # transactions in the previous day with same card id
            prev_day_txns = (transaction_date.between(day_ago, now)) & (same_card_id)

            # transactions in previous month in same country
            prev_month_num_same_country = df[(prev_month_txns) & (same_country)].shape[
                0
            ]

            # transactions in same currency in the previous month
            prev_month_num_same_currency = df[
                (prev_month_txns) & (same_currency)
            ].shape[0]

            row["prev_month_avg_amount"] = df.loc[prev_month_txns, "amount_eur"].mean()
            row["prev_week_avg_amount"] = df.loc[prev_week_txns, "amount_eur"].mean()
            row["prev_day_amount"] = df.loc[prev_day_txns, "amount_eur"].mean()
            row["daily_avg_over_month"] = (
                df.loc[prev_txns, ["date", "amount_eur"]]
                .groupby("date")
                .sum()
                .mean()
                .to_numpy()[0]
            )
            row["prev_day_same_country"] = df[(prev_day_txns) & (same_country)].shape[0]
            row["prev_month_same_country"] = prev_month_num_same_country
            row["prev_month_same_currency"] = prev_month_num_same_currency

            if prev_month_num_same_country == 0.0:
                row["prev_month_avg_amount_same_country"] = 0.0
            else:
                row["prev_month_avg_amount_same_country"] = df.loc[
                    (prev_month_txns) & (same_country), "amount_eur"
                ].mean()

            if prev_month_num_same_currency == 0.0:
                row["prev_month_avg_amount_same_currency"] = 0.0
            else:
                row["prev_month_avg_amount_same_currency"] = df.loc[
                    (prev_month_txns) & (same_currency), "amount_eur"
                ].mean()
            row["prev_total_transactions"] = df.loc[same_card_id].shape[0]
        # default values for single transactions
        else:
            row["prev_month_avg_amount"] = row["amount_eur"]
            row["prev_week_avg_amount"] = row["amount_eur"]
            row["prev_day_amount"] = row["amount_eur"]
            row["daily_avg_over_month"] = row["amount_eur"]
            row["prev_day_same_country"] = 1.0
            row["prev_month_same_country"] = 1.0
            row["prev_month_avg_amount_same_country"] = row["amount_eur"]
            row["prev_month_same_currency"] = 1.0
            row["prev_month_avg_amount_same_currency"] = row["amount_eur"]
            row["prev_total_transactions"] = 1.0
        return row

    # apply the agg_fn() function on each row of the data
    df = df.apply(agg_fn, axis=1)

    # return new df with aggregate features
    return df
