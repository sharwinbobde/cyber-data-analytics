import pandas as pd


def aggregate_features(df):
    def agg_fn(row):
        # # input(row)
        # row["prev_1day_avg_amount"] = row.rolling("1d", on="creationdate")[
        #     "amount_eur"
        # ].mean()
        # row["prev_7day_avg_amount"] = row.rolling("7d", on="creationdate")[
        #     "amount_eur"
        # ].mean()
        # row["prev_30day_avg_amount"] = row.rolling("30d", on="creationdate")[
        #     "amount_eur"
        # ].mean()
        now = row["creationdate"]
        transaction_date = pd.to_datetime(df["creationdate"])
        same_card_id = df["card_id"] == row["card_id"]
        same_country = df["shoppercountrycode"] == row["shoppercountrycode"]
        same_currency = df["currencycode"] = row["currencycode"]
        prev_txns = (transaction_date.lt(now)) & (same_card_id)
        total_prev_txns = prev_txns.sum()

        if total_prev_txns > 1:
            month_ago = now - pd.offsets.DateOffset(months=1)
            week_ago = now - pd.offsets.DateOffset(weeks=1)
            day_ago = now - pd.offsets.DateOffset(days=1)

            prev_month_txns = (transaction_date.between(month_ago, now)) & (
                same_card_id
            )
            prev_week_txns = (transaction_date.between(week_ago, now)) & (same_card_id)
            prev_day_txns = (transaction_date.between(day_ago, now)) & (same_card_id)

            # prev_month_num_txns = df[prev_month_txns].shape[0]
            prev_month_num_same_country = df[(prev_month_txns) & (same_country)].shape[
                0
            ]
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
        else:
            row["prev_month_avg_amount"] = 0.0
            row["prev_week_avg_amount"] = 0.0
            row["prev_day_amount"] = 0.0
            row["daily_avg_over_month"] = 0.0
            row["prev_day_same_country"] = 0.0
            row["prev_month_same_country"] = 0.0
            row["prev_month_avg_amount_same_country"] = 0.0
            row["prev_month_same_currency"] = 0.0
            row["prev_month_avg_amount_same_currency"] = 0.0
            row["prev_total_transactions"] = 0.0
        return row

    df = df.apply(agg_fn, axis=1)
    return df
