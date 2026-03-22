def validate_df(df):
    # remove duplicates
    df = df.drop_duplicates()

    # fill missing
    df = df.fillna("Not Available")

    return df