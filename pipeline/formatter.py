import pandas as pd

def to_dataframe(data, filename):
    df = pd.DataFrame(data)
    df["bid_id"] = filename
    return df