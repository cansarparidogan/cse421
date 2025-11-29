import pandas as pd

def read_data(path):
    col_names = ["user", "activity", "timestamp", "x", "y", "z"]
    df = pd.read_csv(path, header=None, names=col_names)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["z"] = df["z"].astype(float)
    return df
