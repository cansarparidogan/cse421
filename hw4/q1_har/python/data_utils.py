import pandas as pd

def read_data(file_path):
    cols=["user","activity","timestamp","x-axis","y-axis","z-axis"]
    df=pd.read_csv(
        file_path,
        header=None,
        names=cols,
        sep=",",
        engine="python",
        on_bad_lines="skip"
    )
    df["z-axis"]=df["z-axis"].astype(str).str.replace(";","",regex=False)
    df=df[df["z-axis"].str.strip()!=""]
    df["user"]=pd.to_numeric(df["user"],errors="coerce")
    df["timestamp"]=pd.to_numeric(df["timestamp"],errors="coerce")
    df["x-axis"]=pd.to_numeric(df["x-axis"],errors="coerce")
    df["y-axis"]=pd.to_numeric(df["y-axis"],errors="coerce")
    df["z-axis"]=pd.to_numeric(df["z-axis"],errors="coerce")
    df.dropna(inplace=True)
    df["user"]=df["user"].astype(int)
    return df
