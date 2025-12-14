import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

BASE = os.path.join("q1_temp")
DATA_DIR = os.path.join(BASE, "data")
RES_DIR = os.path.join(BASE, "results")
os.makedirs(RES_DIR, exist_ok=True)

FILE_TRAIN = os.path.join(DATA_DIR, "NEW-DATA-1.T15.txt")
FILE_TEST  = os.path.join(DATA_DIR, "NEW-DATA-2.T15.txt")

prev_values_count = 5
downsample = 4

def read_sml2010(path):
    df = None
    for kwargs in [
        dict(sep=r"\s+", engine="python"),
        dict(sep=";", engine="python"),
        dict(sep=",", engine="python"),
        dict(delim_whitespace=True, engine="python"),
    ]:
        try:
            df = pd.read_csv(path, **kwargs)
            if df is not None and df.shape[1] >= 3:
                break
        except Exception:
            df = None
    if df is None or df.shape[1] < 3:
        raise RuntimeError(f"Cannot parse: {path}")

    if "T1" in df.columns:
        df = df.rename(columns={"T1": "Room_Temp"})
    if "Room_Temp" not in df.columns:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(num_cols) == 0:
            raise RuntimeError("No numeric columns found.")
        df["Room_Temp"] = df[num_cols[0]].astype(float)

    y = df["Room_Temp"].astype(float).to_numpy()
    y = y[::downsample]
    return y

def make_supervised(y, k):
    X = np.zeros((len(y) - k, k), dtype=np.float32)
    t = y[k:].astype(np.float32)
    for i in range(k):
        X[:, i] = y[(k - i - 1):(len(y) - i - 1)]
    return X, t

def zscore_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    return mu, sigma

def zscore_apply(X, mu, sigma):
    return (X - mu) / sigma

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def main():
    y_train_raw = read_sml2010(FILE_TRAIN)
    y_test_raw = read_sml2010(FILE_TEST)

    X_train_raw, t_train = make_supervised(y_train_raw, prev_values_count)
    X_test_raw, t_test = make_supervised(y_test_raw, prev_values_count)

    mu, sigma = zscore_fit(X_train_raw)
    X_train = zscore_apply(X_train_raw, mu, sigma)
    X_test = zscore_apply(X_test_raw, mu, sigma)

    model = LinearRegression()
    model.fit(X_train, t_train)

    pred_train = model.predict(X_train).astype(np.float32)
    pred_test = model.predict(X_test).astype(np.float32)

    mae_train = float(mean_absolute_error(t_train, pred_train))
    mae_test = float(mean_absolute_error(t_test, pred_test))
    rmse_train = rmse(t_train, pred_train)
    rmse_test = rmse(t_test, pred_test)

    print("TRAIN: MAE", mae_train, "RMSE", rmse_train)
    print("TEST : MAE", mae_test, "RMSE", rmse_test)

    coef = model.coef_.astype(np.float32)
    intercept = np.float32(model.intercept_)

    np.savez(
        os.path.join(RES_DIR, "q1_linreg_params.npz"),
        prev_values_count=np.int32(prev_values_count),
        downsample=np.int32(downsample),
        z_mu=mu.astype(np.float32),
        z_sigma=sigma.astype(np.float32),
        coef=coef,
        intercept=intercept
    )

    head_n = 10
    rows = []
    for i in range(min(head_n, len(t_test))):
        rows.append([float(t_test[i]), float(pred_test[i])])
    pd.DataFrame(rows, columns=["actual", "pred"]).to_csv(
        os.path.join(RES_DIR, "q1_test_preview.csv"),
        index=False
    )

if __name__ == "__main__":
    main()
