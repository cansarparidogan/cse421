import numpy as np
import pandas as pd

def create_features(df, time_steps, step):
    segments = []
    labels = []
    x = df["x"].values
    y = df["y"].values
    z = df["z"].values
    activities = df["activity"].values

    for i in range(0, len(df) - time_steps, step):
        xs = x[i: i + time_steps]
        ys = y[i: i + time_steps]
        zs = z[i: i + time_steps]

        feats = [
            np.mean(xs),
            np.mean(ys),
            np.mean(zs),
            np.sum(xs > 0),
            np.sum(ys > 0),
            np.sum(zs > 0),
            np.std(np.abs(xs)),
            np.std(np.abs(ys)),
            np.std(np.abs(zs)),
            np.mean(np.abs(xs) + np.abs(ys) + np.abs(zs)),
        ]

        segments.append(feats)
        labels.append(activities[i])

    return pd.DataFrame(segments), np.array(labels)
