import numpy as np
import pandas as pd
def _fft_std(x):
    f=np.abs(np.fft.rfft(x))
    return float(np.std(f))
def create_features(df,time_steps,step_size):
    x_segments=[];y_segments=[];z_segments=[];labels=[]
    for i in range(0,len(df)-time_steps,step_size):
        xs=df["x-axis"].values[i:i+time_steps]
        ys=df["y-axis"].values[i:i+time_steps]
        zs=df["z-axis"].values[i:i+time_steps]
        counts=df["activity"].iloc[i:i+time_steps].value_counts()
        if len(counts)==0: 
            continue
        if int(counts.iloc[0])==time_steps:
            labels.append(counts.index[0])
            x_segments.append(xs);y_segments.append(ys);z_segments.append(zs)
    seg=pd.DataFrame({"x":x_segments,"y":y_segments,"z":z_segments})
    feat=pd.DataFrame()
    feat["x_mean"]=seg["x"].apply(lambda a: float(np.mean(a)))
    feat["y_mean"]=seg["y"].apply(lambda a: float(np.mean(a)))
    feat["z_mean"]=seg["z"].apply(lambda a: float(np.mean(a)))
    feat["x_pos_count"]=seg["x"].apply(lambda a: int(np.sum(np.array(a)>0)))
    feat["y_pos_count"]=seg["y"].apply(lambda a: int(np.sum(np.array(a)>0)))
    feat["z_pos_count"]=seg["z"].apply(lambda a: int(np.sum(np.array(a)>0)))
    feat["x_fft_std"]=seg["x"].apply(_fft_std)
    feat["y_fft_std"]=seg["y"].apply(_fft_std)
    feat["z_fft_std"]=seg["z"].apply(_fft_std)
    feat["sma_fft"]=seg.apply(lambda r: float(np.sum(np.abs(np.fft.rfft(r["x"])))+np.sum(np.abs(np.fft.rfft(r["y"])))+np.sum(np.abs(np.fft.rfft(r["z"])))),axis=1)
    return feat,np.array(labels)
