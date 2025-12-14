import os
import numpy as np

DATA_DIR=os.path.join("q4_temp","data")
OUT_NPZ=os.path.join("q4_temp","results","q4_linreg_params.npz")
OUT_H=os.path.join("q4_temp","mbed","q4_temp_model.h")
OUT_C=os.path.join("q4_temp","mbed","q4_temp_model.c")

os.makedirs(os.path.join("q4_temp","results"),exist_ok=True)
os.makedirs(os.path.join("q4_temp","mbed"),exist_ok=True)

def load_series(path):
    a=np.genfromtxt(path, dtype=str, delimiter=None)
    if a.ndim==1:
        vals=[]
        for s in a.tolist():
            try:
                vals.append(float(s))
            except:
                pass
        return np.array(vals,dtype=np.float32)
    last=a[:,-1]
    vals=np.array([float(x) for x in last],dtype=np.float32)
    return vals

def build_xy(series, downsample, prev_count):
    s=series[::downsample].astype(np.float32)
    n=len(s)
    X=[]
    y=[]
    for i in range(prev_count, n):
        X.append(s[i-prev_count:i].copy())
        y.append(s[i])
    return np.stack(X,axis=0), np.array(y,dtype=np.float32)

def zscore_fit(X):
    mu=X.mean(axis=0).astype(np.float32)
    sd=X.std(axis=0).astype(np.float32)+1e-8
    return mu,sd

def zscore_apply(X,mu,sd):
    return (X-mu)/sd

def linreg_fit(X,y):
    ones=np.ones((X.shape[0],1),dtype=np.float32)
    A=np.concatenate([X,ones],axis=1)
    w=np.linalg.lstsq(A,y,rcond=None)[0].astype(np.float32)
    return w[:-1], float(w[-1])

def mae(y,p):
    return float(np.mean(np.abs(y-p)))

def rmse(y,p):
    return float(np.sqrt(np.mean((y-p)**2)))

paths=[os.path.join(DATA_DIR,f) for f in sorted(os.listdir(DATA_DIR)) if f.endswith(".txt")]
if len(paths)<2:
    raise SystemExit("Need two data files in q4_temp/data")

train_series=load_series(paths[0])
test_series=load_series(paths[1])

downsample=6
prev_count=5

Xtr,ytr=build_xy(train_series,downsample,prev_count)
Xte,yte=build_xy(test_series,downsample,prev_count)

mu,sd=zscore_fit(Xtr)
Xtr_z=zscore_apply(Xtr,mu,sd)
Xte_z=zscore_apply(Xte,mu,sd)

coef,b=linreg_fit(Xtr_z,ytr)

ptr=(Xtr_z@coef + b).astype(np.float32)
pte=(Xte_z@coef + b).astype(np.float32)

print("TRAIN_MAE",mae(ytr,ptr))
print("TRAIN_RMSE",rmse(ytr,ptr))
print("TEST_MAE",mae(yte,pte))
print("TEST_RMSE",rmse(yte,pte))
print("COEF",coef)
print("INTERCEPT",b)
print("MU",mu)
print("SIGMA",sd)

np.savez(OUT_NPZ,
         downsample=np.int32(downsample),
         prev_values_count=np.int32(prev_count),
         z_mu=mu.astype(np.float32),
         z_sigma=sd.astype(np.float32),
         coef=coef.astype(np.float32),
         intercept=np.float32(b))

with open(OUT_H,"w") as f:
    f.write("#ifndef Q4_TEMP_MODEL_H\n#define Q4_TEMP_MODEL_H\n")
    f.write("#define Q4_PREV 5\n")
    f.write("float q4_temp_predict(const float prev[Q4_PREV]);\n")
    f.write("#endif\n")

with open(OUT_C,"w") as f:
    f.write('#include "q4_temp_model.h"\n')
    f.write("static const float MU[Q4_PREV]={"+",".join(f"{float(v):.9g}f" for v in mu)+"};\n")
    f.write("static const float SD[Q4_PREV]={"+",".join(f"{float(v):.9g}f" for v in sd)+"};\n")
    f.write("static const float W[Q4_PREV]={"+",".join(f"{float(v):.9g}f" for v in coef)+"};\n")
    f.write(f"static const float B={float(b):.9g}f;\n")
    f.write("float q4_temp_predict(const float prev[Q4_PREV]){\n")
    f.write("  float s=B;\n")
    f.write("  for(int i=0;i<Q4_PREV;i++){\n")
    f.write("    float z=(prev[i]-MU[i])/SD[i];\n")
    f.write("    s+=W[i]*z;\n")
    f.write("  }\n")
    f.write("  return s;\n}\n")
