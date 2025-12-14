import os
import numpy as np
import tensorflow as tf
from sklearn import metrics
from data_utils import read_data
from feature_utils import create_features
DATA_PATH=os.path.join("q2_har","data","WISDM_ar_v1.1","WISDM_ar_v1.1_raw.txt")
TIME_PERIODS=80
STEP_DISTANCE=40
os.makedirs(os.path.join("q2_har","results"),exist_ok=True)
os.makedirs(os.path.join("q2_har","mbed"),exist_ok=True)
df=read_data(DATA_PATH)
df_train=df[df["user"]<=28]
df_test=df[df["user"]>28]
Xtr,ytr=create_features(df_train,TIME_PERIODS,STEP_DISTANCE)
Xte,yte=create_features(df_test,TIME_PERIODS,STEP_DISTANCE)
ytr=np.where(ytr=="Walking",0,1).astype(int)
yte=np.where(yte=="Walking",0,1).astype(int)
Xtr_np=Xtr.to_numpy().astype(np.float32)
Xte_np=Xte.to_numpy().astype(np.float32)
model=tf.keras.models.Sequential([tf.keras.layers.Dense(1,input_shape=[10],activation="sigmoid")])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss=tf.keras.losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives()])
model.fit(Xtr_np,ytr,epochs=50,verbose=2)
p=model.predict(Xte_np,verbose=0).reshape(-1)
pred=(p>0.5).astype(int)
cm=metrics.confusion_matrix(yte,pred)
acc=float((pred==yte).mean())
fn=int(np.sum((yte==0)&(pred==1)))
print("ACC",acc)
print("FN",fn)
print("CONFUSION_MATRIX\n",cm)
w,b=model.layers[0].get_weights()
w=w.reshape(-1).astype(np.float32)
b=float(b.reshape(-1)[0])
np.savez(os.path.join("q2_har","results","q2_single_neuron_params.npz"),w=w,b=np.float32(b),TIME_PERIODS=np.int32(TIME_PERIODS),STEP_DISTANCE=np.int32(STEP_DISTANCE))
with open(os.path.join("q2_har","mbed","q2_har_model.h"),"w") as f:
    f.write("#ifndef Q2_HAR_MODEL_H\n#define Q2_HAR_MODEL_H\n#define Q2_NUM_FEATURES 10\nfloat q2_sigmoid(float x);\nfloat q2_predict_prob(const float x[Q2_NUM_FEATURES]);\nint q2_predict_label(const float x[Q2_NUM_FEATURES]);\n#endif\n")
with open(os.path.join("q2_har","mbed","q2_har_model.c"),"w") as f:
    f.write('#include "q2_har_model.h"\n#include <math.h>\nstatic const float W[Q2_NUM_FEATURES]={')
    f.write(",".join([f"{float(v):.9g}f" for v in w]))
    f.write("};\n")
    f.write(f"static const float B={b:.9g}f;\n")
    f.write("float q2_sigmoid(float x){return 1.0f/(1.0f+expf(-x));}\n")
    f.write("float q2_predict_prob(const float x[Q2_NUM_FEATURES]){float s=B;for(int i=0;i<Q2_NUM_FEATURES;i++)s+=W[i]*x[i];return q2_sigmoid(s);}\n")
    f.write("int q2_predict_label(const float x[Q2_NUM_FEATURES]){return q2_predict_prob(x)>0.5f?1:0;}\n")
