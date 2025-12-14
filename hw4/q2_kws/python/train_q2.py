import os
import numpy as np
import scipy.signal as sig
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from mfcc_func import create_mfcc_features

RECORDINGS_DIR = os.path.join("q2_kws","recordings")
recordings = [p for p in os.listdir(RECORDINGS_DIR) if p.lower().endswith(".wav")]
recordings_list = [(RECORDINGS_DIR, p) for p in recordings]

FFTSize = 1024
sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13

test_list = {r for r in recordings_list if "yweweler" in r[1]}
train_list = set(recordings_list) - test_list

Xtr, ytr = create_mfcc_features(list(train_list), FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs)
Xte, yte = create_mfcc_features(list(test_list), FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs)

ytr = ytr.copy()
yte = yte.copy()
ytr[ytr != 0] = 1
yte[yte != 0] = 1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[numOfDctOutputs * 2], activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()]
)

model.fit(Xtr, ytr, epochs=50, verbose=2, class_weight={0:10.0, 1:1.0})

p = model.predict(Xte, verbose=0).reshape(-1)
pred = (p > 0.5).astype(np.int32)

cm = confusion_matrix(yte, pred)
acc = float((pred == yte).mean())
fn = int(np.sum((yte == 0) & (pred == 1)))

print("ACC", acc)
print("FN", fn)
print("CONFUSION_MATRIX\n", cm)

os.makedirs(os.path.join("q2_kws","results"), exist_ok=True)
os.makedirs(os.path.join("q2_kws","mbed"), exist_ok=True)

w, b = model.layers[0].get_weights()
w = w.reshape(-1).astype(np.float32)
b = float(b.reshape(-1)[0])

np.savez(os.path.join("q2_kws","results","q2_kws_params.npz"),
         w=w, b=np.float32(b),
         FFTSize=np.int32(FFTSize),
         sample_rate=np.int32(sample_rate),
         numOfMelFilters=np.int32(numOfMelFilters),
         numOfDctOutputs=np.int32(numOfDctOutputs))

with open(os.path.join("q2_kws","mbed","q2_kws_model.h"),"w") as f:
    f.write("#ifndef Q2_KWS_MODEL_H\n#define Q2_KWS_MODEL_H\n#define Q2_NUM_FEATURES 26\nfloat q2_sigmoid(float x);\nfloat q2_predict_prob(const float x[Q2_NUM_FEATURES]);\nint q2_predict_label(const float x[Q2_NUM_FEATURES]);\n#endif\n")

with open(os.path.join("q2_kws","mbed","q2_kws_model.c"),"w") as f:
    f.write('#include "q2_kws_model.h"\n#include <math.h>\nstatic const float W[Q2_NUM_FEATURES]={')
    f.write(",".join([f"{float(v):.9g}f" for v in w]))
    f.write("};\n")
    f.write(f"static const float B={b:.9g}f;\n")
    f.write("float q2_sigmoid(float x){return 1.0f/(1.0f+expf(-x));}\n")
    f.write("float q2_predict_prob(const float x[Q2_NUM_FEATURES]){float s=B;for(int i=0;i<Q2_NUM_FEATURES;i++)s+=W[i]*x[i];return q2_sigmoid(s);}\n")
    f.write("int q2_predict_label(const float x[Q2_NUM_FEATURES]){return q2_predict_prob(x)>0.5f?1:0;}\n")
