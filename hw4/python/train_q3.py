import os
import numpy as np
import scipy.signal as sig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import tensorflow as tf
from mfcc_func import create_mfcc_features

RECORDINGS_DIR = os.path.join("q3_kws","recordings")
recordings_list = [(RECORDINGS_DIR, p) for p in os.listdir(RECORDINGS_DIR) if p.lower().endswith(".wav")]

FFTSize = 1024
sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13
window = sig.get_window("hamming", FFTSize)

test_list = {r for r in recordings_list if "yweweler" in r[1]}
train_list = set(recordings_list) - test_list

train_mfcc_features, train_labels = create_mfcc_features(list(train_list), FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window)
test_mfcc_features, test_labels = create_mfcc_features(list(test_list), FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[numOfDctOutputs * 2], activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()]
)

train_labels = train_labels.copy()
test_labels = test_labels.copy()
train_labels[train_labels != 0] = 1
test_labels[test_labels != 0] = 1

model.fit(train_mfcc_features, train_labels, epochs=50, verbose=2, class_weight={0:10.0, 1:1.0})

p = model.predict(test_mfcc_features, verbose=0).reshape(-1)
pred = (p > 0.5).astype(np.int32)

cm = confusion_matrix(test_labels, pred)
acc = float((pred == test_labels).mean())
fn = int(np.sum((test_labels == 0) & (pred == 1)))

print("ACC", acc)
print("FN", fn)
print("CONFUSION_MATRIX\n", cm)

os.makedirs(os.path.join("q3_kws","results"), exist_ok=True)
os.makedirs(os.path.join("q3_kws","mbed"), exist_ok=True)

w, b = model.layers[0].get_weights()
w = w.reshape(-1).astype(np.float32)
b = float(b.reshape(-1)[0])

np.savez(os.path.join("q3_kws","results","q3_kws_params.npz"),
         w=w, b=np.float32(b),
         FFTSize=np.int32(FFTSize),
         sample_rate=np.int32(sample_rate),
         numOfMelFilters=np.int32(numOfMelFilters),
         numOfDctOutputs=np.int32(numOfDctOutputs))

with open(os.path.join("q3_kws","mbed","q3_kws_model.h"),"w") as f:
    f.write("#ifndef Q3_KWS_MODEL_H\n#define Q3_KWS_MODEL_H\n#define Q3_NUM_FEATURES 26\nfloat q3_sigmoid(float x);\nfloat q3_predict_prob(const float x[Q3_NUM_FEATURES]);\nint q3_predict_label(const float x[Q3_NUM_FEATURES]);\n#endif\n")

with open(os.path.join("q3_kws","mbed","q3_kws_model.c"),"w") as f:
    f.write('#include "q3_kws_model.h"\n#include <math.h>\nstatic const float W[Q3_NUM_FEATURES]={')
    f.write(",".join([f"{float(v):.9g}f" for v in w]))
    f.write("};\n")
    f.write(f"static const float B={b:.9g}f;\n")
    f.write("float q3_sigmoid(float x){return 1.0f/(1.0f+expf(-x));}\n")
    f.write("float q3_predict_prob(const float x[Q3_NUM_FEATURES]){float s=B;for(int i=0;i<Q3_NUM_FEATURES;i++)s+=W[i]*x[i];return q3_sigmoid(s);}\n")
    f.write("int q3_predict_label(const float x[Q3_NUM_FEATURES]){return q3_predict_prob(x)>0.5f?1:0;}\n")

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["zero","not_zero"])
disp.plot()
plt.title("Single Neuron KWS Confusion Matrix")
plt.savefig(os.path.join("q3_kws","results","q3_confusion_matrix.png"), dpi=200, bbox_inches="tight")
