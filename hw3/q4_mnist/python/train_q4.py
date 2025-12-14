import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix

(base_train_x, base_train_y), (base_test_x, base_test_y) = tf.keras.datasets.mnist.load_data()

train_images = base_train_x.astype(np.uint8)
train_labels = base_train_y.astype(np.uint8)
test_images = base_test_x.astype(np.uint8)
test_labels = base_test_y.astype(np.uint8)

train_hu = np.empty((len(train_images),7),dtype=np.float32)
test_hu = np.empty((len(test_images),7),dtype=np.float32)

for i,img in enumerate(train_images):
    m=cv2.moments(img,True)
    train_hu[i]=cv2.HuMoments(m).reshape(7)

for i,img in enumerate(test_images):
    m=cv2.moments(img,True)
    test_hu[i]=cv2.HuMoments(m).reshape(7)

mu=train_hu.mean(axis=0)
sd=train_hu.std(axis=0)+1e-8
train_hu=(train_hu-mu)/sd
test_hu=(test_hu-mu)/sd

ytr=train_labels.copy()
yte=test_labels.copy()
ytr[ytr!=0]=1
yte[yte!=0]=1

model=tf.keras.models.Sequential([tf.keras.layers.Dense(1,input_shape=[7],activation="sigmoid")])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives()])

model.fit(train_hu,ytr,batch_size=128,epochs=50,class_weight={0:8.0,1:1.0},verbose=2)

p=model.predict(test_hu,verbose=0).reshape(-1)
pred=(p>0.5).astype(np.int32)

cm=confusion_matrix(yte,pred)
acc=float((pred==yte).mean())
fn=int(np.sum((yte==0)&(pred==1)))

print("ACC",acc)
print("FN",fn)
print("CONFUSION_MATRIX\n",cm)

os.makedirs(os.path.join("q4_mnist","results"),exist_ok=True)
os.makedirs(os.path.join("q4_mnist","mbed"),exist_ok=True)

w,b=model.layers[0].get_weights()
w=w.reshape(-1).astype(np.float32)
b=float(b.reshape(-1)[0])

np.savez(os.path.join("q4_mnist","results","q4_mnist_params.npz"),
         w=w,b=np.float32(b),mu=mu.astype(np.float32),sd=sd.astype(np.float32))

with open(os.path.join("q4_mnist","mbed","q4_mnist_model.h"),"w") as f:
    f.write("#ifndef Q4_MNIST_MODEL_H\n#define Q4_MNIST_MODEL_H\n#define Q4_NUM_FEATURES 7\nfloat q4_sigmoid(float x);\nfloat q4_predict_prob_from_hu(const float hu[Q4_NUM_FEATURES]);\nint q4_predict_label_from_hu(const float hu[Q4_NUM_FEATURES]);\n#endif\n")

with open(os.path.join("q4_mnist","mbed","q4_mnist_model.c"),"w") as f:
    f.write('#include "q4_mnist_model.h"\n#include <math.h>\n')
    f.write('static const float W[Q4_NUM_FEATURES]={'+",".join([f"{float(v):.9g}f" for v in w])+"};\n")
    f.write('static const float MU[Q4_NUM_FEATURES]={'+",".join([f"{float(v):.9g}f" for v in mu])+"};\n")
    f.write('static const float SD[Q4_NUM_FEATURES]={'+",".join([f"{float(v):.9g}f" for v in sd])+"};\n")
    f.write(f"static const float B={b:.9g}f;\n")
    f.write("float q4_sigmoid(float x){return 1.0f/(1.0f+expf(-x));}\n")
    f.write("float q4_predict_prob_from_hu(const float hu[Q4_NUM_FEATURES]){float s=B;for(int i=0;i<Q4_NUM_FEATURES;i++){float z=(hu[i]-MU[i])/SD[i];s+=W[i]*z;}return q4_sigmoid(s);}\n")
    f.write("int q4_predict_label_from_hu(const float hu[Q4_NUM_FEATURES]){return q4_predict_prob_from_hu(hu)>0.5f?1:0;}\n")
