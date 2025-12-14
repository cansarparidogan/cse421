import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from data_utils import read_data
from feature_utils import create_features

DATA_PATH=os.path.join("q1_har","data","WISDM_ar_v1.1","WISDM_ar_v1.1_raw.txt")
TIME_PERIODS=80
STEP_DISTANCE=40

os.makedirs(os.path.join("q1_har","results"),exist_ok=True)
os.makedirs(os.path.join("q1_har","mbed"),exist_ok=True)

df=read_data(DATA_PATH)
df_train=df[df["user"]<=28]
df_test=df[df["user"]>28]

Xtr,ytr=create_features(df_train,TIME_PERIODS,STEP_DISTANCE)
Xte,yte=create_features(df_test,TIME_PERIODS,STEP_DISTANCE)

Xtr=Xtr.to_numpy().astype(np.float32)
Xte=Xte.to_numpy().astype(np.float32)

enc=OneHotEncoder(sparse_output=False)
ytr_oh=enc.fit_transform(ytr.reshape(-1,1)).astype(np.float32)
yte_oh=enc.transform(yte.reshape(-1,1)).astype(np.float32)

class_names=[str(c) for c in enc.categories_[0]]
num_classes=len(class_names)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(100,input_shape=[10],activation="relu"),
    tf.keras.layers.Dense(100,activation="relu"),
    tf.keras.layers.Dense(num_classes,activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

model.fit(Xtr,ytr_oh,epochs=50,batch_size=128,verbose=2)

p=model.predict(Xte,verbose=0)
pred=np.argmax(p,axis=1)
true=np.argmax(yte_oh,axis=1)

cm=confusion_matrix(true,pred)
acc=float((pred==true).mean())

print("CLASSES",class_names)
print("ACC",acc)
print("CONFUSION_MATRIX\n",cm)

W1,b1=model.layers[0].get_weights()
W2,b2=model.layers[1].get_weights()
W3,b3=model.layers[2].get_weights()

np.savez(
    os.path.join("q1_har","results","q1_har_mlp_params.npz"),
    class_names=np.array(class_names),
    W1=W1.astype(np.float32), b1=b1.astype(np.float32),
    W2=W2.astype(np.float32), b2=b2.astype(np.float32),
    W3=W3.astype(np.float32), b3=b3.astype(np.float32)
)

with open(os.path.join("q1_har","mbed","q1_har_mlp_config.h"),"w") as f:
    f.write("#ifndef Q1_HAR_MLP_CONFIG_H\n#define Q1_HAR_MLP_CONFIG_H\n")
    f.write("#define Q1_NUM_FEATURES 10\n")
    f.write("#define Q1_H1 100\n#define Q1_H2 100\n")
    f.write(f"#define Q1_NUM_CLASSES {num_classes}\n")
    f.write("extern const float Q1_W1[Q1_H1][Q1_NUM_FEATURES];\n")
    f.write("extern const float Q1_B1[Q1_H1];\n")
    f.write("extern const float Q1_W2[Q1_H2][Q1_H1];\n")
    f.write("extern const float Q1_B2[Q1_H2];\n")
    f.write("extern const float Q1_W3[Q1_NUM_CLASSES][Q1_H2];\n")
    f.write("extern const float Q1_B3[Q1_NUM_CLASSES];\n")
    f.write("int q1_har_predict(const float x[Q1_NUM_FEATURES]);\n")
    f.write("#endif\n")

with open(os.path.join("q1_har","mbed","q1_har_mlp_config.c"),"w") as f:
    f.write('#include "q1_har_mlp_config.h"\n#include <math.h>\n')
    def mat(name,arr):
        f.write(f"const float {name}[{arr.shape[0]}][{arr.shape[1]}]={{\n")
        for r in arr:
            f.write("{"+",".join(f"{v:.9g}f" for v in r)+"},\n")
        f.write("};\n")
    def vec(name,arr):
        f.write(f"const float {name}[{arr.shape[0]}]={{")
        f.write(",".join(f"{v:.9g}f" for v in arr))
        f.write("};\n")
    mat("Q1_W1",W1); vec("Q1_B1",b1)
    mat("Q1_W2",W2); vec("Q1_B2",b2)
    mat("Q1_W3",W3); vec("Q1_B3",b3)
    f.write("static float relu(float v){return v>0.0f?v:0.0f;}\n")
    f.write("int q1_har_predict(const float x[Q1_NUM_FEATURES]){\n")
    f.write("static float h1[Q1_H1]; static float h2[Q1_H2];\n")
    f.write("for(int i=0;i<Q1_H1;i++){float s=Q1_B1[i];for(int j=0;j<Q1_NUM_FEATURES;j++)s+=Q1_W1[i][j]*x[j];h1[i]=relu(s);} \n")
    f.write("for(int i=0;i<Q1_H2;i++){float s=Q1_B2[i];for(int j=0;j<Q1_H1;j++)s+=Q1_W2[i][j]*h1[j];h2[i]=relu(s);} \n")
    f.write("int arg=0; float best=-1e30f; \n")
    f.write("for(int i=0;i<Q1_NUM_CLASSES;i++){float s=Q1_B3[i];for(int j=0;j<Q1_H2;j++)s+=Q1_W3[i][j]*h2[j]; if(s>best){best=s;arg=i;}} \n")
    f.write("return arg;}\n")
