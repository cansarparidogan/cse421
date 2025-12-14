import numpy as np
import tensorflow as tf
import cv2

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_x = train_x.astype(np.uint8)
test_x = test_x.astype(np.uint8)

def hu(img):
    m = cv2.moments(img, True)
    return cv2.HuMoments(m).reshape(7).astype(np.float32)

Xtr = np.stack([hu(im) for im in train_x], axis=0)
Xte = np.stack([hu(im) for im in test_x], axis=0)

mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0) + 1e-8
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd

ytr = train_y.copy()
yte = test_y.copy()
ytr[ytr != 0] = 1
yte[yte != 0] = 1

w, *_ = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))], ytr.astype(np.float32), rcond=None)
coef = w[:-1].astype(np.float32)
b = float(w[-1])

pred = ((Xte @ coef + b) > 0.5).astype(np.int32)
acc = float((pred == yte).mean())
fn = int(np.sum((yte == 0) & (pred == 1)))

print("BONUS_ACC", acc)
print("BONUS_FN", fn)
