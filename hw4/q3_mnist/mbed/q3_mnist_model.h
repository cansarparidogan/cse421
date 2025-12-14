#ifndef Q3_MNIST_MODEL_H
#define Q3_MNIST_MODEL_H
#define Q3_NUM_FEATURES 7
float q3_sigmoid(float x);
float q3_predict_prob_from_hu(const float hu[Q3_NUM_FEATURES]);
int q3_predict_label_from_hu(const float hu[Q3_NUM_FEATURES]);
#endif
