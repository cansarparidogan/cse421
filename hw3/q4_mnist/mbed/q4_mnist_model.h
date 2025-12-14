#ifndef Q4_MNIST_MODEL_H
#define Q4_MNIST_MODEL_H
#define Q4_NUM_FEATURES 7
float q4_sigmoid(float x);
float q4_predict_prob_from_hu(const float hu[Q4_NUM_FEATURES]);
int q4_predict_label_from_hu(const float hu[Q4_NUM_FEATURES]);
#endif
