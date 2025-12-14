#ifndef Q2_KWS_MODEL_H
#define Q2_KWS_MODEL_H
#define Q2_NUM_FEATURES 26
float q2_sigmoid(float x);
float q2_predict_prob(const float x[Q2_NUM_FEATURES]);
int q2_predict_label(const float x[Q2_NUM_FEATURES]);
#endif
