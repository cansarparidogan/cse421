#ifndef Q3_KWS_MODEL_H
#define Q3_KWS_MODEL_H
#define Q3_NUM_FEATURES 26
float q3_sigmoid(float x);
float q3_predict_prob(const float x[Q3_NUM_FEATURES]);
int q3_predict_label(const float x[Q3_NUM_FEATURES]);
#endif
