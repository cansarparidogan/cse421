#include "q2_kws_model.h"
#include <math.h>

static const float W[Q2_NUM_FEATURES] = {0};
static const float B = 0.0f;

float q2_sigmoid(float x){ return 1.0f/(1.0f+expf(-x)); }

float q2_predict_prob(const float x[Q2_NUM_FEATURES]){
    float s=B;
    for(int i=0;i<Q2_NUM_FEATURES;i++) s+=W[i]*x[i];
    return q2_sigmoid(s);
}

int q2_predict_label(const float x[Q2_NUM_FEATURES]){
    return q2_predict_prob(x)>0.5f ? 1 : 0;
}
