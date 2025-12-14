#include "q2_har_model.h"
#include <math.h>
static const float W[Q2_NUM_FEATURES]={-0.0229644421f,-0.910675228f,-0.331160545f,-0.0141179534f,-0.0560694747f,0.0474907868f,-0.00130453077f,0.0678504556f,0.047571063f,0.000446661434f};
static const float B=1.27937174f;
float q2_sigmoid(float x){return 1.0f/(1.0f+expf(-x));}
float q2_predict_prob(const float x[Q2_NUM_FEATURES]){float s=B;for(int i=0;i<Q2_NUM_FEATURES;i++)s+=W[i]*x[i];return q2_sigmoid(s);}
int q2_predict_label(const float x[Q2_NUM_FEATURES]){return q2_predict_prob(x)>0.5f?1:0;}
