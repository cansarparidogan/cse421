#include "q2_kws_model.h"
#include <math.h>
static const float W[Q2_NUM_FEATURES]={-0.0342195779f,-0.0698422194f,0.0392694585f,-0.226574779f,-0.279733509f,-0.103771016f,0.00813459139f,0.342335135f,-0.225609526f,-0.516010761f,-0.0186215211f,-0.461740524f,-0.507474482f,0.0180417709f,-0.216135129f,-0.0709328204f,0.275730312f,0.358736157f,-0.206004709f,0.247370586f,0.0633517578f,-0.115216196f,0.395280361f,-0.386475921f,0.0755620524f,-0.527566731f};
static const float B=0.294998318f;
float q2_sigmoid(float x){return 1.0f/(1.0f+expf(-x));}
float q2_predict_prob(const float x[Q2_NUM_FEATURES]){float s=B;for(int i=0;i<Q2_NUM_FEATURES;i++)s+=W[i]*x[i];return q2_sigmoid(s);}
int q2_predict_label(const float x[Q2_NUM_FEATURES]){return q2_predict_prob(x)>0.5f?1:0;}
