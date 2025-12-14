#include "q3_kws_model.h"
#include <math.h>
static const float W[Q3_NUM_FEATURES]={-0.0266536549f,-0.0815469474f,0.0833076462f,-0.257861823f,-0.268905729f,-0.0771287754f,-0.0196593497f,0.342997909f,-0.0822945088f,-0.450852722f,-0.0396065488f,-0.417444199f,-0.460508555f,0.011140638f,-0.165431067f,-0.0912121385f,0.20967783f,0.186076924f,0.107078902f,0.189260349f,0.224677593f,-0.19142881f,0.393361628f,-0.227393001f,-0.411644697f,-0.0796651766f};
static const float B=0.399419039f;
float q3_sigmoid(float x){return 1.0f/(1.0f+expf(-x));}
float q3_predict_prob(const float x[Q3_NUM_FEATURES]){float s=B;for(int i=0;i<Q3_NUM_FEATURES;i++)s+=W[i]*x[i];return q3_sigmoid(s);}
int q3_predict_label(const float x[Q3_NUM_FEATURES]){return q3_predict_prob(x)>0.5f?1:0;}
