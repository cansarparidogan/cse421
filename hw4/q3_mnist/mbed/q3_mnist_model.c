#include "q3_mnist_model.h"
#include <math.h>
static const float W[Q3_NUM_FEATURES]={-1.71439254f,3.97118521f,7.91567278f,4.7557497f,-1.09142804f,-1.85205495f,1.49529684f};
static const float MU[Q3_NUM_FEATURES]={0.334228545f,0.0447678752f,0.00818687677f,0.00141505199f,6.70835198e-06f,0.000155590489f,-6.46120043e-06f};
static const float SD[Q3_NUM_FEATURES]={0.0840459764f,0.0637657493f,0.0140614565f,0.00258283969f,8.24738527e-05f,0.00072079635f,6.31342918e-05f};
static const float B=5.80081272f;
float q3_sigmoid(float x){return 1.0f/(1.0f+expf(-x));}
float q3_predict_prob_from_hu(const float hu[Q3_NUM_FEATURES]){float s=B;for(int i=0;i<Q3_NUM_FEATURES;i++){float z=(hu[i]-MU[i])/SD[i];s+=W[i]*z;}return q3_sigmoid(s);}
int q3_predict_label_from_hu(const float hu[Q3_NUM_FEATURES]){return q3_predict_prob_from_hu(hu)>0.5f?1:0;}
