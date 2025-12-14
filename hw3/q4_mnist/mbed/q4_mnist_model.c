#include "q4_mnist_model.h"
#include <math.h>
static const float W[Q4_NUM_FEATURES]={-1.71729457f,4.00611305f,7.78659296f,5.15007305f,-0.912746906f,-2.39454031f,1.5281986f};
static const float MU[Q4_NUM_FEATURES]={0.334228545f,0.0447678752f,0.00818687677f,0.00141505199f,6.70835198e-06f,0.000155590489f,-6.46120043e-06f};
static const float SD[Q4_NUM_FEATURES]={0.0840459764f,0.0637657493f,0.0140614565f,0.00258283969f,8.24738527e-05f,0.00072079635f,6.31342918e-05f};
static const float B=5.80579472f;
float q4_sigmoid(float x){return 1.0f/(1.0f+expf(-x));}
float q4_predict_prob_from_hu(const float hu[Q4_NUM_FEATURES]){float s=B;for(int i=0;i<Q4_NUM_FEATURES;i++){float z=(hu[i]-MU[i])/SD[i];s+=W[i]*z;}return q4_sigmoid(s);}
int q4_predict_label_from_hu(const float hu[Q4_NUM_FEATURES]){return q4_predict_prob_from_hu(hu)>0.5f?1:0;}
