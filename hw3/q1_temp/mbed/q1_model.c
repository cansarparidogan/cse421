#include "q1_model.h"
static const float W[Q1_K]={5.753911f,-2.5325842f,-1.0652208f,0.34272864f,0.31941232f};
static const float B=19.191431f;
static const float MU[Q1_K]={19.192223f,19.191868f,19.189758f,19.185604f,19.17918f};
static const float SIGMA[Q1_K]={2.860866f,2.8605983f,2.859163f,2.8568242f,2.8541074f};
float q1_predict_temperature(const float x[Q1_K])
{
float y=B;
for(int i=0;i<Q1_K;i++){float xn=(x[i]-MU[i])/SIGMA[i];y+=W[i]*xn;}
return y;
}
