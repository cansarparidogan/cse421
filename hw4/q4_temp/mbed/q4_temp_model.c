#include "q4_temp_model.h"
static const float MU[Q4_PREV]={3.96491218f,3.96491218f,3.96710515f,3.96929836f,3.97149134f};
static const float SD[Q4_PREV]={1.99969113f,1.99969113f,1.99808586f,1.99647462f,1.99485576f};
static const float W[Q4_PREV]={-0.0520860218f,0.00093097653f,1.07658638e-08f,4.06154648e-08f,1.94196689f};
static const float B=3.97368431f;
float q4_temp_predict(const float prev[Q4_PREV]){
  float s=B;
  for(int i=0;i<Q4_PREV;i++){
    float z=(prev[i]-MU[i])/SD[i];
    s+=W[i]*z;
  }
  return s;
}
