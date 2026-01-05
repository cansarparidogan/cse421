#include "q1_har_model.h"
#include <math.h>

static float relu(float x){ return x>0.0f?x:0.0f; }

/* WEIGHTS + BIASES (HW4 Q1) */
static const float W1[Q1_H1][Q1_NUM_FEATURES] = { {0} };
static const float B1[Q1_H1] = {0};
static const float W2[Q1_H2][Q1_H1] = { {0} };
static const float B2[Q1_H2] = {0};
static const float W3[Q1_NUM_CLASSES][Q1_H2] = { {0} };
static const float B3[Q1_NUM_CLASSES] = {0};

int q1_har_predict(const float x[Q1_NUM_FEATURES])
{
    float h1[Q1_H1];
    float h2[Q1_H2];

    for(int i=0;i<Q1_H1;i++){
        float s=B1[i];
        for(int j=0;j<Q1_NUM_FEATURES;j++) s+=W1[i][j]*x[j];
        h1[i]=relu(s);
    }

    for(int i=0;i<Q1_H2;i++){
        float s=B2[i];
        for(int j=0;j<Q1_H1;j++) s+=W2[i][j]*h1[j];
        h2[i]=relu(s);
    }

    int arg=0;
    float best=-1e30f;
    for(int i=0;i<Q1_NUM_CLASSES;i++){
        float s=B3[i];
        for(int j=0;j<Q1_H2;j++) s+=W3[i][j]*h2[j];
        if(s>best){ best=s; arg=i; }
    }
    return arg;
}
