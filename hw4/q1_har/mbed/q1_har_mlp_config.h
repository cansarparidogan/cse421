#ifndef Q1_HAR_MLP_CONFIG_H
#define Q1_HAR_MLP_CONFIG_H
#define Q1_NUM_FEATURES 10
#define Q1_H1 100
#define Q1_H2 100
#define Q1_NUM_CLASSES 6
extern const float Q1_W1[Q1_H1][Q1_NUM_FEATURES];
extern const float Q1_B1[Q1_H1];
extern const float Q1_W2[Q1_H2][Q1_H1];
extern const float Q1_B2[Q1_H2];
extern const float Q1_W3[Q1_NUM_CLASSES][Q1_H2];
extern const float Q1_B3[Q1_NUM_CLASSES];
int q1_har_predict(const float x[Q1_NUM_FEATURES]);
#endif
