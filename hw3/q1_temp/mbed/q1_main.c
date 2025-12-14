#include <stdio.h>
#include "q1_model.h"
#include "q1_dataset.h"
static void update_window(float w[Q1_K],float x)
{
for(int i=0;i<Q1_K-1;i++)w[i]=w[i+1];
w[Q1_K-1]=x;
}
int main(void)
{
float window[Q1_K]={0};
for(int i=0;i<Q1_K;i++)window[i]=q1_samples[i];
for(int t=Q1_K;t<Q1_SAMPLE_COUNT;t++)
{
float pred=q1_predict_temperature(window);
printf("t=%d actual=%.3f pred=%.3f\r\n",t,q1_samples[t],pred);
update_window(window,q1_samples[t]);
}
while(1){}
}
