#include "mbed.h"
#include "q2_kws_model.h"

int main()
{
    float x[Q2_NUM_FEATURES] = {
        0.12f, -0.08f, 0.33f, 0.05f, -0.22f, 0.19f, 0.11f, -0.02f, 0.07f, 0.14f, -0.09f, 0.03f, 0.25f,
        0.04f, 0.06f, -0.01f, 0.02f, 0.10f, -0.03f, 0.08f, -0.05f, 0.09f, 0.01f, -0.07f, 0.13f, 0.05f
    };

    while(true){
        float p = q2_predict_prob(x);
        int y = q2_predict_label(x);
        printf("KWS prob=%.4f label=%d\r\n", p, y);
        thread_sleep_for(2000);
    }
}
