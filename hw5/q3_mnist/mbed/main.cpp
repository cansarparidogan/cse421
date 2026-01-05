#include "mbed.h"
#include "q3_mnist_model.h"

int main()
{
    float hu[Q3_NUM_FEATURES] = {
        0.10f, -0.05f, 0.02f, 0.01f, -0.03f, 0.04f, 0.08f
    };

    while(true){
        float p = q3_predict_prob_from_hu(hu);
        int y = q3_predict_label_from_hu(hu);
        printf("MNIST prob=%.4f label=%d\r\n", p, y);
        thread_sleep_for(2000);
    }
}
