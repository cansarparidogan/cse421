#include "mbed.h"
#include "q1_har_model.h"

UnbufferedSerial pc(USBTX, USBRX, 115200);

int main()
{
    float sample[Q1_NUM_FEATURES] = {
        0.12f, -0.34f, 1.05f, 0.22f, 0.9f,
        -0.1f, 0.44f, 0.3f, -0.2f, 0.6f
    };

    while(true){
        int cls = q1_har_predict(sample);
        printf("Predicted activity class: %d\r\n", cls);
        thread_sleep_for(2000);
    }
}
