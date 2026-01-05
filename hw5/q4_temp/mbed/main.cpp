#include "mbed.h"
#include "q4_temp_model.h"

int main()
{
    float prev[Q4_PREV] = { 12.4f, 12.6f, 12.5f, 12.7f, 12.8f };

    while(true){
        float y = q4_temp_predict(prev);
        printf("Predicted next temperature: %.3f\r\n", y);
        thread_sleep_for(2000);
    }
}
