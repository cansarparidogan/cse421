#include "mbed.h"
#include <math.h>
#include "bayes_har_config.h"

const char* ACTIVITY_NAMES[NUM_CLASSES] = {
    "Downstairs",
    "Jogging",
    "Sitting",
    "Standing",
    "Upstairs",
    "Walking"
};

const float SAMPLE_FEATURES[NUM_FEATURES] = {
    3.6435f,
    9.78425f,
    -0.3905f,
    60.0f,
    80.0f,
    22.0f,
    49.254883f,
    118.84375f,
    18.514763f,
    17.41225f
};

int bayes_har_predict(const float features[NUM_FEATURES])
{
    int best_class = 0;
    float best_score = -1e30f;

    for (int c = 0; c < NUM_CLASSES; c++) {
        float diff[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++)
            diff[i] = features[i] - MEANS[c][i];

        float quad = 0.0f;
        for (int i = 0; i < NUM_FEATURES; i++)
            for (int j = 0; j < NUM_FEATURES; j++)
                quad += diff[i] * INV_COVS[c][i][j] * diff[j];

        float score = -0.5f * (quad + logf(DETS[c] + 1e-12f))
                      + logf(CLASS_PRIORS[c] + 1e-12f);

        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }
    return best_class;
}

int main()
{
    int cls = bayes_har_predict(SAMPLE_FEATURES);
    printf("Predicted class index: %d\r\n", cls);
    printf("Predicted activity   : %s\r\n", ACTIVITY_NAMES[cls]);

    while (1) {
        ThisThread::sleep_for(1s);
    }
}
