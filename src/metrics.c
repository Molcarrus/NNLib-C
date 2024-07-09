#include "metrics.h"

float acc(uint16_t batch_s, float y_true[batch_s][1], float y_pred[batch_s][1], float thr)
{
    float acc = 0.0;
    for (uint16_t i = 0; i < batch_s; i++)
        acc += (float)((y_true[i][1] > thr) == (y_pred[i][1] > thr));
    acc /= (float)batch_s;
    return acc;
}