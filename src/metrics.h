#ifndef METRICS_H
#define METRICS_H

#include <stdint.h>

float acc(uint16_t batch_s, float y_true[batch_s][1], float y_pred[batch_s][1], float thr);

#endif