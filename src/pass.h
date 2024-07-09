#ifndef PASS_H
#define PASS_H

#include <stdint.h>
#include "architecture.h"

int forward(network *nk, uint16_t input_s, float input[input_s], uint16_t output_s, float output[output_s]);
int batch_forward(network *nk, uint16_t batch_s, uint16_t input_s, float input[batch_s][input_s], uint16_t output_s, float output[batch_s][output_s]);
int backpropagation(network *nk, uint16_t output_s, float error[output_s], uint16_t input_s, float input[input_s], float lr);
int backpropagation_batch(network *nk, uint16_t batch_s, uint16_t output_s, float error[batch_s][output_s], uint16_t input_s, float input[batch_s][input_s], float lr);

#endif