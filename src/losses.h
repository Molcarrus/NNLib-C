#ifndef LOSSES_H
#define LOSSES_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>

#define EPS 1E-2

#define MSE_LOSS 1
#define MSLE_LOSS 2
#define MAE_LOSS 3
#define BINARY_CROSS_ENTROPY_LOSS 4
#define HINGE_LOSS 5
#define SQ_HINGE 6
#define MULTIPLE_CROSS_ENTROPY_LOSS 7
#define KULLBACK_LEIBLER_LOSS 8

float mse(float y_pred, float y_true, bool derivative);
float msle(float y_pred, float y_true, bool derivative);
float mae(float y_pred, float y_true, bool derivative);
float binary_cross_entropy(float y_pred, float y_true, bool derivative);
float hinge(float y_pred, float y_true, bool derivative);
float sq_hinge(float y_pred, float y_true, bool derivative);
float multiple_cross_entropy(float y_pred, float y_true, bool derivative);
float kullback_leibler(float y_pred, float y_true, bool derivative);

float loss(int lossid, float y_pred, float y_true, bool derivative);
void loss_multi(int lossid, uint16_t output_s, float y_pred[output_s], float y_true[output_s], bool derivative, float loss_v[output_s]);
void loss_batch(int lossid, uint16_t batch_s, uint16_t output_s, float y_pred[batch_s][output_s], float y_true[batch_s][output_s], bool derivative, float loss_v[batch_s][output_s]);

#endif