#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define IDENTITY_ACT 0
#define SIGMOID_ACT 1
#define RELU_ACT 2
#define TANH_ACT 3

float identity(float x, bool derivative);
float sigmoid(float x, bool derivative);
float relu(float x, bool derivative);
float tanhyp(float x, bool derivative);
float activate(int activation, float x, bool derivative);

#endif