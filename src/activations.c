#include "activations.h"

float identity(float x, bool derivative)
{
    float ret = derivative ? 1 : x;
    return ret;
}

float sigmoid(float x, bool derivative)
{
    float ret = 1 / (1 + exp(-(double)x));
    if (derivative)
        ret *= (1 - ret);
    return ret;
}

float relu(float x, bool derivative)
{
    float ret = 0.0;
    if (x > 0)
        ret = derivative ? 1.0 : x;
    return ret;
}

float tanhyp(float x, bool derivative)
{
    double xd = (double)x;
    float ret = (float)tanh(xd);
    if (derivative)
        ret = 1 - ret * ret;
    return ret;
}

float activate(int activation, float x, bool derivative)
{
    float value = 0.0;
    switch (activation)
    {
    case IDENTITY_ACT:
        value = identity(x, derivative);
        break;
    case SIGMOID_ACT:
        value = sigmoid(x, derivative);
        break;
    case RELU_ACT:
        value = relu(x, derivative);
        break;
    case TANH_ACT:
        value = tanhyp(x, derivative);
        break;
    default:
        printf("[WARNING] activate(): activation function not recognized (activation %i).\n", activation);
        break;
    }
    return value;
}