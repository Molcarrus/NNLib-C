#include "losses.h"

// Loss functions for regression tasks

float mse(float y_pred, float y_true, bool derivative)
{
    float ret = 0.0;
    if (derivative)
        ret = -2.0 * (y_true - y_pred);
    else
        ret = (y_true - y_pred) * (y_true - y_pred);
    return ret;
}

float msle(float y_pred, float y_true, bool derivative)
{
    float ret = 0.0;
    if (derivative)
        ret = 2.0 * (float)(log(1.0 + y_pred + EPS) - log(1.0 + (double)y_true + EPS)) / (1.0 + y_pred + EPS);
    else
        ret = (float)(log(1.0 + y_pred + EPS) - log(1.0 + (double)y_true + EPS)) * (log(1.0 + (double)y_pred + EPS) - (float)log(1.0 + (double)y_true + EPS));
    return ret;
}

float mae(float y_pred, float y_true, bool derivative)
{
    float ret = 0.0;
    if (derivative)
        ret = (y_true > y_pred) ? 1.0 : -1.0;
    else
        ret = (y_true > y_pred) ? y_true - y_pred : y_pred - y_true;
    return ret;
}

// Loss functions for binary classification tasks

float binary_cross_entropy(float y_pred, float y_true, bool derivative)
{
    // y_true = 0 or 1
    float ret = 0.0;
    if (derivative)
        ret = -((y_true / (y_pred + EPS)) - ((1.0 - y_true) / (1.0 - y_pred + EPS)));
    else
        ret = -(float)(y_true * log((double)y_pred + EPS) + (1.0 - y_true) * log(1.0 - (double)y_pred + EPS));
    return ret;
}

float hinge(float y_pred, float y_true, bool derivative)
{
    // y_true = -1 or 1
    float ret = 0.0;
    if (derivative)
        ret = (y_true * y_pred < 0) ? -y_true : 0.0;
    else
        ret = (y_true * y_pred < 0) ? 1 - (y_true * y_pred) : 0.0;
    return ret;
}

float sq_hinge(float y_pred, float y_true, bool derivative)
{
    // y_true = -1 or 1
    float ret = 0.0;
    if (derivative)
        ret = (y_true * y_pred < 0) ? -(y_true * y_true) * y_pred : 0.0;
    else
    {
        ret = (y_true * y_pred < 0) ? 1 - (y_true * y_pred) : 0.0;
        ret *= ret;
    }
    return ret;
}

// TODO: loss functions for multiclass classification (and make biases stable to 1)

float multiple_cross_entropy(float y_pred, float y_true, bool derivative)
{
    float ret = 0.0;
    if (derivative)
        ret = y_true / (y_pred + EPS);
    else
        ret = y_true * (float)log((double)y_pred + EPS);
    return ret;
}

float kullback_leibler(float y_pred, float y_true, bool derivative)
{
    float ret = 0.0;
    if (derivative)
        ret = -(y_true * y_true) / (y_pred + EPS);
    else
        ret = y_true * (float)log((double)(y_true / (y_pred + EPS)));
    return ret;
}

float loss(int lossid, float y_pred, float y_true, bool derivative)
{
    float ret = 0.0;
    switch (lossid)
    {
    case MSE_LOSS:
        ret = mse(y_pred, y_true, derivative);
        break;
    case MSLE_LOSS:
        ret = msle(y_pred, y_true, derivative);
        break;
    case MAE_LOSS:
        ret = mae(y_pred, y_true, derivative);
        break;
    case BINARY_CROSS_ENTROPY_LOSS:
        ret = binary_cross_entropy(y_pred, y_true, derivative);
        break;
    case HINGE_LOSS:
        ret = hinge(y_pred, y_true, derivative);
        break;
    case SQ_HINGE:
        ret = sq_hinge(y_pred, y_true, derivative);
        break;
    case MULTIPLE_CROSS_ENTROPY_LOSS:
        ret = multiple_cross_entropy(y_pred, y_true, derivative);
        break;
    case KULLBACK_LEIBLER_LOSS:
        ret = kullback_leibler(y_pred, y_true, derivative);
        break;
    default:
        printf("[WARNING] loss(): Loss with id=%i not identified.\n", lossid);
        break;
    }
    return ret;
}

void loss_multi(int lossid, uint16_t output_s, float y_pred[output_s], float y_true[output_s], bool derivative, float loss_v[output_s])
{
    for (uint16_t i = 0; i < output_s; i++)
        loss_v[i] = loss(lossid, y_pred[i], y_true[i], derivative);
}

void loss_batch(int lossid, uint16_t batch_s, uint16_t output_s, float y_pred[batch_s][output_s], float y_true[batch_s][output_s], bool derivative, float loss_v[batch_s][output_s])
{
    uint16_t i = 0;
    for (i = 0; i < batch_s; i++)
        loss_multi(lossid, output_s, y_pred[i], y_true[i], derivative, loss_v[i]);
}