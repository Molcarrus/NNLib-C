#include "pass.h"
#include "activations.h"

int forward(network *nk, uint16_t input_s, float input[input_s], uint16_t output_s, float output[output_s])
{
    // Input and output size verification
    if (nk->layers[0]->input_size != input_s)
    {
        fprintf(stderr, "[ERROR] forward(): Input size required by the first layer of the network is %u, but input of size %u was provided.\n", nk->layers[0]->input_size, input_s);
        return EXIT_FAILURE;
    }
    if (nk->layers[nk->current_layer_ind - 1]->n_neurons != output_s)
    {
        fprintf(stderr, "[ERROR] forward(): Array of size %u provided to store the output, but the output of hte network is of size %u.\n", output_s, nk->layers[nk->n_layers - 1]->n_neurons);
        return EXIT_FAILURE;
    }

    // Forward round
    uint16_t is = 0;  // input size of current layer
    float *it = NULL; // input array of current layer
    uint16_t neuron_ind = 0, i = 0;
    float w_input = 0.0;     // weighted imput of neuron
    float neur_output = 0.0; // neuron output

    for (uint16_t layer_ind = 0; layer_ind < nk->current_layer_ind; layer_ind++)
    {
        is = (layer_ind == 0) ? input_s : nk->layers[layer_ind - 1]->n_neurons;
        it = (layer_ind == 0) ? input : nk->layers[layer_ind - 1]->output;
        for (neuron_ind = 0; neuron_ind < nk->layers[layer_ind]->n_neurons; neuron_ind++)
        {
            w_input = 0.0; // weighted
            for (i = 0; i < is; i++)
                w_input += it[i] * nk->layers[layer_ind]->neurons[neuron_ind]->weights[i];
            w_input += nk->layers[layer_ind]->neurons[neuron_ind]->bias;
            nk->layers[layer_ind]->w_input[neuron_ind] = w_input;
            neur_output = activate(nk->layers[layer_ind]->activation, w_input, false);
            nk->layers[layer_ind]->output[neuron_ind] = neur_output;
        }
    }

    memcpy(output, nk->layers[nk->current_layer_ind - 1]->output, output_s * sizeof(float));
    return EXIT_SUCCESS;
}

int batch_forward(network *nk, uint16_t batch_s, uint16_t input_s, float input[batch_s][input_s], uint16_t output_s, float output[batch_s][output_s])
{
    int ret = 0;
    for (uint16_t i = 0; i < batch_s; i++)
    {
        ret = forward(nk, input_s, input[i], output_s, output[i]);
        if (ret == EXIT_FAILURE)
        {
            fprintf(stderr, "[ERROR] batch_forward(): input %u in batch could not be forwarded.\n", i);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

int backpropagation(network *nk, uint16_t output_s, float error[output_s], uint16_t input_s, float input[input_s], float lr)
{
    // Check if output_s and input_s fit
    if (output_s != nk->layers[nk->current_layer_ind - 1]->n_neurons)
    {
        fprintf(stderr, "[ERROR] backpropagation(): Size of error: %u different of network output size: %u.\n", output_s, nk->layers[nk->n_layers - 1]->n_neurons);
        return EXIT_FAILURE;
    }
    if (input_s != nk->layers[0]->input_size)
    {
        fprintf(stderr, "[ERROR] backpropagation(): Network input size: %u different as provided input size: %u.\n", input_s, nk->layers[0]->input_size);
        return EXIT_FAILURE;
    }

    uint16_t neuron_ind, neuron_ind_1, weight_ind; // neuron_ind_1: next layer neurons ind
    int16_t layer_ind;
    float neuron_err = 0.0, sigma_prime = 0.0, op = 0, activation_v = 0.0, w_correction = 0.0; // op: output of current neuron
    float d = 0.0;

    for (layer_ind = (int16_t)nk->current_layer_ind - 1; layer_ind >= 0; layer_ind--)
    {
        for (neuron_ind = 0; neuron_ind < nk->layers[layer_ind]->n_neurons; neuron_ind++)
        {
            if (layer_ind == nk->current_layer_ind - 1)
                neuron_err = error[neuron_ind];
            else
            {
                neuron_err = 0.0;
                for (neuron_ind_1 = 0; neuron_ind_1 < nk->layers[layer_ind + 1]->n_neurons; neuron_ind_1++)
                    neuron_err += nk->layers[layer_ind + 1]->neurons[neuron_ind_1]->weights[neuron_ind] * nk->layers[layer_ind + 1]->delta[neuron_ind_1];
            }
            op = nk->layers[layer_ind]->output[neuron_ind];
            sigma_prime = activate(nk->layers[layer_ind]->activation, op, true);
            d = neuron_err * sigma_prime;
            nk->layers[layer_ind]->delta[neuron_ind] = d;

            // Update neuron biase
            nk->layers[layer_ind]->neurons[neuron_ind]->bias -= lr * d;
            // Update neuron weights
            for (weight_ind = 0; weight_ind < nk->layers[layer_ind]->neurons[neuron_ind]->input_size; weight_ind++)
            {
                activation_v = (layer_ind == 0) ? input[weight_ind] : nk->layers[layer_ind - 1]->output[weight_ind];
                w_correction = lr * activation_v * d;
                nk->layers[layer_ind]->neurons[neuron_ind]->weights[weight_ind] -= w_correction;
            }
        }
    }
    return EXIT_SUCCESS;
}

int backpropagation_batch(network *nk, uint16_t batch_s, uint16_t output_s, float error[batch_s][output_s], uint16_t input_s, float input[batch_s][input_s], float lr)
{
    int ret;
    for (uint16_t i = 0; i < batch_s; i++)
    {
        ret = backpropagation(nk, output_s, error[i], input_s, input[i], lr);
        if (ret == EXIT_FAILURE)
        {
            perror("[ERROR] backpropagation_batch(): Backpropagation failed for batch.");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}