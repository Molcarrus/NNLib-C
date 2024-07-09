#include "architecture.h"
#include "nnc_utils.h"

int init_neuron(neuron *neur, uint16_t input_size, int initialization, float init_param)
{
    neur->input_size = input_size;

    neur->weights = (float *)malloc(input_size * sizeof(float));
    if (neur->weights == NULL)
    {
        fprintf(stderr, "[ERROR] init_neuron(): Could not allocate %u weights\n", (unsigned int)input_size);
        return EXIT_FAILURE;
    }

    if (initialization == NO_INIT)
    {
        memset(neur->weights, 0, input_size * sizeof(float));
        return EXIT_SUCCESS;
    }

    for (uint16_t i = 0; i < input_size; i++)
    {

        switch (initialization)
        {
        case GLOROT_UNIFORM_INIT:
        case UNIFORM_INIT:
            neur->weights[i] = uniform_rand(-init_param, init_param);
            if (i == 0)
                neur->bias = uniform_rand(-init_param, init_param);
            break;
        case GLOROT_GAUSSIAN_INIT:
        case GAUSSIAN_INIT:
            neur->weights[i] = gauss_rand(0.0, init_param);
            if (i == 0)
                neur->bias = gauss_rand(0.0, init_param);
            break;
        case NO_INIT:
        default:
            neur->weights[i] = 0.0;
            if (i == 0)
                neur->bias = 0.0;
        }
    }

    return EXIT_SUCCESS;
}

int init_layer(layer *l, uint16_t input_size, uint16_t n_neurons, int initialization, int activation)
{
    l->input_size = input_size;
    l->n_neurons = n_neurons;
    l->initialization = initialization;
    l->activation = activation;

    l->w_input = (float *)malloc(n_neurons * sizeof(float));
    if (l->w_input == NULL)
    {
        fprintf(stderr, "[ERROR] init_layer(): Could not allocate w_input (%u floats)\n", (unsigned int)n_neurons);
        return EXIT_FAILURE;
    }
    memset(l->w_input, 0, n_neurons * sizeof(float));

    l->output = (float *)malloc(n_neurons * sizeof(float));
    if (l->output == NULL)
    {
        fprintf(stderr, "[ERROR] init_layer(): Could not allocate output (%u floats)\n", (unsigned int)n_neurons);
        free(l->w_input);
        return EXIT_FAILURE;
    }
    memset(l->output, 0, n_neurons * sizeof(float));

    l->delta = (float *)malloc(n_neurons * sizeof(float));
    if (l->delta == NULL)
    {
        fprintf(stderr, "[ERROR] init_layer(): Could not allocate delta (%u floats)\n", (unsigned int)n_neurons);
        free(l->w_input);
        free(l->output);
        return EXIT_FAILURE;
    }
    memset(l->delta, 0, n_neurons * sizeof(float));

    l->neurons = (neuron **)malloc(n_neurons * sizeof(neuron *));
    if (l->neurons == NULL)
    {
        fprintf(stderr, "[ERROR] init_layer(): Could not allocate %u neurons in layer.\n", (unsigned int)n_neurons);
        free(l->output);
        free(l->w_input);
        free(l->delta);
        return EXIT_FAILURE;
    }

    float init_param = 0.0;
    switch (initialization)
    {
    case GLOROT_UNIFORM_INIT:
        init_param = (float)sqrt(6.0 / ((float)input_size + (float)n_neurons));
        break;
    case GLOROT_GAUSSIAN_INIT:
        init_param = sqrt(2.0 / ((float)input_size + (float)n_neurons));
        break;
    case UNIFORM_INIT:
        init_param = 0.01;
        break;
    case GAUSSIAN_INIT:
    default:
        init_param = 0.1;
        break;
    }
    int ret = EXIT_SUCCESS;
    for (uint16_t i = 0; i < n_neurons; i++)
    {
        l->neurons[i] = (neuron *)malloc(sizeof(neuron));
        if (l->neurons[i] == NULL)
        {
            fprintf(stderr, "[ERROR] init_layer(): Could not allocate neuron %u\n", (unsigned int)i);
            for (uint16_t j = 0; j < i; j++)
                free_neuron(l->neurons[j]);
            return EXIT_FAILURE;
        }
        ret = init_neuron(l->neurons[i], input_size, initialization, init_param);
        if (ret == EXIT_FAILURE)
        {
            fprintf(stderr, "[ERROR] init_layer(): Could not initiate neuron %u\n", (unsigned int)i);
            for (uint16_t j = 0; j < i; j++)
                free_neuron(l->neurons[j]);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

int init_network(network *nk, uint16_t n_layers)
{
    nk->n_layers = n_layers;
    nk->current_layer_ind = 0;
    nk->layers = (layer **)malloc(n_layers * sizeof(layer *));
    if (nk->layers == NULL)
    {
        perror("[ERROR] init_network(): Could not allocate layers");
        return EXIT_FAILURE;
    }
    memset(nk->layers, 0, n_layers * sizeof(layer *));
    return EXIT_SUCCESS;
}

int addinit_layer(network *nk, uint16_t input_size, uint16_t n_neurons, int initialization, int activation)
{
    if (nk->current_layer_ind >= nk->n_layers)
    {
        fprintf(stderr, "[ERROR] addinit_layer(): Network contains already %u out of max. %u layers, cannot add a new layer.\n",
                (unsigned int)nk->current_layer_ind + 1, (unsigned int)nk->n_layers);
        return EXIT_FAILURE;
    }
    layer *l = (layer *)malloc(sizeof(layer));
    if (l == NULL)
    {
        perror("[ERROR] add_layer(): Could not allocate new layer.");
        return EXIT_FAILURE;
    }
    if (nk->current_layer_ind > 0) // if layer is not the first, input_size ignored and forced to output size of previous layer
        input_size = nk->layers[nk->current_layer_ind - 1]->n_neurons;
    int ret = init_layer(l, input_size, n_neurons, initialization, activation);
    if (ret == EXIT_FAILURE)
    {
        perror("[ERROR] add_layer(): Could not initialize layer.");
        return EXIT_FAILURE;
    }
    nk->layers[nk->current_layer_ind] = l;
    nk->current_layer_ind = nk->current_layer_ind + 1;
    return EXIT_SUCCESS;
}

int add_layer(network *nk, layer *l)
{
    if (nk->current_layer_ind >= nk->n_layers - 1)
    {
        fprintf(stderr, "[ERROR] addinit_layer(): Network contains already %u out of max. %u layers, cannot add a new layer.\n",
                (unsigned int)nk->current_layer_ind + 1, (unsigned int)nk->n_layers);
        return EXIT_FAILURE;
    }
    nk->layers[nk->current_layer_ind] = l;
    nk->current_layer_ind = nk->current_layer_ind + 1;
    return EXIT_SUCCESS;
}

void free_neuron(neuron *neur)
{
    free(neur->weights);
    return;
}

void free_layer(layer *l)
{
    for (uint16_t i = 0; i < l->n_neurons; i++)
        free_neuron(l->neurons[i]);
    free(l->w_input);
    free(l->output);
    free(l->delta);
    free(l->neurons);
    return;
}

void free_network(network *nk)
{
    for (uint16_t i = 0; i < nk->current_layer_ind; i++)
        free_layer(nk->layers[i]);
    free(nk->layers);
    return;
}
