#ifndef ARCHITECTURE_H
#define ARCHITECTURE_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define NO_INIT -1
#define GLOROT_UNIFORM_INIT 0
#define GLOROT_GAUSSIAN_INIT 1
#define UNIFORM_INIT 2
#define GAUSSIAN_INIT 3
#define HE_INIT 4

typedef struct neuron
{
    uint16_t input_size;
    float bias;
    float *weights;
} neuron;

typedef struct layer
{
    uint16_t input_size;
    uint16_t n_neurons;
    int activation;
    int initialization;
    neuron **neurons;
    float *w_input; // weighted input
    float *output;
    float *delta;
} layer;

typedef struct network
{
    uint16_t n_layers;
    uint16_t current_layer_ind;
    layer **layers;
} network;

int init_neuron(neuron *neur, uint16_t input_size, int initialization, float init_param);
int init_layer(layer *l, uint16_t input_size, uint16_t n_neurons, int initialization, int activation);
int init_network(network *nk, uint16_t n_layers);
int addinit_layer(network *nk, uint16_t input_size, uint16_t n_neurons, int initialization, int activation);
int add_layer(network *nk, layer *l);
void free_neuron(neuron *neur);
void free_layer(layer *l);
void free_network(network *nk);

#endif