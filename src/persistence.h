#ifndef PERSISTENCE_H
#define PERSISTENCE_H

#include <stdlib.h>
#include <stdbool.h>
#include "architecture.h"

#define NO_STAGE 0
#define NEURON_STAGE 1
#define LAYER_STAGE 2
#define NETWORK_STAGE 3

int persist_network(network *nk, char *fp);
int name2activation(char *name);
int load_network(network *nk, char *fp);

#endif