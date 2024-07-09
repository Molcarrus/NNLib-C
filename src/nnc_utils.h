#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include "architecture.h"

#define NAME_SIZE 20

float gauss_rand(float mean, float sigma);
float uniform_rand(float min_v, float max_v);
void print_array(FILE *f, size_t n, float arr[n]);
void print_matrix(FILE *f, size_t n_rows, size_t n_cols, float mat[n_rows][n_cols]);
void print_neuron(FILE *f, neuron *neur);
void get_activation_name(char activation_name[NAME_SIZE], int activation);
void print_layer(FILE *f, layer *l);
void print_network(FILE *f, network *nk);
bool is_figure(char c);
bool is_numeric(char c);
float avg_matrix(size_t n_rows, size_t n_cols, float arr[n_rows][n_cols]);

#endif