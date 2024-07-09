#include "nnc_utils.h"
#include "architecture.h"
#include "activations.h"

// Knuth & Marsaglia: https://c-faq.com/lib/gaussian.html
float gauss_rand(float mean, float sigma)
{
	static float V1, V2, S;
	static int phase = 0;
	float X;

	if (phase == 0)
	{
		do
		{
			float U1 = (float)rand() / RAND_MAX;
			float U2 = (float)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	}
	else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return sigma * (X + mean);
}

float uniform_rand(float min_v, float max_v)
{
	float value = (float)rand() / RAND_MAX;
	return value * (max_v - min_v) + min_v;
}

// Functions for printing neuron, layer, network

void print_array(FILE *f, size_t n, float arr[n])
{
	fprintf(f, "[");
	for (size_t i = 0; i < n; i++)
	{
		fprintf(f, "%f%s", arr[i], (i == n - 1) ? "]\n" : ", ");
	}
	return;
}

void print_matrix(FILE *f, size_t n_rows, size_t n_cols, float mat[n_rows][n_cols])
{
	fprintf(f, "[\n");
	for (size_t i = 0; i < n_rows; i++)
	{
		fprintf(f, "\t");
		print_array(f, n_cols, mat[i]);
	}
	fprintf(f, "]\n");
}

void print_neuron(FILE *f, neuron *neur)
{
	fprintf(f, "Input size: %u\n", (unsigned int)neur->input_size);
	fprintf(f, "weights: ");
	print_array(f, (size_t)neur->input_size, neur->weights);
	fprintf(f, "bias: %f\n", neur->bias);
}

void get_activation_name(char activation_name[NAME_SIZE], int activation)
{
	switch (activation)
	{
	case IDENTITY_ACT:
		strncpy(activation_name, "Identity", 9);
		break;
	case SIGMOID_ACT:
		strncpy(activation_name, "Sigmoid", 8);
		break;
	case RELU_ACT:
		strncpy(activation_name, "ReLu", 5);
		break;
	case TANH_ACT:
		strncpy(activation_name, "Tanh", 5);
		break;
	default:
		strncpy(activation_name, "Unknown", 8);
		break;
	}
}

void print_layer(FILE *f, layer *l)
{
	char activation_name[NAME_SIZE];
	get_activation_name(activation_name, l->activation);
	fprintf(f, "Input Size: %u\nNumber of Neurons: %u\nActivation: %s\n",
			(unsigned int)l->input_size, (unsigned int)l->n_neurons, activation_name);
	for (uint16_t i = 0; i < l->n_neurons; i++)
	{
		fprintf(f, "= Neuron %u\n", (unsigned int)i + 1);
		print_neuron(f, l->neurons[i]);
	}
}

void print_network(FILE *f, network *nk)
{
	fprintf(f, "=== Network\nMax Number of Layers: %u\nCurrent Number of Layers: %u\n",
			(unsigned int)nk->n_layers, (unsigned int)nk->current_layer_ind);
	for (uint16_t i = 0; i < nk->current_layer_ind; i++)
	{
		fprintf(f, "== Layer %u\n", (unsigned int)i + 1);
		print_layer(f, nk->layers[i]);
	}
}

bool is_figure(char c)
{
	return ((c == '0') || (c == '1') || (c == '2') || (c == '3') || (c == '4') || (c == '5') || (c == '6') || (c == '7') || (c == '8') || (c == '9'));
}

bool is_numeric(char c)
{
	return (is_figure(c) || (c == '-') || (c == '.'));
}

float avg_matrix(size_t n_rows, size_t n_cols, float arr[n_rows][n_cols])
{
	float sum = 0.0;
	for (size_t i = 0; i < n_rows; i++)
		for (size_t j = 0; j < n_cols; j++)
			sum += arr[i][j];
	return sum / (float)(n_rows * n_cols);
}
