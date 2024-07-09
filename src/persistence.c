#include "persistence.h"
#include "nnc_utils.h"
#include "activations.h"

int persist_network(network *nk, char *fp)
{
    FILE *f = fopen(fp, "w");
    if (f == NULL)
    {
        fprintf(stderr, "[ERROR] persist_network(): Could not open %s (in write mode)\n", fp);
        return EXIT_FAILURE;
    }
    print_network(f, nk);
    printf("[INFO] persist(): Network persisted in %s\n", fp);
    fclose(f);
    return EXIT_SUCCESS;
}

int name2activation(char *name)
{
    int activation = -1;
    if (!strncmp(name, "Identity", 9))
        activation = IDENTITY_ACT;
    else if (!strncmp(name, "Sigmoid", 8))
        activation = SIGMOID_ACT;
    else if (!strncmp(name, "ReLu", 5))
        activation = RELU_ACT;
    else if (!strncmp(name, "Tanh", 5))
        activation = TANH_ACT;
    return activation;
}

int load_network(network *nk, char *fp)
{
    FILE *f = fopen(fp, "r");
    if (f == NULL)
    {
        fprintf(stderr, "[ERROR] load_network(): Could not open %s (in read mode)\n", fp);
        return EXIT_FAILURE;
    }

    int ret;
    int c;
    bool preq;         // previous character was '='
    uint8_t neq = 0;   // number of previous eq signs
    int cur_stage = 0; // current stage
    bool gonextline = false;
    bool is_firstword = true; // is first word in line?
    char firstword[11];       // used to identify following numerical value(s)
    uint8_t ind_firstword = 0;
    bool read_int = false;
    bool read_array = false;
    bool read_word = false;
    char current_ch[10] = {0}; // currently read value (integer, float or word)
    uint8_t current_ind = 0;   // character index when reading word or number
    uint8_t array_ind = 0;
    uint16_t layer_ind_1 = 0, neuron_ind_1 = 0; // current index + 1
    // current layer params
    int cur_activation = -1;
    uint16_t cur_input_size = 0, cur_n_neurons = 0;

    while ((c = fgetc(f)) != EOF)
    {
        if ((gonextline && (char)c != '\n') || (read_word && ((char)c == ' ' || (char)c == ':')))
            continue;

        else if ((char)c == '\n')
        {
            gonextline = false;
            is_firstword = true;

            if (neq > 0)
                neq = 0;

            if (read_int)
            {
                current_ch[current_ind] = '\0';
                current_ind = 0;
                read_int = false;
                // TODO: attribute number to right value
                if ((cur_stage == NETWORK_STAGE) && !strncmp(firstword, "Max", 4))
                {
                    init_network(nk, (uint16_t)atoi(current_ch));
                    nk->current_layer_ind = 0;
                }
                if ((cur_stage == LAYER_STAGE) && !strncmp(firstword, "Input", 6))
                    cur_input_size = (uint16_t)atoi(current_ch);
                if ((cur_stage == LAYER_STAGE) && !strncmp(firstword, "Number", 7))
                    cur_n_neurons = (uint16_t)atoi(current_ch);
            }
            if (read_word)
            {
                current_ch[current_ind] = '\0';
                current_ind = 0;
                read_word = false;
                if ((cur_stage == LAYER_STAGE) && !strncmp(firstword, "Activation", 11))
                {
                    cur_activation = name2activation(current_ch);
                    if (cur_activation == -1)
                    {
                        fprintf(stderr, "[ERROR] load_network(): Activation for layer %u could not be determined\n", layer_ind_1 + 1);
                        free_network(nk);
                        return EXIT_FAILURE;
                    }
                    ret = addinit_layer(nk, cur_input_size, cur_n_neurons, NO_INIT, cur_activation);
                    if (ret == EXIT_FAILURE)
                    {
                        fprintf(stderr, "[ERROR] load_network(): Could not initialize layer %u\n", layer_ind_1 + 1);
                        free_network(nk);
                        return EXIT_FAILURE;
                    }
                    else
                        layer_ind_1++;
                }
            }
            continue;
        }

        else if ((char)c == '=' && (firstword || preq))
        {
            neq++;
            if (!preq)
                preq = true;
        }

        else if (preq && (char)c != '=')
        {
            if ((cur_stage != NEURON_STAGE) && (neq == NEURON_STAGE))
                neuron_ind_1 = 0;

            cur_stage = neq;
            neq = 0;
            preq = false;
            gonextline = true;

            if (cur_stage == NEURON_STAGE)
                neuron_ind_1++;
        }

        else if (is_firstword && (char)c != ' ' && (char)c != ':')
        {
            firstword[ind_firstword] = (char)c;
            ind_firstword++;
        }

        else if (is_firstword && ((char)c == ' ' || (char)c == ':'))
        {
            is_firstword = false;
            firstword[ind_firstword] = '\0';
            ind_firstword = 0;
            if (!strncmp(firstword, "Max", 4) || !strncmp(firstword, "Current", 8) || !strncmp(firstword, "Input", 5) || !strncmp(firstword, "Number", 7))
                read_int = true;
            else if (!strncmp(firstword, "weights", 9) || !strncmp(firstword, "biases", 7))
                read_array = true;
            else if (!strncmp(firstword, "Activation", 11))
                read_word = true;
        }

        else if ((read_int && is_figure((char)c)) || (read_array && is_numeric((char)c)))
        {
            current_ch[current_ind] = (char)c;
            current_ind++;
        }

        else if (read_array && (char)c == ',')
        {
            current_ch[current_ind] = '\0';
            current_ind = 0;
            if (!strncmp(firstword, "weights", 8))
                nk->layers[layer_ind_1 - 1]->neurons[neuron_ind_1 - 1]->weights[array_ind] = (float)atof(current_ch);
            else if (!strncmp(firstword, "biases", 7))
                nk->layers[layer_ind_1 - 1]->neurons[neuron_ind_1 - 1]->biases[array_ind] = (float)atof(current_ch);
            else
                printf("[WARNING] load_network(): Could not assign %s at position %u\n", current_ch, array_ind);
            array_ind++;
        }

        else if (read_array && (char)c == ']')
        {
            current_ch[current_ind] = '\0';
            current_ind = 0;
            read_array = false;

            if (!strncmp(firstword, "weights", 8))
                nk->layers[layer_ind_1 - 1]->neurons[neuron_ind_1 - 1]->weights[array_ind] = (float)atof(current_ch);
            else if (!strncmp(firstword, "biases", 7))
                nk->layers[layer_ind_1 - 1]->neurons[neuron_ind_1 - 1]->biases[array_ind] = (float)atof(current_ch);
            else
                printf("[WARNING] load_network(): Could not assign %s at position %u\n", current_ch, array_ind);

            array_ind = 0;
        }

        else if (read_word)
        {
            current_ch[current_ind] = (char)c;
            current_ind++;
        }
    }

    fclose(f);
    return EXIT_SUCCESS;
}