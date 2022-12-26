#ifndef _ENGINE_H_
#define _ENGINE_H_

#define SUCCESS_INIT 0
#define ERR_INIT 1

#define SUCCESS_DINIT 0
#define ERR_DINIT 1

#define SUCCESS_INIT_WEIGHTS 0
#define ERR_INIT_WEIGHTS 1

#define SUCCESS_UPDATE_WEIGHTS 0

#define SUCCESS_CREATE_ARCHITECTURE 0
#define ERR_CREATE_ARCHITECTURE 1

#include "layer.h"
extern int create_architecture(int num_layers, int *num_neurons, struct layer_t** lay);
extern void train_neural_net(int num_layers, int num_training_ex,
    struct layer_t* lay, float** input, float **desired_outputs,
    float learn_rate);
extern void test_neural_net(int num_layers, struct layer_t *lay);

#endif //_ENGINE_H_
