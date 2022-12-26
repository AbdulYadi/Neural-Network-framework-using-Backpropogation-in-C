#ifndef _LAYER_H_
#define _LAYER_H_

#include "neuron.h"

struct layer_t
{
	int num_neu;
	struct neuron_t *neu;
};

extern void layer_create(struct layer_t* layer, int num_neurons);
extern void layer_destroy(struct layer_t* layer);

#endif //_LAYER_H_
