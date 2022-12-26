#include <stdlib.h>
#include "layer.h"

void layer_create(struct layer_t* layer, int num_neurons)
{
	layer->num_neu = num_neurons;
	layer->neu = (struct neuron_t *) malloc(num_neurons * sizeof(struct neuron_t));
	return;
}

void layer_destroy(struct layer_t* layer)
{
	int i;
	for(i=0; i<layer->num_neu; i++)
		neuron_destroy( &layer->neu[i] );
	free(layer->neu);
	return;
}
