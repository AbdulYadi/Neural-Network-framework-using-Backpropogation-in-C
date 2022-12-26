#include <stdlib.h>
#include "neuron.h"

void neuron_create(struct neuron_t *p, int num_out_weights)
{
	p->actv = 0.0;
	p->out_weights = num_out_weights > 0 ? (float*) malloc(num_out_weights * sizeof(float)) : NULL;
	p->bias=0.0;
	p->z = 0.0;

	p->dactv = 0.0;
	p->dw = num_out_weights > 0 ? (float*) malloc(num_out_weights * sizeof(float)) : NULL;
	p->dbias = 0.0;
	p->dz = 0.0;

	return;
}

void neuron_destroy(struct neuron_t *p)
{
	if( p->out_weights!=NULL ) {
		free( p->out_weights );
		p->out_weights = NULL;
	}

	if( p->dw!=NULL ) {
		free( p->dw );
		p->dw = NULL;
	}

	return;
}
