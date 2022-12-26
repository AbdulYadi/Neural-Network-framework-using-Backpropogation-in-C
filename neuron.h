#ifndef _NEURON_H_
#define _NEURON_H_

typedef struct neuron_t
{
	float actv;
	float *out_weights;
	float bias;
	float z;

	float dactv;
	float *dw;
	float dbias;
	float dz;
} neuron;

extern void neuron_create(struct neuron_t *p, int num_out_weights);
extern void neuron_destroy(struct neuron_t *p);

#endif //_NEURON_H_
