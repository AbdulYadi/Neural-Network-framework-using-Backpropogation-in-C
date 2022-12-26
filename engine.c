#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "engine.h"
#include "neuron.h"

static int initialize_weights(int num_layers, struct layer_t* lay);
static void feed_input(struct layer_t *input_layer, float* input, bool print);
static void forward_prop(int num_layers, struct layer_t *lay, bool print);
static float compute_cost(struct layer_t *output_layer, float *desired_outputs);
static void back_prop(int num_layers, struct layer_t *lay, float *desired_outputs);
static void update_weights(int num_layers, struct layer_t *lay, float learn_rate);
static float sigmoid(float z);
static float dsigmoid(float s);
static float relu(float z);

int create_architecture(int num_layers, int *num_neurons, struct layer_t** lay)
{
    int i=0, j=0, num_out_weights;
    *lay = (struct layer_t*) malloc(num_layers * sizeof(struct layer_t));

    for(i=0; i<num_layers; i++)
    {
        layer_create(&(*lay)[i], num_neurons[i]);
        printf("Created Layer: %d\n", i+1);
        printf("Number of Neurons in Layer %d: %d\n", i+1, (*lay)[i].num_neu);

        for(j=0; j<num_neurons[i]; j++)
        {
            num_out_weights = i < (num_layers-1) ? num_neurons[i+1]
                : 0/*output layer does not have output weights*/;
            neuron_create(&(*lay)[i].neu[j], num_out_weights);
            printf("Neuron %d in Layer %d created\n", j+1, i+1);
        }
        printf("\n");
    }

    printf("\n");

    // Initialize the weights
    if( initialize_weights(num_layers, *lay) != SUCCESS_INIT_WEIGHTS )
    {
        printf("Error Initilizing weights...\n");
        return ERR_CREATE_ARCHITECTURE;
    }

    return SUCCESS_CREATE_ARCHITECTURE;
}

void train_neural_net(int num_layers, int num_training_ex,
    struct layer_t* lay, float** input, float **desired_outputs,
    float learn_rate)
{
    int i, it;
    struct layer_t *input_layer, *output_layer, *layer, *next_layer;

    float sse/*sum of squared error*/, mse/*mean of squared error*/;

    FILE* fp;

    fp = fopen("mse.csv", "w");

    // Stochastic Gradient Descent
    input_layer = &lay[0];
    output_layer = &lay[num_layers - 1];
    for(it=0; it<20000; it++)
    {
        for(i=0; i<num_training_ex; i++)
        {
            feed_input(input_layer, input[i], false);
            forward_prop(num_layers, lay, false);
            back_prop(num_layers, lay, desired_outputs[i]);
            update_weights(num_layers, lay, learn_rate);
        }
        sse = 0.0;
        for(i=0; i<num_training_ex; i++)
        {
            feed_input(input_layer, input[i], false);
            forward_prop(num_layers, lay, false);
            sse += compute_cost(output_layer, desired_outputs[i]);
        }
        mse = sse / (num_training_ex * output_layer->num_neu);
        if(fp)
            fprintf(fp, "%f\n", mse);
        if(mse<=0.001)
            break;
    }
    printf("Optimized mse %f after %d iteration\n", mse, it);

    if(fp)
        fclose(fp);

    // show best estimation
    int j, k;
    for(i=0; i<num_layers; i++)
    {
        layer = &lay[i];
        next_layer = layer + 1;
        printf("Layer %d (%s)\n", i+1, i==0 ? "Input" : (i==num_layers-1 ? "Output" : "Hidden"));
        for(j=0; j<layer->num_neu; j++)
        {
            printf("\tNeuron %d\n", j+1);
            for(k=0;
                i<num_layers-1 /*only for input and hidden*/
                && k<next_layer->num_neu;
                k++)
                printf("\t\tk %d weight %f\n", k+1, layer->neu[j].out_weights[k]);
            if(i>0)//only for hidden layer and output layer
                printf("\t\tbias %f\n", layer->neu[j].bias);
        }
    }
}

void test_neural_net(int num_layers, struct layer_t *lay)
{
    int i, stop = 0;
    struct layer_t *input_layer;

    input_layer = &lay[0];
    while( !stop )
    {
        printf("Enter input to test (-9999 to stop):\n");
        for(i=0; !stop && i<input_layer->num_neu; i++) {
            scanf("%f", &input_layer->neu[i].actv);
            if( input_layer->neu[i].actv == -9999 )
                stop = 1;
        }
        if(!stop)
            forward_prop(num_layers, lay, true);
    }
}

static int initialize_weights(int num_layers, struct layer_t* lay)
{
    int i,j,k;
    struct layer_t *layer, *next_layer, *output_layer;

    if(lay == NULL)
    {
        printf("No layers in Neural Network...\n");
        return ERR_INIT_WEIGHTS;
    }

    printf("Initializing weights...\n");

    for(i=0; i<num_layers-1/*exclude output layer*/; i++)
    {
        printf("Layer %d\n", i+1);
        layer = &lay[i];
        next_layer = layer + 1;
        for(j=0; j<layer->num_neu;j++)
        {
            printf("\tNeuron %d\n", j+1);
            for(k=0; k<next_layer->num_neu; k++)
            {
                // Initialize Output Weights for each neuron 0.0 <= weight <= 1.0
                layer->neu[j].out_weights[k] = ((double)rand())/((double)RAND_MAX);
                printf("\t\tk: %d weight: %f\n",k, layer->neu[j].out_weights[k]);
                //printf("%d:w[%d][%d]: %f\n",k,i,j, layer->neu[j].out_weights[k]);
                layer->neu[j].dw[k] = 0.0;
            }
            if(i>0)//only for hidden layer
                layer->neu[j].bias = ((double)rand())/((double)RAND_MAX);
        }
    }


    output_layer = &lay[num_layers-1];
    printf("Output Layer %d\n", i+1);
    for (j=0; j<output_layer->num_neu; j++) {
        output_layer->neu[j].bias = ((double)rand())/((double)RAND_MAX);
        printf("\tNeuron %d bias %f\n", j+1, output_layer->neu[j].bias);
    }

    printf("\n");

    return SUCCESS_INIT_WEIGHTS;
}

static void feed_input(struct layer_t *input_layer, float* input, bool print)
{
    int j;
    for(j=0; j<input_layer->num_neu; j++)
    {
        input_layer->neu[j].actv = input[j];
        if(print)
            printf("Input: %f\n", input_layer->neu[j].actv);
    }
}

static void forward_prop(int num_layers, struct layer_t *lay, bool print)
{
    int i,j,k;
    struct layer_t *layer, *prevlayer;

    for(i=1/*skip input layer*/; i<num_layers; i++)//only for hidden and output layers
    {
        layer = &lay[i];
        prevlayer = layer - 1;
        for(j=0; j<layer->num_neu; j++)
        {
            layer->neu[j].z = layer->neu[j].bias;
            for(k=0; k<prevlayer->num_neu; k++)
                layer->neu[j].z  += ((prevlayer->neu[k].out_weights[j]) * (prevlayer->neu[k].actv));

            layer->neu[j].actv = (i == num_layers-1)/*output*/ ? sigmoid(layer->neu[j].z) : relu(layer->neu[j].z);
            if(print && i == num_layers-1)//output layer
            {
                printf("Output: %d\n", (int)round(layer->neu[j].actv));
                printf("\n");
            }
        }
    }
}

static float compute_cost(struct layer_t *output_layer, float *desired_outputs)
{
    int j;
    float tmpcost;
    float tcost = 0;

    for(j=0; j<output_layer->num_neu; j++)
    {
        tmpcost = desired_outputs[j] - output_layer->neu[j].actv;
        tcost += (tmpcost * tmpcost);
    }

    return tcost;
}

static void back_prop(int num_layers, struct layer_t *lay, float *desired_outputs)
{
    int i,j,k;
    struct layer_t *output_layer, *layer, *prevlayer;

    output_layer = &lay[num_layers-1];
    prevlayer = output_layer - 1;
    // Output Layer
    for(j=0; j<output_layer->num_neu; j++)//assumption: output layer activation is sigmoid function
    {
        output_layer->neu[j].dz = (output_layer->neu[j].actv - desired_outputs[j]) * dsigmoid(output_layer->neu[j].actv);

        for(k=0; k<prevlayer->num_neu; k++)
        {
            prevlayer->neu[k].dw[j] = output_layer->neu[j].dz * prevlayer->neu[k].actv;
            prevlayer->neu[k].dactv = output_layer->neu[j].dz * prevlayer->neu[k].out_weights[j];
        }

        output_layer->neu[j].dbias = output_layer->neu[j].dz;
    }

    // Hidden Layers
    for(i=num_layers-2; i>0; i--)
    {
        layer = &lay[i];
        prevlayer = layer - 1;
        for(j=0; j<layer->num_neu; j++)
        {
            layer->neu[j].dz = layer->neu[j].z >= 0 ? layer->neu[j].dactv : 0;

            for(k=0; k<prevlayer->num_neu; k++)
            {
                prevlayer->neu[k].dw[j] = layer->neu[j].dz * prevlayer->neu[k].actv;
                if( i>1 )//exclude input layer i-1 == 0 for i==1
                    prevlayer->neu[k].dactv = layer->neu[j].dz * prevlayer->neu[k].out_weights[j];
            }

            layer->neu[j].dbias = layer->neu[j].dz;
        }
    }
}

static void update_weights(int num_layers, struct layer_t *lay, float learn_rate)
{
    int i,j,k;
    struct layer_t *layer, *nextlayer;

    for(i=0; i<num_layers-1; i++)
    {
        layer = &lay[i];
        nextlayer = layer + 1;

        for(j=0; j<layer->num_neu; j++)
        {
            for(k=0; k<nextlayer->num_neu; k++)// Update Weights
                layer->neu[j].out_weights[k] = (layer->neu[j].out_weights[k]) - (learn_rate * layer->neu[j].dw[k]);
            // Update Bias
            layer->neu[j].bias = layer->neu[j].bias - (learn_rate * layer->neu[j].dbias);
        }
    }
}

static float sigmoid(float z)
{
    return 1/(1+exp(-z));
}

static float dsigmoid(float s)
{
    return s * (1 - s);
}

static float relu(float z)
{
    return z < 0 ? 0 : z;
}
