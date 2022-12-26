#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "engine.h"

static void get_inputs(int num_training_ex, int num_output_neurons, float** input);
static void get_desired_outputs(int num_training_ex, int num_output_neurons, float** desired_outputs);

long long current_millis() {
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // calculate milliseconds
    // printf("milliseconds: %lld\n", milliseconds);
    return milliseconds;
}

int main(int argc, char* argv[])
{
    int num_layers; //the number of layers
    int *num_neurons2; //the number of neurons in each layer
    int i;
    struct layer_t *lay, *input_layer, *output_layer;
    float learn_rate;
    int num_training_ex;
    float **input, **desired_outputs;

    time_t tm = time(NULL);
    srand( tm );

////prepare architecture parameter:BEGIN
    printf("Enter the number of Layers in Neural Network:\n");
    scanf("%d", &num_layers);

    num_neurons2 = (int*) malloc(num_layers * sizeof(int));
    memset(num_neurons2, 0, num_layers *sizeof(int));

    for(i=0; i<num_layers; i++)// Get number of neurons per layer
    {
        printf("Enter number of neurons in layer[%d]: \n",i+1);
        scanf("%d", &num_neurons2[i]);
    }

    printf("\n");
////prepare architecture parameter:END

////init:BEGIN
    if(create_architecture(num_layers, num_neurons2, &lay) != SUCCESS_CREATE_ARCHITECTURE)
    {
        printf("Error in creating architecture...\n");
        free( num_neurons2 );
        return ERR_INIT;
    }
    free( num_neurons2 );
    printf("Neural Network Created Successfully...\n\n");
////init:END

////input:BEGIN
    printf("Enter the learning rate (Usually 0.15): \n");
    scanf("%f", &learn_rate);
    printf("\n");

    printf("Enter the number of training examples: \n");
    scanf("%d", &num_training_ex);
    printf("\n");
////input:END

    input_layer = &lay[0];
    output_layer = &lay[num_layers-1];

    input = (float**) malloc(num_training_ex * sizeof(float*));
    desired_outputs = (float**) malloc(num_training_ex* sizeof(float*));
    for(i=0; i<num_training_ex; i++)
    {
        input[i] = (float*)malloc(input_layer->num_neu * sizeof(float));
        desired_outputs[i] = (float*)malloc(output_layer->num_neu * sizeof(float));
    }

//// Get Training Examples
    get_inputs(num_training_ex, input_layer->num_neu, input);
//// Get Output Labels
    get_desired_outputs(num_training_ex, output_layer->num_neu, desired_outputs);

////train_neural_net
    long long t0 = current_millis();
    train_neural_net(num_layers, num_training_ex, lay, input, desired_outputs, learn_rate);
    long long t1 = current_millis();
    printf("train_neural_net time %lld milli seconds\n", t1 - t0);

    test_neural_net(num_layers, lay);

////cleanup:BEGIN
    for(i=0; i<num_layers; i++)
        layer_destroy( &lay[i] );
    free( lay );
    for(i=0; i<num_training_ex; i++)
    {
        free( input[i] );
        free( desired_outputs[i] );
    }
    free( input );
    free( desired_outputs );
////cleanup:END

    return 0;
}

static void get_inputs(int num_training_ex, int num_output_neurons, float** input)
{
    int i,j;
    for(i=0; i<num_training_ex; i++)
    {
        printf("Enter the Inputs for training example[%d]:\n", i);
        for(j=0; j < num_output_neurons; j++)
            scanf("%f", &input[i][j]);
        printf("\n");
    }
}

static void get_desired_outputs(int num_training_ex, int num_output_neurons, float** desired_outputs)
{
    int i,j;

    for(i=0;i<num_training_ex;i++)
    {
        for(j=0; j<num_output_neurons; j++)
        {
            printf("Enter the Desired Outputs (Labels) for training example[%d]: \n",i);
            scanf("%f",&desired_outputs[i][j]);
            printf("\n");
        }
    }
}
