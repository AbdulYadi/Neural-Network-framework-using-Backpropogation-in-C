.PHONY: clean
backprop: main.o engine.o neuron.o layer.o
	gcc -o backprop -Wall main.o engine.o neuron.o layer.o -lm
main.o: main.c
	gcc -c -Wall main.c
engine.o: engine.c
	gcc -c -Wall engine.c
neuron.o: neuron.c
	gcc -c -Wall neuron.c
layer.o: layer.c
	gcc -c -Wall layer.c
clean:
	rm *.o
