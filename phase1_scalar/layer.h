#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <random>
#include <memory>
#include "activation.h"

using namespace std;

class DenseLayer
{
    private:
        int input_size; // how many inputs a layer recieves
        int output_size; //how many neurons this layer contains
        shared_ptr<Activation> activation; // stores which activation function this layer uses, way to store ptr to sigmoid obj

        // Weights and biases
        vector<vector<double>> W; // output_size x input_size, keep it private as we dont want main() to change it accidentaly
        vector<double> b; // output_size

        // Cache for backprop
        vector<double> last_input; // what u r recieving from prev layer
        vector<double> last_z; // pre-activation
        vector<double> last_a; // post-activation

        // Gradients
        vector<vector<double>> dW;
        vector<double> db;

    public:
        // Constructor
        DenseLayer(int in_size, int out_size, shared_ptr<Activation> act); //initialises layer (act = which activation func to use)

        // Forward pass
        vector<double> forward(const vector<double>& input);

        // Backward pass
        vector<double> backward(const vector<double>& grad_output);

        // Update weights
        void update(double learning_rate);
};

DenseLayer::DenseLayer(int in_size, int out_size, shared_ptr<Activation> act)
{
    this->input_size = in_size;
    this->output_size = out_size;
    activation = act;

    // Initialize weight matrix (output_size x input_size)
    W.resize(output_size, vector<double>(input_size)); //o/p size = no of nuerons, i/p size = weights of each neuron
    b.resize(output_size);
    dW.resize(output_size, vector<double>(input_size));
    db.resize(output_size);

    // Random initialization
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-0.5, 0.5);

    //initialising the hidden layer
    for (int i=0; i<output_size; i++)
    {
        b[i] = dis(gen);
        for (int j=0; j<input_size; j++)
        {
            W[i][j] = dis(gen);
        }
    }
}

vector<double> DenseLayer :: forward(const vector<double>& input) // will return the vector of all the neurons after activation
{
    //saves copy of input for back propogation
    last_input = input; // vector<double> last_input; // what u r recieving from prev layer

    // Compute z = W * input + b (pre-activation)
    last_z.resize(output_size); // prepare cache vectors to store values of this layer

    for (int i=0; i<output_size; i++) // i = nueron
    {
        last_z[i] = b[i]; //biases
        for (int j=0; j<input_size; j++)
        {
            last_z[i] += W[i][j] * input[j];
        }
    }

    // Apply activation function element-wise
    last_a.resize(output_size);
    for (int i=0; i<output_size; i++)
    {
        last_a[i] = activation->apply(last_z[i]);
    }

    return last_a; // return vector of activated outputs
}

vector<double> DenseLayer :: backward(const vector<double>& grad_output) // grad_output = gradient coming from the next layer //gradients which will be sent to the prev layer
{
    // grad_output is dL/dA (gradient of loss wrt output of this layer)
    vector<double> grad_input(input_size, 0.0);  // will store dL/dInput // holds grads we need to send to next layer (we are coming back)
    vector<double> dZ(output_size, 0.0);         // dL/dZ = dL/dA * dA/dZ //

    // Compute dZ = grad_output * activation'(z)
    for (int i=0; i<output_size; i++)
    {
        dZ[i] = grad_output[i] * activation->derivative(last_z[i]);
    }

    // Compute gradients for weights and biases
    for (int i=0; i<output_size; i++)
    {
        db[i] = dZ[i];
        for (int j=0; j<input_size; j++)
        {
            dW[i][j] = dZ[i] * last_input[j];
        }
    }

    // Compute gradient for previous layer (grad_input)
    for (int j=0; j<input_size; j++)
    {
        grad_input[j] = 0.0;
        for (int i=0; i<output_size; i++)
        {
            grad_input[j] += W[i][j] * dZ[i];
        }
    }

    return grad_input;
}


void DenseLayer :: update(double learning_rate)
{
    for (int i=0; i<output_size; i++)
    {
        b[i] -= learning_rate * db[i];
        for (int j=0; j<input_size; j++)
        {
            W[i][j] -= learning_rate * dW[i][j];
        }
    }
}

#endif