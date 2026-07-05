#ifndef LAYER_MATRIX_H
#define LAYER_MATRIX_H

#include <Eigen/Dense>
#include <memory>
#include <random>
#include "activation.h"

using namespace std;
using namespace Eigen;

class DenseLayerMatrix
{
    private:
        int input_size; //no of inputs taken
        int output_size; //no of neurons in the respective layer
        shared_ptr<Activation> activation;

        MatrixXd W;
        VectorXd b;

        //cache variable
        MatrixXd last_input; // cache x
        MatrixXd last_z; // cache z - pre-activation
        MatrixXd last_a; // cache the activated layer - post-activation

        //gradient variables
        MatrixXd dW; // gradient for weights
        VectorXd db; // gradient for biases

    public:
        //constructor
        DenseLayerMatrix (int in_size, int out_size, shared_ptr<Activation> act);

        //forward pass
        MatrixXd forward(const MatrixXd& X);

        //backward pass
        MatrixXd backward(const MatrixXd& grad_output);

        //update the weights using the gradients computed
        void update(double learning_rate);

        //getter functions
        int get_input_size() const { return input_size; }
        int get_output_size() const { return output_size; }

        MatrixXd get_weights() const { return W; };
        VectorXd get_biases() const { return b; };
        
        void set_weights(const MatrixXd& weights, const VectorXd& biases)
        {
            W = weights;
            b = biases;
        }
};

DenseLayerMatrix :: DenseLayerMatrix (int in_size, int out_size, shared_ptr<Activation> act)
{
    input_size = in_size;
    output_size = out_size;
    activation = act;

    W.resize(out_size, in_size);
    b.resize(out_size);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-0.5, 0.5);

    for (int i=0; i<out_size; i++)
    {
        b(i) = dis(gen);
        for (int j=0; j<in_size; j++)
        {
            W(i, j) = dis(gen);
        }
    }
}

MatrixXd DenseLayerMatrix :: forward (const MatrixXd& X)
{
    //batch size
    int batch_size = X.rows(); //(batch_size × input_size)
    //W^T is (input_size × output_size)

    //cache input
    last_input = X;

    //compute z = x*w^t
    MatrixXd Z(batch_size, output_size);
    Z = X * W.transpose();

    //add bias to each row
    for (int i=0; i<batch_size; i++)
    {
        for (int j=0; j<output_size; j++)
        {
            Z(i, j) += b(j);
        }
    }

    //cache z
    last_z = Z;

    //apply activation
    MatrixXd output(batch_size, output_size);
    for (int i=0; i<batch_size; i++)
    {
        for (int j=0; j<output_size; j++)
        {
            output(i, j) = activation->apply(Z(i, j));
        }
    }

    //cache activated output
    last_a = output;

    return output;
}

MatrixXd DenseLayerMatrix :: backward(const MatrixXd& grad_output) //dL/dA (gradient of loss w.r.t. this layer's output)
{
    int batch_size = last_input.rows(); //last_input is cache of x

    //Compute dZ = grad_output ⊙ activation'(last_z)
    MatrixXd dZ(batch_size, output_size);
    //dZ = dL/dA ⊙ activation'(Z) — element-wise multiplication
    for (int i=0; i<batch_size; i++)
    {
        for (int j=0; j<output_size; j++)
        {
            dZ(i, j) = grad_output(i, j) * activation->derivative(last_z(i, j));
        }
    }

    //Compute dW = (dZ^T × last_input) / batch_size
    //dZ is (batch_size × output_size)
    //dZ^T is (output_size × batch_size)
    //last_input is (batch_size × input_size)
    //dZ^T × last_input is (output_size × input_size) - This is the shape of W
    dW = (dZ.transpose() * last_input) / batch_size;

    //Compute db = mean of each column of dZ
    db.resize(output_size);
    for (int j=0; j<output_size; j++)
    {
        double sum = 0.0;
        for (int i=0; i<batch_size; i++)
        {
            sum += dZ(i, j);
        }

        db(j) = sum / batch_size;
    }

    //Compute grad_input = dZ × W (to pass to previous layer)
    MatrixXd grad_input = dZ * W;

    return grad_input;
}

void DenseLayerMatrix :: update(double learning_rate)
{
    //W_new = W_old - learning_rate × dW
    W = W - learning_rate * dW;

    //b_new = b_old - learning_rate × db
    b = b - learning_rate * db;
}

#endif