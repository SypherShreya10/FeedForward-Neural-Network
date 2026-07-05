#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include "activation.h"
#include "layer_matrix.h"
#include "loss_matrix.h"

using namespace std;
using namespace Eigen;

int main()
{
    // XOR dataset
    MatrixXd X(4, 2);
    X << 0.0, 0.0,
         0.0, 1.0,
         1.0, 0.0,
         1.0, 1.0;
    
    MatrixXd Y(4, 1);
    Y << 0.0,
         1.0,
         1.0,
         0.0;
    
    cout << "Training XOR with batch backpropagation\n\n";
    
    // Create network: 2 -> 4 -> 1
    shared_ptr<Activation> sigmoid = make_shared<Sigmoid>();
    DenseLayerMatrix layer1(2, 4, sigmoid);
    DenseLayerMatrix layer2(4, 1, sigmoid);
    
    // Create loss function
    shared_ptr<LossMatrix> loss_fn = make_shared<MSEMatrix>();
    
    // Training parameters
    double learning_rate = 0.5;
    int epochs = 10000;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        // Forward pass
        MatrixXd hidden = layer1.forward(X);
        MatrixXd output = layer2.forward(hidden);
        
        // Compute loss
        double loss = loss_fn->compute(output, Y);
        
        // Backward pass
        MatrixXd grad_output = loss_fn->gradient(output, Y);
        MatrixXd grad_hidden = layer2.backward(grad_output);
        MatrixXd grad_input = layer1.backward(grad_hidden);
        
        // Update weights
        layer2.update(learning_rate);
        layer1.update(learning_rate);
        
        // Print progress
        if (epoch % 500 == 0)
        {
            cout << "Epoch " << epoch << " - Loss: " << loss << endl;
        }
    }
    
    // Final predictions
    cout << "\nFinal predictions:\n";
    MatrixXd hidden = layer1.forward(X);
    MatrixXd final_output = layer2.forward(hidden);
    
    for (int i = 0; i < 4; i++)
    {
        cout << X(i, 0) << ", " << X(i, 1) << " -> " 
             << final_output(i, 0) << " (target: " << Y(i, 0) << ")\n";
    }
    
    return 0;
}