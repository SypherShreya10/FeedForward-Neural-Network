#include <iostream>
#include "activation.h"
#include "layer_matrix.h"
#include "network.h"
#include "dataloader.h"
#include "loss_matrix.h"

using namespace std;

int main()
{
    // Create network
    NetworkMatrix net;
    
    // Add layers
    shared_ptr<Activation> sigmoid = make_shared<Sigmoid>();
    net.add_layer(make_shared<DenseLayerMatrix>(4, 8, sigmoid));
    net.add_layer(make_shared<DenseLayerMatrix>(8, 3, sigmoid));
    
    cout << "Network has " << net.get_num_layers() << " layers" << endl;
    
    // Load Iris data
    DataLoader loader("iris_dataset.csv", 0.8, 32);
    loader.normalise();
    
    MatrixXd X = loader.get_train_x();
    MatrixXd Y = loader.get_train_y();
    
    // Test forward pass
    MatrixXd output = net.forward(X);
    cout << "Output shape: " << output.rows() << " x " << output.cols() << endl;
    
    // Test backward and update
    shared_ptr<LossMatrix> loss_fn = make_shared<MSEMatrix>();
    MatrixXd grad = loss_fn->gradient(output, Y);
    
    net.backward(grad);
    net.update(0.1);
    
    cout << "Network forward/backward/update test: SUCCESS!" << endl;
    
    return 0;
}