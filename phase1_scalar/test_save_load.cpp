#include <iostream>
#include "activation.h"
#include "layer_matrix.h"
#include "network.h"
#include "dataloader.h"
#include "loss_matrix.h"

using namespace std;

int main()
{
    cout << "=== Testing Save/Load ===" << endl << endl;
    
    // 1. Create and train a network
    cout << "Creating network..." << endl;
    NetworkMatrix net;
    
    shared_ptr<Activation> sigmoid = make_shared<Sigmoid>();
    net.add_layer(make_shared<DenseLayerMatrix>(4, 8, sigmoid));
    net.add_layer(make_shared<DenseLayerMatrix>(8, 3, sigmoid));
    
    // Load data
    DataLoader loader("iris_dataset.csv", 0.8, 32);
    loader.normalise();
    
    // Quick training (just a few iterations)
    cout << "\nTraining for 10 epochs..." << endl;
    shared_ptr<LossMatrix> loss_fn = make_shared<MSEMatrix>();
    
    for (int epoch = 0; epoch < 500; epoch++)
    {
        loader.reset();
        MatrixXd X_batch, Y_batch;
        
        while (loader.get_next_batch(X_batch, Y_batch))
        {
            MatrixXd output = net.forward(X_batch);
            MatrixXd grad = loss_fn->gradient(output, Y_batch);
            net.backward(grad);
            net.update(0.5);
        }
    }
    
    // Make prediction before saving
    MatrixXd X_val = loader.get_val_x();
    MatrixXd pred_before = net.forward(X_val);
    cout << "Prediction before save (first sample): " 
         << pred_before(0, 0) << ", " 
         << pred_before(0, 1) << ", " 
         << pred_before(0, 2) << endl;
    
    // 2. Save the model
    cout << "\n";
    net.save("iris_model.bin");
    
    // 3. Create a NEW network and load
    cout << "\nCreating new network and loading..." << endl;
    NetworkMatrix net2;
    net2.load("iris_model.bin");
    
    // 4. Make prediction with loaded model
    MatrixXd pred_after = net2.forward(X_val);
    cout << "\nPrediction after load (first sample): " 
         << pred_after(0, 0) << ", " 
         << pred_after(0, 1) << ", " 
         << pred_after(0, 2) << endl;
    
    // 5. Compare
    cout << "\nAre predictions identical? ";
    bool identical = true;
    for (int i = 0; i < pred_before.rows(); i++)
    {
        for (int j = 0; j < pred_before.cols(); j++)
        {
            if (abs(pred_before(i, j) - pred_after(i, j)) > 1e-10)
            {
                identical = false;
                break;
            }
        }
    }
    
    cout << (identical ? "YES!" : "NO") << endl;
    
    return 0;
}