// 1. **Column 0 (first):** `sepal_length` - Length of the sepal in cm
// 2. **Column 1 (second):** `sepal_width` - Width of the sepal in cm
// 3. **Column 2 (third):** `petal_length` - Length of the petal in cm
// 4. **Column 3 (fourth):** `petal_width` - Width of the petal in cm
// 5. **Column 4 (fifth):** `species` - The flower species name

#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include "activation.h"
#include "layer_matrix.h"
#include "loss_matrix.h"
#include "dataloader.h"

using namespace std;
using namespace Eigen;

// Helper function to calculate accuracy
double calculate_accuracy(const MatrixXd& predictions, const MatrixXd& targets)
{
    int correct = 0; // count to check how many predictions were right
    int total = predictions.rows(); // total number of prediction samples (no of samples in validation set)
    
    for (int i = 0; i < total; i++) // we are checking of each of the sample in the validation set, how correct the predictions are
    {
        // Find predicted class (index of max value)
        int pred_class = 0;
        double max_val = predictions(i, 0);
        // finding which class the network predicted with highest confidence.
        for (int j = 1; j < predictions.cols(); j++)
        {
            if (predictions(i, j) > max_val)
            {
                max_val = predictions(i, j);
                pred_class = j;
            }
        }
        
        // Find true class (index of 1 in one-hot encoding)
        int true_class = 0;
        for (int j = 0; j < targets.cols(); j++)
        {
            if (targets(i, j) == 1.0)
            {
                true_class = j;
                break;
            }
        }
        
        if (pred_class == true_class)
            correct++;
    }
    
    return (double)correct / total;
}

int main()
{
    cout << "=== Training Neural Network on Iris Dataset ===" << endl << endl;
    
    // Load data
    DataLoader loader("iris_dataset.csv", 0.8, 32);  // 80% train, 20% val, batch_size=32
    loader.normalise();
    
    cout << "Training samples: " << loader.get_num_train_samples() << endl;
    cout << "Validation samples: " << loader.get_num_val_samples() << endl;
    cout << "Batches per epoch: " << loader.get_num_batches() << endl << endl;
    
    // Create network: 4 -> 8 -> 3
    shared_ptr<Activation> sigmoid = make_shared<Sigmoid>();
    DenseLayerMatrix layer1(4, 8, sigmoid);  // Input layer: 4 features
    DenseLayerMatrix layer2(8, 3, sigmoid);  // Output layer: 3 classes
    
    // Create loss function
    shared_ptr<LossMatrix> loss_fn = make_shared<MSEMatrix>();
    
    // Training parameters
    double learning_rate = 0.5;
    int epochs = 350;
    
    // Get validation data (for evaluation)
    MatrixXd X_val = loader.get_val_x();
    MatrixXd Y_val = loader.get_val_y();
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        loader.shuffle();  // Shuffle training data each epoch
        loader.reset();    // Reset batch iterator
        
        double epoch_loss = 0.0;
        int num_batches = 0;
        
        // Mini-batch training
        MatrixXd X_batch, Y_batch;
        while (loader.get_next_batch(X_batch, Y_batch))
        {
            // Forward pass
            MatrixXd hidden = layer1.forward(X_batch);
            MatrixXd output = layer2.forward(hidden);
            
            // Compute loss
            double batch_loss = loss_fn->compute(output, Y_batch);
            epoch_loss += batch_loss;
            
            // Backward pass
            MatrixXd grad_output = loss_fn->gradient(output, Y_batch);
            MatrixXd grad_hidden = layer2.backward(grad_output);
            MatrixXd grad_input = layer1.backward(grad_hidden);
            
            // Update weights
            layer2.update(learning_rate);
            layer1.update(learning_rate);
            
            num_batches++;
        }
        
        epoch_loss /= num_batches;  // Average loss
        
        // Evaluate on validation set every 10 epochs
        if (epoch % 10 == 0)
        {
            MatrixXd val_hidden = layer1.forward(X_val);
            MatrixXd val_output = layer2.forward(val_hidden);
            
            double val_loss = loss_fn->compute(val_output, Y_val);
            double val_accuracy = calculate_accuracy(val_output, Y_val);
            
            cout << "Epoch " << epoch 
                 << " | Train Loss: " << epoch_loss 
                 << " | Val Loss: " << val_loss
                 << " | Val Accuracy: " << (val_accuracy * 100) << "%" << endl;
        }
    }
    
    // Final evaluation
    cout << "\n=== Final Evaluation ===" << endl;
    MatrixXd val_hidden = layer1.forward(X_val);
    MatrixXd val_output = layer2.forward(val_hidden);
    double final_accuracy = calculate_accuracy(val_output, Y_val);
    cout << "Final Validation Accuracy: " << (final_accuracy * 100) << "%" << endl;
    
    return 0;
}