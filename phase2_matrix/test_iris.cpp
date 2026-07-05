// #include <iostream>
// #include <Eigen/Dense>
// #include <memory>
// #include "activation.h"
// #include "layer_matrix.h"
// #include "loss_matrix.h"
// #include "dataloader.h"

// using namespace std;
// using namespace Eigen;

// string get_class_name(int class_idx)
// {
//     if (class_idx == 0) return "Iris-setosa";
//     if (class_idx == 1) return "Iris-versicolor";
//     if (class_idx == 2) return "Iris-virginica";
//     return "Unknown";
// }

// int get_predicted_class(const MatrixXd& output, int sample_idx)
// {
//     int pred_class = 0;
//     double max_val = output(sample_idx, 0);
    
//     for (int j = 1; j < output.cols(); j++)
//     {
//         if (output(sample_idx, j) > max_val)
//         {
//             max_val = output(sample_idx, j);
//             pred_class = j;
//         }
//     }
//     return pred_class;
// }

// int get_true_class(const MatrixXd& targets, int sample_idx)
// {
//     for (int j = 0; j < targets.cols(); j++)
//     {
//         if (targets(sample_idx, j) == 1.0)
//             return j;
//     }
//     return -1;
// }

// int main()
// {
//     cout << "=== Testing Neural Network on Iris Dataset ===" << endl << endl;
    
//     // Load data (same as training)
//     DataLoader loader("iris_dataset.csv", 0.8, 32);  // 80% train, 20% val, batch_size=32
//     loader.normalise();
    
//     // Create network with SAME architecture as training
//     shared_ptr<Activation> sigmoid = make_shared<Sigmoid>();
//     DenseLayerMatrix layer1(4, 8, sigmoid);
//     DenseLayerMatrix layer2(8, 3, sigmoid);
    
//     // TODO: In real scenario, you'd LOAD trained weights here
//     // For now, we'll train quickly
    
//     cout << "Training network..." << endl;
//     shared_ptr<LossMatrix> loss_fn = make_shared<MSEMatrix>();
//     double learning_rate = 0.5;
    
//     // Quick training (100 epochs)
//     for (int epoch = 0; epoch < 500; epoch++)
//     {
//         loader.shuffle();
//         loader.reset();
        
//         MatrixXd X_batch, Y_batch;
//         while (loader.get_next_batch(X_batch, Y_batch))
//         {
//             MatrixXd hidden = layer1.forward(X_batch);
//             MatrixXd output = layer2.forward(hidden);
            
//             MatrixXd grad_output = loss_fn->gradient(output, Y_batch);
//             MatrixXd grad_hidden = layer2.backward(grad_output);
//             layer1.backward(grad_hidden);
            
//             layer2.update(learning_rate);
//             layer1.update(learning_rate);
//         }
//     }
//     cout << "Training complete!" << endl << endl;
    
//     // Now test on validation set
//     MatrixXd X_val = loader.get_val_x();
//     MatrixXd Y_val = loader.get_val_y();
    
//     // Get predictions
//     MatrixXd val_hidden = layer1.forward(X_val);
//     MatrixXd val_output = layer2.forward(val_hidden);
    
//     // Show detailed results
//     cout << "=== Detailed Test Results ===" << endl;
//     cout << "Sample | True Class      | Predicted Class | Correct?" << endl;
//     cout << "-------|-----------------|-----------------|----------" << endl;
    
//     int correct = 0;
//     int total = X_val.rows();
    
//     for (int i = 0; i < total; i++)
//     {
//         int true_class = get_true_class(Y_val, i);
//         int pred_class = get_predicted_class(val_output, i);
        
//         bool is_correct = (true_class == pred_class);
//         if (is_correct) correct++;
        
//         cout << i+1 << "      | " 
//              << get_class_name(true_class) << " | "
//              << get_class_name(pred_class) << " | "
//              << (is_correct ? "YES" : "NO") << endl;
//     }
    
//     cout << "\n=== Summary ===" << endl;
//     cout << "Total samples: " << total << endl;
//     cout << "Correct predictions: " << correct << endl;
//     cout << "Incorrect predictions: " << (total - correct) << endl;
//     cout << "Accuracy: " << (100.0 * correct / total) << "%" << endl;
    
//     return 0;
// }


#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <iomanip>
#include "activation.h"
#include "layer_matrix.h"
#include "loss_matrix.h"
#include "dataloader.h"

using namespace std;
using namespace Eigen;

string get_class_name(int class_idx)
{
    if (class_idx == 0) return "Iris-setosa";
    if (class_idx == 1) return "Iris-versicolor";
    if (class_idx == 2) return "Iris-virginica";
    return "Unknown";
}

int main()
{
    cout << fixed << setprecision(2);
    cout << "=== Iris Species Predictor ===" << endl << endl;
    
    DataLoader loader("iris_dataset.csv", 0.8, 32);
    loader.normalise();
    
    shared_ptr<Activation> sigmoid = make_shared<Sigmoid>();
    DenseLayerMatrix layer1(4, 8, sigmoid);
    DenseLayerMatrix layer2(8, 3, sigmoid);
    
    cout << "Training network..." << endl;
    shared_ptr<LossMatrix> loss_fn = make_shared<MSEMatrix>();
    double learning_rate = 0.5;

    for (int epoch = 0; epoch < 500; epoch++)
    {
        loader.shuffle();
        loader.reset();
        MatrixXd X_batch, Y_batch;
        while (loader.get_next_batch(X_batch, Y_batch))
        {
            MatrixXd hidden = layer1.forward(X_batch);
            MatrixXd output = layer2.forward(hidden);
            MatrixXd grad_output = loss_fn->gradient(output, Y_batch);
            MatrixXd grad_hidden = layer2.backward(grad_output);
            layer1.backward(grad_hidden);
            layer2.update(learning_rate);
            layer1.update(learning_rate);
        }
    }
    cout << "Training complete!" << endl << endl;
    
    // Interactive prediction
    while (true)
    {
        cout << "\n--- Enter Flower Measurements ---" << endl;
        cout << "(Enter -1 for any value to quit)" << endl;
        
        double sepal_length, sepal_width, petal_length, petal_width;
        cout << "Sepal Length (cm): ";  cin >> sepal_length;  if (sepal_length < 0) break;
        cout << "Sepal Width (cm): ";   cin >> sepal_width;   if (sepal_width < 0) break;
        cout << "Petal Length (cm): ";  cin >> petal_length;  if (petal_length < 0) break;
        cout << "Petal Width (cm): ";   cin >> petal_width;   if (petal_width < 0) break;
        
        MatrixXd input(1, 4);
        input(0, 0) = (sepal_length - 4.3) / (7.9 - 4.3);
        input(0, 1) = (sepal_width - 2.0) / (4.4 - 2.0);
        input(0, 2) = (petal_length - 1.0) / (6.9 - 1.0);
        input(0, 3) = (petal_width - 0.1) / (2.5 - 0.1);
        
        MatrixXd hidden = layer1.forward(input);
        MatrixXd output = layer2.forward(hidden);
        
        int pred_class = 0;
        double max_conf = output(0, 0);
        for (int j = 1; j < 3; j++)
        {
            if (output(0, j) > max_conf)
            {
                max_conf = output(0, j);
                pred_class = j;
            }
        }
        
        cout << "\n--- Prediction ---" << endl;
        cout << "Predicted Species: " << get_class_name(pred_class) << endl;
        cout << "Confidence: " << (max_conf * 100) << "%" << endl;
        cout << "\nProbabilities:\n"
             << "  Setosa:     " << (output(0, 0) * 100) << "%\n"
             << "  Versicolor: " << (output(0, 1) * 100) << "%\n"
             << "  Virginica:  " << (output(0, 2) * 100) << "%\n";
    }
    
    cout << "\nGoodbye!" << endl;
    return 0;
}
