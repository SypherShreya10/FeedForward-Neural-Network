#ifndef NETWORK_MATRIX_H
#define NETWORK_MATRIX_H

#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "layer_matrix.h"
#include "loss_matrix.h"

using namespace std;
using namespace Eigen;

class NetworkMatrix
{
    private:
        vector<shared_ptr<DenseLayerMatrix>> layers;
        VectorXd norm_min;
        VectorXd norm_range;
        vector<string> class_labels;
        int input_size = 0;
        int output_size = 0;

    public:
        NetworkMatrix();
        void add_layer(shared_ptr<DenseLayerMatrix> layer);
        MatrixXd forward(const MatrixXd& X);
        void backward(const MatrixXd& grad_output);
        void update(double learning_rate);

        int get_num_layers() const {return layers.size();};

        void save(const string& filename) const;
        bool load(const string& filename);

        void set_normalization(const VectorXd& mn, const VectorXd& rng)
        {
            norm_min = mn;
            norm_range = rng;
        }

        void set_class_labels(const vector<string>& labels)
        {
            class_labels = labels;
        }

        VectorXd get_norm_min() const { return norm_min; }
        VectorXd get_norm_range() const { return norm_range; }
        vector<string> get_class_labels() const { return class_labels; }

};

NetworkMatrix :: NetworkMatrix()
{

}

void NetworkMatrix::add_layer(shared_ptr<DenseLayerMatrix> layer)
{
    layers.push_back(layer);
    cout << "Added layer: " << layer->get_input_size() 
         << " -> " << layer->get_output_size() << endl;
}

MatrixXd NetworkMatrix :: forward(const MatrixXd& X)
{
    MatrixXd output = X;

    for (size_t i=0; i<layers.size(); i++)
    {
        output = layers[i]->forward(output);
    }

    return output;
}

void NetworkMatrix::backward(const MatrixXd& grad_output)
{
    MatrixXd grad = grad_output;  // Start with gradient from loss
    
    // Go through layers in REVERSE order
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        grad = layers[i]->backward(grad);
    }
}

void NetworkMatrix::update(double learning_rate)
{
    // Update all layers with the same learning rate
    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i]->update(learning_rate);
    }
}

// void NetworkMatrix::save(const string& filename) const
// {
//     ofstream file(filename, ios::binary);
    
//     if (!file.is_open())
//     {
//         cout << "Error: Could not create file " << filename << endl;
//         return;
//     }
    
//     // Save number of layers
//     int num_layers = layers.size();
//     file.write((char*)&num_layers, sizeof(int));
    
//     cout << "Saving " << num_layers << " layers to " << filename << endl;
    
//     // Save each layer
//     for (int i = 0; i < num_layers; i++)
//     {
//         // Get layer dimensions
//         int in_size = layers[i]->get_input_size();
//         int out_size = layers[i]->get_output_size();
        
//         // Save dimensions
//         file.write((char*)&in_size, sizeof(int));
//         file.write((char*)&out_size, sizeof(int));
        
//         // Get weights and biases
//         MatrixXd W = layers[i]->get_weights();
//         VectorXd b = layers[i]->get_biases();
        
//         // Save weights
//         for (int row = 0; row < out_size; row++)
//         {
//             for (int col = 0; col < in_size; col++)
//             {
//                 double value = W(row, col);
//                 file.write((char*)&value, sizeof(double));
//             }
//         }
        
//         // Save biases
//         for (int j = 0; j < out_size; j++)
//         {
//             double value = b(j);
//             file.write((char*)&value, sizeof(double));
//         }
        
//         cout << "  Layer " << i << ": " << in_size << " -> " << out_size << endl;
//     }
    
//     file.close();
//     cout << "Model saved successfully!" << endl;
// }

void NetworkMatrix::save(const string& filename) const
{
    ofstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cout << "Error: Could not create file " << filename << endl;
        return;
    }

    // 1. Save input/output sizes
    int in_size = layers[0]->get_input_size();
    int out_size = layers.back()->get_output_size();
    file.write((char*)&in_size, sizeof(int));
    file.write((char*)&out_size, sizeof(int));

    // 2. Save normalization vectors
    int f = norm_min.size();
    file.write((char*)&f, sizeof(int));
    for (int i = 0; i < f; i++) file.write((char*)&norm_min(i), sizeof(double));
    for (int i = 0; i < f; i++) file.write((char*)&norm_range(i), sizeof(double));

    // 3. Save class labels
    int num_labels = class_labels.size();
    file.write((char*)&num_labels, sizeof(int));
    for (const auto& s : class_labels)
    {
        int len = s.size();
        file.write((char*)&len, sizeof(int));
        file.write(s.c_str(), len);
    }

    // 4. Save layers
    int num_layers = layers.size();
    file.write((char*)&num_layers, sizeof(int));

    for (int i = 0; i < num_layers; i++)
    {
        int in = layers[i]->get_input_size();
        int out = layers[i]->get_output_size();
        file.write((char*)&in, sizeof(int));
        file.write((char*)&out, sizeof(int));

        MatrixXd W = layers[i]->get_weights();
        VectorXd b = layers[i]->get_biases();

        for (int r = 0; r < out; r++)
            for (int c = 0; c < in; c++)
                file.write((char*)&W(r,c), sizeof(double));

        for (int r = 0; r < out; r++)
            file.write((char*)&b(r), sizeof(double));
    }

    cout << "Model saved successfully!\n";
}

// bool NetworkMatrix::load(const string& filename)
// {
//     ifstream file(filename, ios::binary);
    
//     if (!file.is_open())
//     {
//         cout << "Error: Could not open file " << filename << endl;
//         return false;
//     }
    
//     // Clear existing layers
//     layers.clear();
    
//     // Read number of layers
//     int num_layers;
//     file.read((char*)&num_layers, sizeof(int));
    
//     cout << "Loading " << num_layers << " layers from " << filename << endl;
    
//     // Load each layer
//     for (int i = 0; i < num_layers; i++)
//     {
//         // Read dimensions
//         int in_size, out_size;
//         file.read((char*)&in_size, sizeof(int));
//         file.read((char*)&out_size, sizeof(int));
        
//         // Create new layer (with random weights initially)
//         shared_ptr<Activation> sigmoid = make_shared<Sigmoid>();
//         shared_ptr<DenseLayerMatrix> layer = make_shared<DenseLayerMatrix>(in_size, out_size, sigmoid);
        
//         // Read weights
//         MatrixXd W(out_size, in_size);
//         for (int row = 0; row < out_size; row++)
//         {
//             for (int col = 0; col < in_size; col++)
//             {
//                 double value;
//                 file.read((char*)&value, sizeof(double));
//                 W(row, col) = value;
//             }
//         }
        
//         // Read biases
//         VectorXd b(out_size);
//         for (int j = 0; j < out_size; j++)
//         {
//             double value;
//             file.read((char*)&value, sizeof(double));
//             b(j) = value;
//         }
        
//         // Set loaded weights
//         layer->set_weights(W, b);
        
//         // Add layer to network
//         layers.push_back(layer);
        
//         cout << "  Layer " << i << ": " << in_size << " -> " << out_size << endl;
//     }
    
//     file.close();
//     cout << "Model loaded successfully!" << endl;
//     return true;
// }

bool NetworkMatrix::load(const string& filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cout << "Error: Could not open file!" << endl;
        return false;
    }

    layers.clear();

    // 1. Input / Output sizes
    file.read((char*)&input_size, sizeof(int));
    file.read((char*)&output_size, sizeof(int));

    // 2. Normalization vectors
    int f;
    file.read((char*)&f, sizeof(int));
    norm_min = VectorXd(f);
    norm_range = VectorXd(f);

    for (int i = 0; i < f; i++) file.read((char*)&norm_min(i), sizeof(double));
    for (int i = 0; i < f; i++) file.read((char*)&norm_range(i), sizeof(double));

    // 3. Class labels
    int num_labels;
    file.read((char*)&num_labels, sizeof(int));

    class_labels.clear();
    for (int i = 0; i < num_labels; i++)
    {
        int len;
        file.read((char*)&len, sizeof(int));
        string s(len, ' ');
        file.read(&s[0], len);
        class_labels.push_back(s);
    }

    // 4. Layers
    int num_layers;
    file.read((char*)&num_layers, sizeof(int));

    for (int i = 0; i < num_layers; i++)
    {
        int in, out;
        file.read((char*)&in, sizeof(int));
        file.read((char*)&out, sizeof(int));

        auto act = make_shared<Sigmoid>(); // default
        auto layer = make_shared<DenseLayerMatrix>(in, out, act);

        MatrixXd W(out, in);
        VectorXd b(out);

        for (int r = 0; r < out; r++)
            for (int c = 0; c < in; c++)
                file.read((char*)&W(r,c), sizeof(double));

        for (int r = 0; r < out; r++)
            file.read((char*)&b(r), sizeof(double));

        layer->set_weights(W, b);
        layers.push_back(layer);
    }

    cout << "Model loaded.\n";
    return true;
}

#endif