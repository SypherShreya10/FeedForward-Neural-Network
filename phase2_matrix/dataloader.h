#ifndef DATALOADER_H
#define DATALOADER_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iostream>
#include <map>

using namespace std;
using namespace Eigen;

class DataLoader
{
private:
    MatrixXd X_data;
    MatrixXd Y_data;
    
    MatrixXd X_train, Y_train;
    MatrixXd X_val, Y_val;

    int batch_size;
    int current_index;

    int num_features;
    int num_classes;
    map<string, int> label_mapping;
    vector<string> class_names;   // NEW: index → label name

    void shuffle_all_data();
    
    // Helper method - we'll implement this later
    void load_and_process(const string& filename,
                         const vector<int>& feature_cols,
                         int label_col,
                         double train_ratio);

    VectorXd min_vals;
    VectorXd ranges;


public:
    // Constructor 1: Auto-detect columns
    DataLoader(string filename, double train_ratio, int batch_size);
    
    // Constructor 2: User specifies columns  
    DataLoader(string filename, 
               const vector<int>& feature_cols,
               int label_col,
               double train_ratio,
               int batch_size);
    
    // Getters
    int get_num_features() const { return num_features; }
    int get_num_classes() const { return num_classes; }
    
    void normalise();
    void shuffle();

    MatrixXd get_train_x() { return X_train; }
    MatrixXd get_train_y() { return Y_train; }
    MatrixXd get_val_x() { return X_val; }
    MatrixXd get_val_y() { return Y_val; }

    int get_num_train_samples() { return X_train.rows(); }
    int get_num_val_samples() { return X_val.rows(); }

    bool get_next_batch(MatrixXd& X_batch, MatrixXd& Y_batch);
    void reset();
    int get_num_batches();

    RowVectorXd normalize_single(const RowVectorXd& x);
    vector<string> get_class_names() const { return class_names; }

    VectorXd get_min_vals() const { return min_vals; }
    VectorXd get_ranges() const { return ranges; }
};

// Constructor 1: Auto-detect
DataLoader::DataLoader(string filename, double train_ratio, int batch_size)
{
    this->batch_size = batch_size;
    this->current_index = 0;
    
    // Read entire CSV to detect structure
    vector<vector<string>> raw_data;
    
    ifstream file(filename);
    if (!file.is_open())
    {
        cout << "Error: Could not open file!" << endl;
        return;
    }
    
    string line;
    while (getline(file, line))
    {
        stringstream ss(line);
        string token;
        vector<string> row;
        
        while (getline(ss, token, ','))
        {
            row.push_back(token);
        }
        
        if (!row.empty())
            raw_data.push_back(row);
    }
    
    file.close();
    
    if (raw_data.empty())
    {
        cout << "Error: No data loaded!" << endl;
        return;
    }
    
    int total_cols = raw_data[0].size();
    cout << "Loaded " << raw_data.size() << " samples with " 
         << total_cols << " columns" << endl;
    
    // Auto-detect: all columns except last are features
    vector<int> feature_cols;
    for (int i = 0; i < total_cols - 1; i++)
        feature_cols.push_back(i);
    
    int label_col = total_cols - 1;
    
    cout << "Auto-detected configuration:" << endl;
    cout << "  Feature columns: 0 to " << (total_cols - 2) << endl;
    cout << "  Label column: " << label_col << endl;
    
    // Call helper to process the data
    load_and_process(filename, feature_cols, label_col, train_ratio);
}

// Constructor 2: User-specified columns
DataLoader::DataLoader(string filename,
                       const vector<int>& feature_cols,
                       int label_col,
                       double train_ratio,
                       int batch_size)
{
    this->batch_size = batch_size;
    this->current_index = 0;
    
    cout << "Using user-specified configuration:" << endl;
    cout << "  Feature columns: ";
    for (int col : feature_cols)
        cout << col << " ";
    cout << endl;
    cout << "  Label column: " << label_col << endl;
    
    // Call helper to process the data
    load_and_process(filename, feature_cols, label_col, train_ratio);
}

// Helper method that does the actual loading and processing
void DataLoader::load_and_process(const string& filename,
                                  const vector<int>& feature_cols,
                                  int label_col,
                                  double train_ratio)
{
    num_features = feature_cols.size();
    
    // Step 1: Read entire CSV
    vector<vector<string>> raw_data;
    
    ifstream file(filename);
    if (!file.is_open())
    {
        cout << "Error: Could not open file!" << endl;
        return;
    }
    
    string line;
    while (getline(file, line))
    {
        stringstream ss(line);
        string token;
        vector<string> row;
        
        while (getline(ss, token, ','))
        {
            row.push_back(token);
        }
        
        if (!row.empty())
            raw_data.push_back(row);
    }
    file.close();
    
    int num_samples = raw_data.size();
    cout << "Processing " << num_samples << " samples..." << endl;
    
    // Step 2: Extract features
    vector<vector<double>> features_list;
    
    for (int i = 0; i < num_samples; i++)
    {
        vector<double> row_features;
        
        for (int col : feature_cols)
        {
            try
            {
                double value = stod(raw_data[i][col]);
                row_features.push_back(value);
            }
            catch (...)
            {
                cout << "Warning: Invalid value at row " << i 
                     << ", col " << col << ". Using 0.0" << endl;
                row_features.push_back(0.0);
            }
        }
        
        features_list.push_back(row_features);
    }
    
    // Step 3: Extract and encode labels
    vector<string> unique_labels;
    vector<int> labels_list;
    
    for (int i = 0; i < num_samples; i++)
    {
        string label_str = raw_data[i][label_col];
        
        auto it = find(unique_labels.begin(), unique_labels.end(), label_str);
        
        int label_idx;
        if (it == unique_labels.end())
        {
            label_idx = unique_labels.size();
            unique_labels.push_back(label_str);
            label_mapping[label_str] = label_idx;
        }
        else
        {
            label_idx = distance(unique_labels.begin(), it);
        }
        
        labels_list.push_back(label_idx);
    }
    
    num_classes = unique_labels.size();
    class_names = unique_labels;   // NEW: store labels for prediction output
    
    cout << "Detected " << num_classes << " classes: ";
    for (const auto& label : unique_labels)
        cout << label << " ";
    cout << endl;
    
    // Step 4: Convert to Eigen matrices
    X_data = MatrixXd(num_samples, num_features);
    
    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            X_data(i, j) = features_list[i][j];
        }
    }
    
    Y_data = MatrixXd::Zero(num_samples, num_classes);
    
    for (int i = 0; i < num_samples; i++)
    {
        int label = labels_list[i];
        if (label >= 0 && label < num_classes)
        {
            Y_data(i, label) = 1.0;
        }
    }
    
    cout << "X_data shape: (" << X_data.rows() << ", " << X_data.cols() << ")" << endl;
    cout << "Y_data shape: (" << Y_data.rows() << ", " << Y_data.cols() << ")" << endl;
    
    // Step 5: Shuffle and split
    shuffle_all_data();
    
    int train_size = (int)(num_samples * train_ratio);
    
    X_train = X_data.topRows(train_size);
    Y_train = Y_data.topRows(train_size);
    X_val = X_data.bottomRows(num_samples - train_size);
    Y_val = Y_data.bottomRows(num_samples - train_size);
    
    cout << "Training samples: " << X_train.rows() << endl;
    cout << "Validation samples: " << X_val.rows() << endl;
}

void DataLoader::shuffle_all_data()
{
    int num_samples = X_data.rows();
    
    vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; i++)
        indices[i] = i;
    
    random_device rd;
    mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    MatrixXd X_shuffled(num_samples, X_data.cols());
    MatrixXd Y_shuffled(num_samples, Y_data.cols());
    
    for (int i = 0; i < num_samples; i++)
    {
        int old_index = indices[i];
        X_shuffled.row(i) = X_data.row(old_index);
        Y_shuffled.row(i) = Y_data.row(old_index);
    }
    
    X_data = X_shuffled;
    Y_data = Y_shuffled;
}

// void DataLoader::normalise()
// {
//     int num_features = X_train.cols();
    
//     for (int j = 0; j < num_features; j++)
//     {
//         double min_val = X_train.col(j).minCoeff();
//         double max_val = X_train.col(j).maxCoeff();
//         double range = (max_val - min_val > 1e-10) ? (max_val - min_val) : 1.0;
        
//         for (int i = 0; i < X_train.rows(); i++)
//         {
//             X_train(i, j) = (X_train(i, j) - min_val) / range;
//         }
        
//         for (int i = 0; i < X_val.rows(); i++)
//         {
//             X_val(i, j) = (X_val(i, j) - min_val) / range;
//         }
//     }
    
//     cout << "Data normalized!" << endl;
// }


void DataLoader::normalise()
{
    int f = X_train.cols();

    min_vals = VectorXd(f);
    ranges   = VectorXd(f);

    for (int j = 0; j < f; j++)
    {
        double mn = X_train.col(j).minCoeff();
        double mx = X_train.col(j).maxCoeff();
        double range = (mx - mn > 1e-10) ? (mx - mn) : 1.0;

        min_vals(j) = mn;
        ranges(j) = range;

        // normalize training
        for (int i = 0; i < X_train.rows(); i++)
            X_train(i, j) = (X_train(i, j) - mn) / range;

        // normalize validation
        for (int i = 0; i < X_val.rows(); i++)
            X_val(i, j) = (X_val(i, j) - mn) / range;
    }

    cout << "Data normalized!" << endl;
}


void DataLoader::shuffle()
{
    int num_samples = X_train.rows();
    
    vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; i++)
        indices[i] = i;
    
    random_device rd;
    mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    MatrixXd X_shuffled(num_samples, X_train.cols());
    MatrixXd Y_shuffled(num_samples, Y_train.cols());
    
    for (int i = 0; i < num_samples; i++)
    {
        int old_index = indices[i];
        X_shuffled.row(i) = X_train.row(old_index);
        Y_shuffled.row(i) = Y_train.row(old_index);
    }
    
    X_train = X_shuffled;
    Y_train = Y_shuffled;
}

bool DataLoader::get_next_batch(MatrixXd& X_batch, MatrixXd& Y_batch)
{
    int num_train = X_train.rows();
    
    if (current_index >= num_train)
        return false;
    
    int start = current_index;
    int end = min(start + batch_size, num_train);
    int actual_batch_size = end - start;
    
    X_batch = X_train.block(start, 0, actual_batch_size, X_train.cols());
    Y_batch = Y_train.block(start, 0, actual_batch_size, Y_train.cols());
    
    current_index += batch_size;
    
    return true;
}

void DataLoader::reset()
{
    current_index = 0;
}

int DataLoader::get_num_batches()
{
    int num_train = X_train.rows();
    return (num_train + batch_size - 1) / batch_size;
}

RowVectorXd DataLoader :: normalize_single(const RowVectorXd& x)
{
    RowVectorXd out = x;

    for (int j = 0; j < x.size(); j++)
    {
        out(j) = (x(j) - min_vals(j)) / ranges(j);
    }

    return out;
}


#endif