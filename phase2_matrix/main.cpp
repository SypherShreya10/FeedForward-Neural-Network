// main.cpp
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <filesystem>
#include <limits>
#include <windows.h>
#include <algorithm>

#include "dataloader.h"
#include "network.h"
#include "activation.h"
#include "loss_matrix.h"

using namespace std;
using namespace Eigen;
// namespace fs = std::filesystem;

// ---------- helpers ----------
int argmax(const Eigen::VectorXd &v)
{
    int index;
    v.maxCoeff(&index);
    return index;
}

double compute_accuracy(NetworkMatrix &net, DataLoader &loader)
{
    MatrixXd X = loader.get_val_x();
    MatrixXd Y = loader.get_val_y();

    MatrixXd pred = net.forward(X);

    int correct = 0;
    for (int i = 0; i < pred.rows(); i++)
    {
        int p = argmax(pred.row(i));
        int t = argmax(Y.row(i));
        if (p == t) correct++;
    }
    return (double)correct / pred.rows();
}

// vector<string> list_bin_models(const string &dir = ".")
// {
//     vector<string> out;
//     for (auto &p : fs::directory_iterator(dir))
//     {
//         if (!p.is_regular_file()) continue;
//         string name = p.path().filename().string();
//         if (name.size() >= 4 && name.substr(name.size()-4) == ".bin")
//             out.push_back(name);
//     }
//     sort(out.begin(), out.end());
//     return out;
// }

vector<string> list_bin_models(const string &folder)
{
    vector<string> files;

    string search_path = folder + "\\*.bin";

    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(search_path.c_str(), &fd);

    if (hFind == INVALID_HANDLE_VALUE)
        return files;

    do
    {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
        {
            files.push_back(fd.cFileName);
        }

    } while (FindNextFileA(hFind, &fd));

    FindClose(hFind);

    // Optional: sort alphabetically
    sort(files.begin(), files.end());

    return files;
}

// parse a line of numbers (space or comma separated). returns empty vector if 'exit' detected.
bool parse_feature_line(const string &line, vector<double> &vals)
{
    vals.clear();
    // trim leading/trailing
    auto s = line;
    auto trim = [](string &t) {
        while (!t.empty() && isspace((unsigned char)t.front())) t.erase(t.begin());
        while (!t.empty() && isspace((unsigned char)t.back())) t.pop_back();
    };
    trim(s);
    if (s == "exit" || s == "EXIT" || s == "Exit") return false;

    // allow comma or spaces
    string tmp;
    for (char c : s)
    {
        if (c == ',') tmp.push_back(' ');
        else tmp.push_back(c);
    }
    stringstream ss(tmp);
    double x;
    while (ss >> x)
    {
        vals.push_back(x);
    }
    return true;
}

// normalize user vector using model normalization (norm_min, norm_range)
RowVectorXd normalize_with_model(const vector<double> &vals, const VectorXd &norm_min, const VectorXd &norm_range)
{
    int n = (int)vals.size();
    RowVectorXd out(n);
    for (int i = 0; i < n; ++i)
    {
        double mn = norm_min.size() > i ? norm_min(i) : 0.0;
        double rg = (norm_range.size() > i && abs(norm_range(i)) > 1e-12) ? norm_range(i) : 1.0;
        out(i) = (vals[i] - mn) / rg;
    }
    return out;
}

// ---------- main ----------
int main()
{
    cout << "==============================\n";
    cout << "   Neural Network Simulator\n";
    cout << "==============================\n\n";

    while (true) // outer loop to allow restart if wanted
    {
        cout << "Choose an option:\n";
        cout << "  1. Train a new model (provide dataset)\n";
        cout << "  2. Load existing saved model\n";
        cout << "  3. Exit\n";
        cout << "Enter choice (1/2/3): ";

        int choice_main = 0;
        if (!(cin >> choice_main)) { cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n'); continue; }
        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        if (choice_main == 3) {
            cout << "Exiting. Bye.\n";
            break;
        }

        NetworkMatrix net;
        vector<string> class_labels;
        int input_size = 0;

        if (choice_main == 1)
        {
            // TRAIN NEW MODEL FLOW
            string filename;
            cout << "\nEnter dataset file name (CSV): ";
            cin >> filename;

            double train_ratio = 0.8;
            cout << "Enter train split fraction (e.g. 0.8 for 80% train, default 0.8): ";
            if (!(cin >> train_ratio)) { train_ratio = 0.8; cin.clear(); }
            if (train_ratio <= 0.0 || train_ratio >= 1.0) train_ratio = 0.8;

            int batch_size = 32;
            cout << "Enter batch size (default 32): ";
            if (!(cin >> batch_size)) { batch_size = 32; cin.clear(); }
            if (batch_size <= 0) batch_size = 32;

            // auto/manual columns
            char auto_choice;
            cout << "Auto-detect feature/output columns? (y/n): ";
            cin >> auto_choice;

            DataLoader *loader = nullptr;
            if (auto_choice == 'y' || auto_choice == 'Y')
            {
                loader = new DataLoader(filename, train_ratio, batch_size);
            }
            else
            {
                int num_features;
                cout << "Enter number of feature columns: ";
                cin >> num_features;
                vector<int> feature_cols(num_features);
                cout << "Enter feature column indices (space separated): ";
                for (int i = 0; i < num_features; ++i) cin >> feature_cols[i];
                int label_col;
                cout << "Enter label column index: ";
                cin >> label_col;
                loader = new DataLoader(filename, feature_cols, label_col, train_ratio, batch_size);
            }

            loader->normalise();

            input_size = loader->get_num_features();
            int output_size = loader->get_num_classes();
            class_labels = loader->get_class_names();

            cout << "\n--- Dataset Analysis ---\n";
            cout << "Total Samples: " << loader->get_num_train_samples() + loader->get_num_val_samples() << endl;
            cout << "Features: " << input_size << endl;
            cout << "Classes: " << output_size << endl;

            // architecture
            int num_hidden;
            cout << "\nHow many hidden layers? (0 for none): ";
            cin >> num_hidden;
            if (num_hidden < 0) num_hidden = 0;
            vector<int> hidden_sizes(max(0, num_hidden));
            for (int i = 0; i < num_hidden; ++i)
            {
                cout << "Enter neurons in hidden layer " << (i+1) << ": ";
                cin >> hidden_sizes[i];
            }

            // learning params
            double learning_rate;
            cout << "\nEnter learning rate (e.g. 0.05): ";
            cin >> learning_rate;
            if (learning_rate <= 0) learning_rate = 0.01;

            int epochs;
            cout << "Enter number of epochs: ";
            cin >> epochs;
            if (epochs <= 0) epochs = 100;

            // build network
            shared_ptr<Activation> relu = make_shared<ReLu>();
            shared_ptr<Activation> sigmoid = make_shared<Sigmoid>();

            // handle case: zero hidden layers
            if (num_hidden == 0)
            {
                net.add_layer(make_shared<DenseLayerMatrix>(input_size, output_size, sigmoid));
            }
            else
            {
                // input -> first hidden
                net.add_layer(make_shared<DenseLayerMatrix>(input_size, hidden_sizes[0], relu));
                // hidden -> hidden
                for (int i = 1; i < num_hidden; ++i)
                    net.add_layer(make_shared<DenseLayerMatrix>(hidden_sizes[i-1], hidden_sizes[i], relu));
                // last hidden -> output
                net.add_layer(make_shared<DenseLayerMatrix>(hidden_sizes.back(), output_size, sigmoid));
            }

            // set normalization & labels on network so save() will include them later
            net.set_normalization(loader->get_min_vals(), loader->get_ranges());
            net.set_class_labels(class_labels);

            // training loop
            shared_ptr<LossMatrix> loss_fn = make_shared<MSEMatrix>();
            cout << "\nTraining started...\n";

            for (int epoch = 1; epoch <= epochs; ++epoch)
            {
                loader->reset();
                MatrixXd Xb, Yb;
                while (loader->get_next_batch(Xb, Yb))
                {
                    MatrixXd pred = net.forward(Xb);
                    MatrixXd grad = loss_fn->gradient(pred, Yb);
                    net.backward(grad);
                    net.update(learning_rate);
                }
                if (epoch % 10 == 0 || epoch == epochs)
                {
                    double acc = compute_accuracy(net, *loader);
                    cout << "Epoch " << epoch << " | Val Accuracy: " << acc << endl;
                }
            }

            double final_acc = compute_accuracy(net, *loader);
            cout << "\nFinal validation accuracy: " << final_acc << endl;

            // sample predictions
            cout << "\nSample Predictions:\n";
            MatrixXd Xv = loader->get_val_x();
            MatrixXd Yv = loader->get_val_y();
            for (int i = 0; i < min(5, (int)Xv.rows()); ++i)
            {
                MatrixXd o = net.forward(Xv.row(i));
                int t = argmax(Yv.row(i));
                int p = argmax(o.row(0));
                cout << "True: " << class_labels[t] << " | Predicted: " << class_labels[p] << endl;
            }

            // Save option
            cout << "\nSave model? (y/n): ";
            char save_choice;
            cin >> save_choice;
            if (save_choice == 'y' || save_choice == 'Y')
            {
                string save_name;
                cout << "Enter filename to save (e.g. iris_model.bin): ";
                cin >> save_name;
                net.save(save_name);
                cout << "Saved model as: " << save_name << "\n";
            }

            // free loader object
            delete loader;
        }
        else if (choice_main == 2)
        {
            // LOAD EXISTING MODEL
            auto models = list_bin_models(".");
            if (models.empty())
            {
                cout << "No .bin models found in current directory.\n";
                continue;
            }

            cout << "\nSaved models:\n";
            for (size_t i = 0; i < models.size(); ++i)
                cout << "  [" << i << "] " << models[i] << "\n";

            int sel = -1;
            cout << "Enter index of model to load: ";
            cin >> sel;
            if (sel < 0 || sel >= (int)models.size())
            {
                cout << "Invalid selection.\n";
                continue;
            }
            string file_to_load = models[sel];
            cout << "Loading " << file_to_load << " ...\n";
            if (!net.load(file_to_load))
            {
                cout << "Failed to load model.\n";
                continue;
            }
            // after load we can get class labels and normalization from net
            class_labels = net.get_class_labels();
            VectorXd norm_min = net.get_norm_min();
            VectorXd norm_range = net.get_norm_range();
            input_size = (int)norm_min.size(); // expected feature count
            cout << "Model loaded. Detected " << class_labels.size() << " classes and input size " << input_size << ".\n";
        }
        else
        {
            cout << "Invalid option. Returning to main menu.\n";
            continue;
        }

        // At this point either we trained a model (net already has normalization & labels set)
        // or we loaded a model. Enter prediction loop.
        cout << "\nEntering prediction loop. Type 'exit' to quit prediction loop and return to main menu.\n";
        // fetch normalization and class labels from net
        VectorXd norm_min = net.get_norm_min();
        VectorXd norm_range = net.get_norm_range();
        if (input_size == 0)
        {
            // try to infer input size from normalization or layer sizes
            if (norm_min.size() > 0) input_size = (int)norm_min.size();
            else if (net.get_num_layers() > 0) input_size = net.get_num_layers() > 0 ? net.get_norm_min().size() : 0;
        }
        // flush newline
        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        while (true)
        {
            cout << "\nEnter feature values (" << input_size << " numbers, comma or space separated) or 'exit':\n> ";
            string line;
            if (!std::getline(cin, line)) { break; }

            // trim
            auto trim = [](string &t) {
                while (!t.empty() && isspace((unsigned char)t.front())) t.erase(t.begin());
                while (!t.empty() && isspace((unsigned char)t.back())) t.pop_back();
            };
            trim(line);
            if (line.empty()) { continue; }
            // exit?
            if (line == "exit" || line == "EXIT" || line == "Exit")
            {
                cout << "Exiting prediction loop.\n";
                break;
            }

            vector<double> vals;
            bool ok = parse_feature_line(line, vals);
            if (!ok) { cout << "Exit command received.\n"; break; }

            if ((int)vals.size() != input_size)
            {
                cout << "Expected " << input_size << " values but got " << vals.size() << ". Please retry.\n";
                continue;
            }

            RowVectorXd norm_user = normalize_with_model(vals, norm_min, norm_range);
            MatrixXd X1 = norm_user;
            MatrixXd out = net.forward(X1);
            int pred = argmax(out.row(0));
            string label = (pred >= 0 && pred < (int)class_labels.size()) ? class_labels[pred] : to_string(pred);
            cout << "Predicted class: " << label << "\n";
        }

        // after prediction loop, return to top menu
        cout << "\nReturning to main menu...\n\n";
    } // outer while

    return 0;
}
