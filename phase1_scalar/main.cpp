#include <iostream>
#include <vector>
#include <memory>
#include "activation.h"
#include "layer.h"
#include "loss.h"
#include "network.h"

using namespace std;

int main()
{
    // XOR dataset
    vector<vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    vector<vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Create loss
    shared_ptr<Loss> loss = make_shared<MSE>(); //made a copy of the shared pointer object

    // Build network
    Network net(loss); //passed into the Network class

    // Hidden layer: 2 -> 4 with Sigmoid
    shared_ptr<Activation> sig = make_shared<Sigmoid>();
    shared_ptr<DenseLayer> l1 = make_shared<DenseLayer>(2, 4, sig);
    net.add_layer(l1);

    // Output layer: 4 -> 1 with Sigmoid
    shared_ptr<DenseLayer> l2 = make_shared<DenseLayer>(4, 1, sig);
    net.add_layer(l2);

    // Train for epochs
    double lr = 0.5;
    for (int epoch = 0; epoch < 2000; ++epoch)
    {
        net.train_epoch(inputs, targets, lr, 1, true);

        if (epoch % 200 == 0)
        {
            double total_loss = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                vector<double> p = net.predict(inputs[i]);
                for (size_t j = 0; j < p.size(); ++j)
                    total_loss += loss->value(p[j], targets[i][j]);
            }
            cout << "Epoch " << epoch << " loss: " << (total_loss / inputs.size()) << endl;
        }
    }

    // Print final predictions
    cout << "\nFinal predictions:\n";
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        vector<double> p = net.predict(inputs[i]);
        cout << inputs[i][0] << "," << inputs[i][1] << " -> " << p[0] << "\n";
    }

    return 0;
}
