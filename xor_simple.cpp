#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>
using namespace std;

// sigmoid function
double sigmoid (double z)
{
    return 1.0 / (1.0 + exp(-z));
}

// sigmoid derivative function
// What this does:
// When we do backpropagation (learning from mistakes), we need the derivative of sigmoid
// This tells us "how sensitive is the sigmoid output to changes in its input"
// The formula is: s × (1 - s) where s is the sigmoid output we already computed
double dsigmoid_from_output (double s)
{
    return s * (1.0 - s);
}

int main()
{
    //xor training data
    vector<vector<double>> X =
    {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    vector<double> Y = {0.0, 1.0, 1.0, 0.0};

    //random number generator for weights
    random_device rd; //random_device is a class in C++ that provides truly random numbers (not pseudo-random), usually obtained from your system’s hardware.
    mt19937 gen(rd()); //We’re creating an object gen of type mt19937.
                       //It is initialized (seeded) using the output of rd() — so that the random sequence starts from a different point every time the program runs.
    uniform_real_distribution<> dis(-0.5, 0.5); //uniform_real_distribution is a class template that defines a range of real (floating-point) numbers for random generation.


    // Hidden layer weights (2 neurons, each takes 2 inputs)
    double w11 = dis(gen);
    double w12 = dis(gen);
    double w21 = dis(gen);
    double w22 = dis(gen);


    // Hidden layer biases (one for each neuron)
    double b1 = dis(gen);
    double b2 = dis(gen);


    // Output layer weights (1 neuron takes 2 inputs from hidden layer)
    double v1 = dis(gen);
    double v2 = dis(gen);

    // Output layer bias
    double bo = dis(gen);


    // Training parameters
    double lr = 2.0; // learning rate
    int epochs = 5000;


    cout << "Starting Training.........." << endl;

    for (int epoch=0; epoch<epochs; epoch++)
    {
        // Variables to accumulate gradients across all samples
        double sum_dW11 = 0.0, sum_dW12 = 0.0;
        double sum_dW21 = 0.0, sum_dW22 = 0.0;
        double sum_db1 = 0.0, sum_db2 = 0.0;
        double sum_dV1 = 0.0, sum_dV2 = 0.0;
        double sum_dbo = 0.0;

        double total_loss = 0.0;

        // Loop through all 4 training examples
        for (int i=0; i<4; i++)
        {
            double x1 = X[i][0];
            double x2 = X[i][1];
            double y_true = Y[i];

            // Forward pass: Hidden layer
            double z1 = x1 * w11 + x2 * w12 + b1;
            double z2 = x1 * w21 + x2 * w22 + b2;

            // calculating activation for each z1 result
            double a1 = sigmoid(z1);
            double a2 = sigmoid(z2);

            // Forward pass: Output layer
            double z_out = a1 * v1 + a2 * v2 + bo;
            double y_pred = sigmoid(z_out);

            // Compute loss for this sample
            double loss_sample = 0.5 * (y_pred - y_true) * (y_pred - y_true);
            total_loss += loss_sample;

            // Backward pass: Output layer
            double delta_out =  (y_pred - y_true) * dsigmoid_from_output(y_pred); // calculate how wrong the outut was

            // Gradients for output weights
            double dV1 = delta_out * a1; // how much should v1 change (depends on a1's value)
            double dV2 = delta_out * a2; // how much should v2 change (depends on a2's value)
            double dbo = delta_out; // how much should bo change (just the delta)

            // Propagate error back to hidden layer
            double delta_hidden1 = v1 * delta_out * dsigmoid_from_output(a1); //error signal for hidden neuron 1
            double delta_hidden2 = v2 * delta_out * dsigmoid_from_output(a2);

            // Gradients for hidden layer weights
            double dW11 = delta_hidden1 * x1;
            double dW12 = delta_hidden1 * x2;
            double dW21 = delta_hidden2 * x1;
            double dW22 = delta_hidden2 * x2;
            double db1 = delta_hidden1;
            double db2 = delta_hidden2;

            // Accumulate gradients
            sum_dW11 += dW11;
            sum_dW12 += dW12;
            sum_dW21 += dW21;
            sum_dW22 += dW22;
            sum_db1 += db1;
            sum_db2 += db2;
            sum_dV1 += dV1;
            sum_dV2 += dV2;
            sum_dbo += dbo;
        }

        // Average loss and gradients over all 4 samples
            total_loss /= 4.0;
            sum_dW11 /= 4.0;
            sum_dW12 /= 4.0;
            sum_dW21 /= 4.0;
            sum_dW22 /= 4.0;
            sum_db1 /= 4.0;
            sum_db2 /= 4.0;
            sum_dV1 /= 4.0;
            sum_dV2 /= 4.0;
            sum_dbo /= 4.0;

            // Update weights using gradient descent
            w11 -= lr * sum_dW11;
            w12 -= lr * sum_dW12;
            w21 -= lr * sum_dW21;
            w22 -= lr * sum_dW22;
            b1 -= lr * sum_db1;
            b2 -= lr * sum_db2;
            v1 -= lr * sum_dV1;
            v2 -= lr * sum_dV2;
            bo -= lr * sum_dbo;

            // Print progress every 1000 epochs
            if ((epoch + 1) % 1000 == 0)
            {
                cout << "Epoch " << epoch + 1 << " | Loss: " << total_loss << endl;
            }
    }

    // Test the trained network
    cout << "\nTraining complete! Testing predictions:\n" << endl;
    cout << "Input  | Target | Prediction" << endl;
    cout << "-------|--------|------------" << endl;

    for (int i=0; i<4; i++)
    {
        double x1 = X[i][0];
        double x2 = X[i][1];
        double y_true = Y[i];

        // Forward pass:
        double z1 = w11 * x1 + w12 * x2 + b1;
        double z2 = w21 * x1 + w22 * x2 + b2;
        double a1 = sigmoid(z1);
        double a2 = sigmoid(z2);
        double z_out = v1 * a1 + v2 * a2 + bo;
        double y_pred = sigmoid(z_out);
        
        cout << x1 << " " << x2 << "  |   " << y_true << "    | " << y_pred << endl;
    }
}